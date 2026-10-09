"""Real GPU numerical check of both engines, including editable phrase boundaries."""
import json
import numpy as np
import onnxruntime as ort
from types import SimpleNamespace
from pathlib import Path
import time

from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor, AutoProcessor

from dinov3_cases.grounding_trt import GroundingTRT
from dinov3_cases.trt_backend import ARTIFACTS, TRTBackbone, model_variant


@torch.inference_mode()
def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(__file__).resolve().parents[2]
    image = Image.open(root / 'dinov3_cases_data/stomach/1790733215386907348_2.png').convert('RGB')
    model_id = 'onnx-community/grounding-dino-tiny-ONNX'
    metadata = json.loads((ARTIFACTS / 'grounding_dino_community_fp32.json').read_text())
    processor = AutoProcessor.from_pretrained(metadata['processor_path'], local_files_only=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    original = ort.InferenceSession(metadata['source_onnx'], sess_options=options,
                                   providers=['CPUExecutionProvider'])
    patched = ort.InferenceSession(str(ARTIFACTS / 'grounding_dino_community.onnx'),
                                  sess_options=options, providers=['CPUExecutionProvider'])
    engine = GroundingTRT(ARTIFACTS / 'grounding_dino_community_fp32.engine', model_id, 'cuda')
    report = {'source': metadata, 'scope': 'Original community ONNX vs equivalent patched ONNX vs TRT on one real image, multiple prompts/orientations. Not an accuracy benchmark. Accepted detections compared at threshold 0.3.',
              'grounding_dino': [], 'vits16': []}
    for picture, prompt in [(image, 'medicine bottle. medicine box. bottle. box.'),
                            (image, 'box. bottle.'),
                            (image.rotate(90, expand=True), 'a medicine bottle. packaging box.')]:
        inputs = engine.prepare_inputs(processor, picture, prompt)
        feed = {value.name: inputs[value.name].cpu().numpy() for value in original.get_inputs()}
        original_values = original.run(['logits', 'pred_boxes'], feed)
        patched_values = patched.run(['logits', 'pred_boxes'], feed)
        patch_errors = {}
        for name, before, after in zip(['logits', 'pred_boxes'], original_values, patched_values):
            assert np.array_equal(np.isfinite(before), np.isfinite(after))
            finite = np.isfinite(before)
            error = float(np.max(np.abs(before[finite] - after[finite])))
            assert error < 1e-6, (name, error)
            patch_errors[name] = error
        expected = SimpleNamespace(logits=torch.from_numpy(original_values[0]).cuda(),
                                   pred_boxes=torch.from_numpy(original_values[1]).cuda())
        start = time.perf_counter()
        actual = engine(**inputs)
        elapsed = (time.perf_counter() - start) * 1000
        valid = torch.isfinite(expected.logits)
        assert torch.equal(valid, torch.isfinite(actual.logits))
        assert torch.isfinite(actual.pred_boxes).all()
        logit_error = (actual.logits[valid] - expected.logits[valid]).abs().max().item()
        box_error = (actual.pred_boxes - expected.pred_boxes).abs().max().item()
        # Unordered queries may differ near tied proposal scores; inspect final
        # high-confidence detections as well as reporting raw tensor errors.
        results = []
        for outputs in (expected, actual):
            result = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=.3, text_threshold=.25,
                target_sizes=[(picture.height, picture.width)])[0]
            results.append(result)
        # Two-stage top-k proposal selection is numerically discontinuous;
        # low-confidence query order/content need not agree. Match final accepted
        # detections by geometry and compare their labels, scores and coordinates.
        from scipy.optimize import linear_sum_assignment
        assert len(results[0]['boxes']) == len(results[1]['boxes'])
        detection_box_error = detection_score_error = 0.
        if len(results[0]['boxes']):
            costs = torch.cdist(results[0]['boxes'], results[1]['boxes'], p=1).cpu().numpy()
            rows, cols = linear_sum_assignment(costs)
            detection_box_error = (results[0]['boxes'][rows] - results[1]['boxes'][cols]).abs().max().item()
            detection_score_error = (results[0]['scores'][rows] - results[1]['scores'][cols]).abs().max().item()
            assert [results[0]['text_labels'][i] for i in rows] == [results[1]['text_labels'][i] for i in cols]
            assert detection_score_error < .005, detection_score_error
            assert detection_box_error < 1., detection_box_error
        row = {'prompt': prompt, 'image_wh': list(picture.size), 'max_logit_error': logit_error,
               'max_normalized_box_error': box_error, 'detections': len(results[1]['boxes']),
               'text_tokens': inputs.input_ids.shape[1], 'onnx_patch_max_error': patch_errors,
               'trt_wall_ms': elapsed, 'accepted_box_max_error_pixels': detection_box_error,
               'accepted_score_max_error': detection_score_error}
        report['grounding_dino'].append(row)
        print(json.dumps(row), flush=True)
    del engine, original, patched
    torch.cuda.empty_cache()
    spec = model_variant('vits16')
    backbone = TRTBackbone(spec['engine_path'], spec['model_id'])
    reference = AutoModel.from_pretrained(spec['model_id'], cache_dir=spec['cache_dir'],
                                         local_files_only=True, dtype=torch.float32,
                                         attn_implementation='eager').cuda().eval()
    processor = AutoImageProcessor.from_pretrained(spec['model_id'], cache_dir=spec['cache_dir'], local_files_only=True)
    for picture in (image.crop((0, 0, 87, 53)), image.resize((448, 320)), image.resize((1280, 720))):
        pixels = processor(images=picture, do_resize=False, do_center_crop=False,
                           return_tensors='pt').pixel_values.cuda().contiguous()
        expected = reference(pixel_values=pixels).last_hidden_state
        actual = backbone(pixels).last_hidden_state
        assert expected.shape == actual.shape
        assert torch.isfinite(actual).all()
        cosine = (F.normalize(expected[:, 5:], dim=-1) * F.normalize(actual[:, 5:], dim=-1)).sum(-1)
        row = {'image_wh': list(picture.size), 'output_shape': list(actual.shape),
               'cosine_mean': cosine.mean().item(), 'cosine_min': cosine.min().item()}
        assert row['cosine_mean'] > .99 and row['cosine_min'] > .95, row
        report['vits16'].append(row)
        print(json.dumps(row), flush=True)
    path = ARTIFACTS / 'verification_grounding_community_vits16.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'PASS: {path}', flush=True)


if __name__ == '__main__':
    main()
