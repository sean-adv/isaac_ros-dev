"""Compare official PyTorch -> ONNX -> TensorRT, including real-image detection."""
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

from dinov3_cases.omdet_trt import OmDetDetector, prepare_image, postprocess
from dinov3_cases.runtime import GROUNDING_ENGINE, WORKSPACE


def compare_queries(actual, expected):
    # Decoder Top-K may permute equally ranked anchors. Compare paired sets.
    from scipy.optimize import linear_sum_assignment
    a = {name: np.asarray(value).reshape(-1, np.asarray(value).shape[-1]) for name,value in actual.items()}
    e = {name: np.asarray(value).reshape(-1, np.asarray(value).shape[-1]) for name,value in expected.items()}
    cost = np.max(np.abs(e['boxes'][:,None] - a['boxes'][None,:]),axis=-1)
    cost += .05*np.max(np.abs(e['logits'][:,None] - a['logits'][None,:]),axis=-1)
    rows, cols = linear_sum_assignment(cost)
    errors = {}
    for name in ['boxes','logits']:
        np.testing.assert_allclose(a[name][cols], e[name][rows], atol=3e-3, rtol=3e-3)
        errors[name] = float(np.max(np.abs(a[name][cols] - e[name][rows])))
    return errors


def main():
    metadata = json.loads(GROUNDING_ENGINE.with_suffix('.json').read_text())
    if metadata['precision'] == 'FP16':
        from check_omdet_fp16 import main as validate_fp16
        return validate_fp16()
    detector = OmDetDetector(GROUNDING_ENGINE, 'cuda')
    artifact = GROUNDING_ENGINE.parent
    data = np.load(artifact / 'export_reference.npz', allow_pickle=False)
    inputs = {n: torch.from_numpy(data[n]) for n in detector.inputs}
    actual = detector.infer(inputs)
    errors = compare_queries({n: actual[n].cpu().numpy() for n in ['boxes','logits']},
                             {n: data[n] for n in ['boxes','logits']})
    import onnxruntime as ort
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 8
    session = ort.InferenceSession(str(artifact / 'omdet_tiny.onnx'), sess_options=session_options, providers=['CPUExecutionProvider'])
    onnx_outputs = session.run(None, {n: data[n] for n in detector.inputs})
    errors.update({'onnx_' + n: v for n,v in compare_queries(dict(zip(['boxes','logits'],onnx_outputs)),
                                                           {n:data[n] for n in ['boxes','logits']}).items()})
    sys.path[:0] = [str(WORKSPACE / 'OmDet'), str(WORKSPACE / 'detectron2')]
    from detectron2.config import get_cfg
    from omdet.omdet_v2_turbo.config import add_omdet_v2_turbo_config
    from omdet.omdet_v2_turbo.infer_model import OmDetV2TurboInfer
    cfg = get_cfg()
    add_omdet_v2_turbo_config(cfg)
    cfg.merge_from_file(str(WORKSPACE / 'OmDet/configs/OmDet-Turbo_tiny_SWIN_T.yaml'))
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.META_ARCHITECTURE = 'OmDetV2TurboInfer'
    model = OmDetV2TurboInfer(cfg).eval()
    state = torch.load(WORKSPACE / 'OmDet/resources/OmDet-Turbo_tiny_SWIN_T.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state.get('model', state), strict=False)
    torch.set_num_threads(8)
    torch.backends.mha.set_fastpath_enabled(False)
    image = Image.open(WORKSPACE / 'dinov3_cases_data/stomach/1790733215386907348_2.png').convert('RGB')
    profile = detector.bank.profiles[0]
    prompt = profile['prompt']
    _, features = detector.bank.resolve(prompt)
    real_inputs = dict(img_tensor=prepare_image(image, detector.metadata),
                       **{n: torch.from_numpy(v) for n,v in features.items()})
    with torch.inference_mode():
        boxes, logits = model(*(real_inputs[n] for n in ['img_tensor','label_feats','task_feats','task_mask']))
    result = detector.infer(real_inputs)
    errors.update({'real_' + n: v for n,v in compare_queries({n:result[n].cpu().numpy() for n in ['boxes','logits']},
                                                           dict(boxes=boxes.numpy(),logits=logits.numpy())).items()})
    # Padding to engine capacity must preserve the official trimmed task semantics.
    trimmed = dict(real_inputs)
    length = int(real_inputs['task_mask'].sum())
    trimmed['task_feats'] = trimmed['task_feats'][:length]
    trimmed['task_mask'] = trimmed['task_mask'][:, :length]
    with torch.inference_mode():
        trimmed_boxes, trimmed_logits = model(*(trimmed[n] for n in ['img_tensor','label_feats','task_feats','task_mask']))
    errors.update({'task_padding_' + n: v for n,v in compare_queries(dict(boxes=boxes.numpy(),logits=logits.numpy()),
                                dict(boxes=trimmed_boxes.numpy(),logits=trimmed_logits.numpy())).items()})
    expected_proposals = postprocess(boxes, logits, profile['labels'], image.size, .3, .5, 20)
    proposals = detector.detect(image, prompt, .3, .5, 20)
    assert proposals and len(proposals) == len(expected_proposals), (proposals, expected_proposals)
    for actual, expected in zip(proposals, expected_proposals):
        assert actual['detection_label'] == expected['detection_label']
        np.testing.assert_allclose(actual['bbox'], expected['bbox'], atol=1)
        np.testing.assert_allclose(actual['detection_score'], expected['detection_score'], atol=.001)
    for _ in range(3):
        detector.detect(image, prompt, .3, .5, 20)
    measurements = []
    for _ in range(10):
        start = time.perf_counter()
        detector.detect(image, prompt, .3, .5, 20)
        measurements.append((time.perf_counter()-start)*1000)
    report = dict(passed=True, errors=errors, detections=proposals,
                  median_detect_ms=float(np.median(measurements)), precision=detector.metadata['precision'])
    (artifact / 'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
