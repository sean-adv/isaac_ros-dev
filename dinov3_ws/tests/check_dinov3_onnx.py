"""Validate ViT-L TRT against the ORIGINAL ONNX, including normalized patch banks."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
from PIL import Image
import torch

from dinov3_cases.trt_backend import TRTBackbone
from dinov3_cases.grounded import prepare_object
from dinov3_cases.patches import preprocess_image, patch_features
from transformers import AutoImageProcessor
from build_dinov3 import Engine, metrics, THRESHOLDS, digest
from build_dinov3_onnx import DEFAULT_OUTPUT, MODEL_ID, REVISION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()
    out = args.output.resolve()
    (out / 'READY').unlink(missing_ok=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(out / 'source/onnx/model.onnx'), options,
                                  providers=['CPUExecutionProvider'])
    processor = AutoImageProcessor.from_pretrained(out / 'source', local_files_only=True)
    model = TRTBackbone(out / 'model.engine', MODEL_ID)
    root = Path(__file__).resolve().parents[2]
    images = [root / 'TensorRT-Model-Connect/families/dinov3/tests/data/test_img.jpeg',
              root / 'dinov3_cases_data/stomach/1790733215386907348_2.png',
              root / 'dinov3_cases_data/stomach/1790733214842141931_1.png']
    rows = []
    # Include two deterministic out-of-distribution inputs to catch numeric issues.
    rng = np.random.default_rng(0)
    cases = [(str(path), prepare_object(Image.open(path).convert('RGB'), 224)) for path in images]
    cases += [('black', Image.new('RGB', (224,224))),
              ('noise', Image.fromarray(rng.integers(0,256,(224,224,3),dtype=np.uint8)))]
    with torch.inference_mode():
        direct = Engine(out / "model.engine", torch.zeros((1,3,224,224),device="cuda"))
        for name, image in cases:
            x = torch.from_numpy(np.array(image)).permute(2,0,1).cuda()
            pixels = preprocess_image(x, processor)
            reference = dict(zip([o.name for o in session.get_outputs()],
                                session.run(None, {'pixel_values':pixels.cpu().numpy()})))
            result = model(pixel_values=pixels)
            hidden = result.last_hidden_state
            direct.set_input(pixels)
            engine_outputs = direct.run()
            torch.testing.assert_close(hidden, engine_outputs["last_hidden_state"])
            actual = {k:v.cpu().numpy() for k,v in engine_outputs.items()}
            values = metrics({k:v.astype(np.float64) for k,v in actual.items()},
                             {k:v.astype(np.float64) for k,v in reference.items()}, 4)
            passed = all(values[k] <= v if k == 'relative_frobenius' else values[k] >= v
                         for k,v in THRESHOLDS.items())
            features = patch_features(result, model,224,224).reshape(196,1024)
            assert torch.isfinite(features).all()
            torch.testing.assert_close(features.norm(dim=-1), torch.ones(196,device='cuda'))
            rows.append(dict(image=name, passed=passed, metrics=values,
                             feature_shape=list(features.shape)))
            print(json.dumps(rows[-1]),flush=True)
        for _ in range(10):
            model(pixel_values=pixels)
        timings = []
        for _ in range(50):
            start = time.perf_counter()
            model(pixel_values=pixels)
            timings.append((time.perf_counter()-start)*1000)
    report = dict(passed=all(r['passed'] for r in rows), model_id=MODEL_ID,
                  revision=REVISION, reference='Original ONNX Runtime CPU FP32',
                  engine_sha256=digest(out/'model.engine'), thresholds=THRESHOLDS, checks=rows,
                  latency_ms=dict(p50=float(np.median(timings)),p95=float(np.percentile(timings,95)),
                                  samples=50,warmup=10),
                  scope='Synchronized TRTBackbone calls; CUDA-resident input; excludes image preprocessing',
                  versions=dict(torch=torch.__version__, onnxruntime=ort.__version__))
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    assert report['passed'], 'Accuracy check failed; see validation.json'
    (out/'READY').write_text('Original ONNX parity and patch normalization passed.\n')
    print(json.dumps(report['latency_ms']),flush=True)


if __name__ == '__main__':
    main()
