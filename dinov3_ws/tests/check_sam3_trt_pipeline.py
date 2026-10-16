"""Validate SAM3 TRT10 direct/IPC parity and DINOv3 crop encoding."""
import json
from pathlib import Path
import sys
import time
import subprocess

import numpy as np
from PIL import Image
import tensorrt as trt

from dinov3_cases.encoder import Encoder
from dinov3_cases.grounded import prepare_object
from dinov3_cases.grounding_ipc import SharedMemoryGroundingDetector
from dinov3_cases.runtime import WORKSPACE, GROUNDING_ENGINE, GROUNDING_PYTHON
from dinov3_cases.sam3_trt import Sam3Detector


def sample(name):
    assets = GROUNDING_ENGINE.parent
    baseline = json.loads((assets / 'bbox_smoke.jpg.boxes.jsonl').read_text().splitlines()[0])
    baseline['image'] = str(WORKSPACE / 'DART/dartf/data/train2017/000000230971.jpg')
    baseline['prompts'] = ['person'] if name == 'single' else ['person', 'car', 'bicycle', 'dog']
    return baseline


def iou(a, b):
    lo = np.maximum(a[:2], b[:2])
    hi = np.minimum(a[2:], b[2:])
    intersection = np.prod(np.maximum(0, hi-lo))
    area = lambda box: np.prod(np.maximum(0, np.array(box[2:])-box[:2]))
    return float(intersection / max(area(a)+area(b)-intersection, 1e-9))


def main():
    if '--direct-worker' in sys.argv:
        assert trt.__version__.startswith('10.')
        direct = Sam3Detector(GROUNDING_ENGINE, 'cuda:0')
        detections = {}
        for name in ['single', 'multi']:
            baseline = sample(name)
            image = Image.open(baseline['image']).convert('RGB')
            detections[name] = direct.detect(image, ','.join(baseline['prompts']), .3, 1., 20)
        print(json.dumps(detections))
        return
    assert trt.__version__.startswith('11.')
    # The original SAM3 plans must be deserialized in a separate TRT10 process.
    import os
    env = os.environ.copy()
    venv = GROUNDING_PYTHON.parent.parent
    env['LD_LIBRARY_PATH'] = os.pathsep.join([
        *[str(p) for p in venv.glob('lib/python*/site-packages/tensorrt_libs')],
        '/usr/local/cuda/lib64', '/usr/lib/aarch64-linux-gnu'])
    output = subprocess.check_output(
        [str(GROUNDING_PYTHON), str(Path(__file__).resolve()), '--direct-worker'],
        env=env, text=True)
    direct = json.loads(output.strip().splitlines()[-1])
    encoder = Encoder()
    worker = SharedMemoryGroundingDetector(GROUNDING_ENGINE, 'cuda:0', python=GROUNDING_PYTHON)
    rows = []
    try:
        assert Path(worker.worker_info['python']).absolute() == GROUNDING_PYTHON.absolute()
        assert worker.worker_info['tensorrt'].startswith('10.')
        for name in ['single', 'multi']:
            baseline = sample(name)
            image = Image.open(baseline['image']).convert('RGB')
            prompt = ','.join(baseline['prompts'])
            start = time.monotonic()
            actual = worker.detect(image, prompt, .3, 1., 20)
            elapsed = (time.monotonic()-start)*1000
            remote = direct[name]
            assert len(actual) == len(baseline['detections']) == len(remote), (name, actual)
            np.testing.assert_allclose([d['bbox'] for d in actual], [d['bbox'] for d in remote], atol=0)
            np.testing.assert_allclose([d['detection_score'] for d in actual],
                                       [d['detection_score'] for d in remote], atol=1e-6)
            comparisons = []
            for detection in actual:
                candidates = [d for d in baseline['detections'] if d['label'] == detection['detection_label']]
                reference = max(candidates, key=lambda d: iou(detection['bbox'], d['bbox_xyxy']))
                overlap = iou(detection['bbox'], reference['bbox_xyxy'])
                score_error = abs(detection['detection_score']-reference['score'])
                assert overlap > .95 and score_error < .05, (name, overlap, score_error)
                crop = image.crop(detection['bbox'])
                features = encoder.encode(prepare_object(crop))
                assert features.ndim == 2 and features.shape[-1] == 768
                assert features.isfinite().all()
                comparisons.append(dict(label=detection['detection_label'], iou=overlap,
                                        score_error=score_error, feature_shape=list(features.shape)))
            rows.append(dict(image=name, detections=actual, baseline=comparisons, detect_ms=elapsed))
        report = dict(passed=True, python=sys.executable, tensorrt=trt.__version__,
                      worker=worker.worker_info, engine=str(GROUNDING_ENGINE), results=rows)
        path = GROUNDING_ENGINE.parent / 'pipeline_validation.json'
        path.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
    finally:
        worker.close()


if __name__ == '__main__':
    main()
