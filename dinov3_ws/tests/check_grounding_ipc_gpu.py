"""Compare legacy direct results with shared-memory results and time warm IPC."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import tensorrt

from dinov3_cases.grounded import GroundingDetector
from dinov3_cases.grounding_ipc import SharedMemoryGroundingDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--direct', action='store_true')
    parser.add_argument('--samples', type=int, default=20)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    artifacts = root / 'dinov3_ws/artifacts'
    output = artifacts / 'tensorrt11'
    image = Image.open(root / 'dinov3_cases_data/stomach/1790733215386907348_2.png').convert('RGB')
    config = dict(model_id='IDEA-Research/grounding-dino-tiny', device='cuda',
                  cache_dir=None, backend='trt',
                  engine_path=artifacts / 'tensorrt/grounding_dino_wingdzero_fp16.engine')
    detector = (GroundingDetector(**config) if args.direct else
                SharedMemoryGroundingDetector(**config, python=str(root / 'dinov3_venv/bin/python')))
    prompt = 'medicine bottle. medicine box. bottle. box.'
    cases = [(image, prompt), (image, 'box. bottle.'), (image.rotate(90, expand=True), prompt)]
    def detect(im, text):
        return detector.detect(im, text, .3, .25, .5, 20)
    try:
        if args.direct:
            assert tensorrt.__version__ == '10.16.2.10'
        else:
            assert tensorrt.__version__ == '11.1.0.106'
            assert detector.worker_info['tensorrt'] == '10.16.2.10'
        results = [detect(im, text) for im, text in cases]
        if not args.direct:
            reference = json.loads((output / 'ipc-direct-reference.json').read_text())['results']
            for actual, expected in zip(results, reference):
                assert len(actual) == len(expected)
                for a, b in zip(actual, expected):
                    assert a['bbox'] == b['bbox'] and a['detection_label'] == b['detection_label']
                    np.testing.assert_allclose(a['detection_score'], b['detection_score'], atol=1e-6, rtol=0)
        for _ in range(3):
            detect(image, prompt)
        measurements = []
        previous_id = 0
        for _ in range(args.samples):
            start = time.monotonic_ns()
            detect(image, prompt)
            elapsed = (time.monotonic_ns() - start) / 1e6
            entry = dict(detector.last_metrics) if not args.direct else {}
            entry['caller_detect_ms'] = elapsed
            if not args.direct:
                assert entry['frame_id'] > previous_id
                previous_id = entry['frame_id']
            measurements.append(entry)
        summary = {key: dict(median=float(np.median([m[key] for m in measurements])),
                             p95=float(np.percentile([m[key] for m in measurements], 95)))
                   for key in measurements[0] if key.endswith('_ms')}
        report = dict(passed=True, parent_tensorrt=tensorrt.__version__,
                      worker=getattr(detector, 'worker_info', None), image_size=list(image.size),
                      samples=args.samples, warmups=3, summary_ms=summary,
                      results=results, measurements=measurements)
        path = output / ('ipc-direct-reference.json' if args.direct else 'ipc-latency.json')
        path.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(report=str(path), passed=True, summary_ms=summary), indent=2))
    finally:
        if not args.direct:
            detector.close()


if __name__ == '__main__':
    main()
