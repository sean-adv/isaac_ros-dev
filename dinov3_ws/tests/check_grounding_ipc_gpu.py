"""Check real offline OmDet parity between direct and shared-memory execution."""
import argparse
import json
import time
import numpy as np
from PIL import Image
from dinov3_cases.grounded import OmDetDetector, DEFAULT_PROMPT
from dinov3_cases.grounding_ipc import SharedMemoryGroundingDetector
from dinov3_cases.runtime import GROUNDING_ENGINE, TRT_PYTHON, WORKSPACE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples',type=int,default=20)
    parser.add_argument('--direct',action='store_true')
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    image = Image.open(WORKSPACE / 'dinov3_cases_data/stomach/1790733215386907348_2.png').convert('RGB')
    direct = OmDetDetector(GROUNDING_ENGINE,'cuda')
    worker = None if args.direct else SharedMemoryGroundingDetector(GROUNDING_ENGINE,'cuda',python=str(TRT_PYTHON))
    detector = worker or direct
    try:
        for frame in [image,image.rotate(90,expand=True)]:
            expected = direct.detect(frame,DEFAULT_PROMPT,.3,.5,20)
            actual = detector.detect(frame,DEFAULT_PROMPT,.3,.5,20)
            assert len(actual)==len(expected)
            for a,e in zip(actual,expected):
                assert a['bbox']==e['bbox'] and a['detection_label']==e['detection_label']
                np.testing.assert_allclose(a['detection_score'],e['detection_score'],atol=1e-6)
        detector.validate_prompt(DEFAULT_PROMPT)
        try:
            detector.validate_prompt('unencoded category.')
        except ValueError:
            pass
        else:
            raise AssertionError('Unencoded prompt accepted')
        assert detector.offline_prompts==direct.offline_prompts
        for _ in range(3):
            detector.detect(image,DEFAULT_PROMPT,.3,.5,20)
        rows=[]
        for _ in range(args.samples):
            start=time.monotonic()
            detector.detect(image,DEFAULT_PROMPT,.3,.5,20)
            rows.append(dict(getattr(detector,'last_metrics',{}),caller_detect_ms=(time.monotonic()-start)*1000))
        report=dict(passed=True,worker=getattr(detector,'worker_info',None),
                    median_detect_ms=float(np.median([r['caller_detect_ms'] for r in rows])),rows=rows)
        path=GROUNDING_ENGINE.parent / ('ipc_direct.json' if args.direct else 'ipc_validation.json')
        path.write_text(json.dumps(report,indent=2)+'\n')
        print('PASS: real OmDet offline IPC parity',path)
    finally:
        if worker:
            worker.close()


if __name__=='__main__':
    main()
