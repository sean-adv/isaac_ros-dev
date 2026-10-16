"""Measure current SAM3 worker and shared-memory transport on a saved frame."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from dinov3_cases.grounded import DEFAULT_PROMPT
from dinov3_cases.runtime import GROUNDING_ENGINE, GROUNDING_PYTHON
from dinov3_cases.grounding_ipc import SharedMemoryGroundingDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=60)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    detector = SharedMemoryGroundingDetector(GROUNDING_ENGINE, 'cuda', python=str(GROUNDING_PYTHON))
    image = Image.open(args.image).convert('RGB')
    rows = []
    try:
        for i in range(10+args.samples):
            start = time.monotonic()
            proposals = detector.detect(image,DEFAULT_PROMPT,.3,.5,20)
            elapsed = (time.monotonic()-start)*1000
            if i>=10:
                rows.append(dict(getattr(detector,'last_metrics',{}),total_ms=elapsed,objects=len(proposals)))
    finally:
        if hasattr(detector,'close'):
            detector.close()
    summary = {key:dict(median=float(np.median([r[key] for r in rows])),p95=float(np.percentile([r[key] for r in rows],95)))
               for key in rows[0] if key.endswith('_ms')}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(model='SAM3 W8A8',samples=args.samples,summary=summary,rows=rows),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
