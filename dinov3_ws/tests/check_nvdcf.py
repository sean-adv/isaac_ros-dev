"""Real GPU smoke test: tracking motion, metadata, IDs, and TensorRT isolation."""
import json
import os
import time

import numpy as np

from dinov3_cases.nvdcf import NvDCF


def main():
    # Reproduce the model process's pip TensorRT preference. Native overrides it.
    os.environ['LD_LIBRARY_PATH'] = os.path.abspath(
        'dinov3_ws/.venv/lib/python3.12/site-packages/tensorrt_libs')
    texture = np.random.default_rng(42).integers(30, 255, (100, 80, 3), dtype=np.uint8)
    tracker = NvDCF()
    latencies, errors, ids = [], [], set()
    try:
        for index in range(40):
            image = np.full((540, 960, 3), 25, dtype=np.uint8)
            x, y = 100 + index * 2, 160
            image[y:y+100, x:x+80] = texture
            detections = [dict(bbox=[x,y,x+80,y+100], detection_score=.95,
                               label='synthetic-medicine', track_id=77)] if index == 0 else None
            started = time.monotonic()
            output = tracker.update(image, index, detections)
            latencies.append((time.monotonic()-started)*1000)
            assert len(output) == 1, (index, output)
            item = output[0]
            assert item['label'] == 'synthetic-medicine' and item['track_id'] == 77
            assert item['detection_score'] == .95
            ids.add(item['nvdcf_id'])
            errors.append(abs((item['bbox'][0]+item['bbox'][2])/2 - (x+40)))
            assert errors[-1] < 25, (index, item['bbox'], x)
        assert len(ids) == 1
        # A repeated admitted frame does not advance native state.
        serial = tracker.serial
        assert tracker.update(image, 39) is tracker.proposals
        assert tracker.serial == serial
        print(json.dumps(dict(frames=40, native_ids=list(ids), max_center_error_px=max(errors),
                              median_update_ms=float(np.median(latencies[5:])),
                              first_update_ms=latencies[0]), indent=2))
    finally:
        tracker.close()
    # Empty detector input must be valid too.
    tracker = NvDCF()
    try:
        assert tracker.update(image, 0, []) == []
        assert tracker.update(image, 1) == []
    finally:
        tracker.close()


if __name__ == '__main__':
    main()
