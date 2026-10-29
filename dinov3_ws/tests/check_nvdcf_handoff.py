"""Real GPU check: partial refresh retains native state until three SAM3 misses."""
import json

import numpy as np

from dinov3_cases.async_nvdcf import AsyncNvDCFProposals
from dinov3_cases.nvdcf import NvDCF


def main():
    textures = [np.random.default_rng(i).integers(30, 255, (100, 80, 3), dtype=np.uint8)
                for i in range(2)]
    trackers = []

    def factory():
        tracker = NvDCF()
        trackers.append(tracker)
        return tracker

    def image(frame):
        pixels = np.full((540, 960, 3), 25, dtype=np.uint8)
        for index, texture in enumerate(textures):
            x = 100 + index * 300 + frame * 2
            pixels[160:260, x:x+80] = texture
        return pixels

    def detections(frame, labels):
        return [dict(bbox=[100+i*300+frame*2, 160, 180+i*300+frame*2, 260],
                     detection_score=.95, label=label) for i, label in labels]

    scheduler = AsyncNvDCFProposals(factory=factory)

    def refresh(frame, labels):
        detector = lambda _: (detections(frame, labels), {})
        scheduler.force_detection = True
        scheduler.update(image(frame), frame, 'camera', 100, detector)
        scheduler.thread.join(30)
        assert not scheduler.thread.is_alive()
        return scheduler.update(image(frame+1), frame+1, 'camera', 100, detector)[0]

    try:
        first, _ = scheduler.update(image(0), 0, 'camera', 100,
                                    lambda _: (detections(0, [(0, 'A'), (1, 'B')]), {}),
                                    wait_for_initial_result=True)
        assert len(first) == 2, first
        ids = {p['label']: p['track_id'] for p in first}
        original = trackers[0]
        native_b = next(p['nvdcf_id'] for p in first if p['label'] == 'B')
        result = refresh(2, [(0, 'A-new')])
        by_label = {p['label']: p for p in result}
        assert by_label['A-new']['track_id'] == ids['A'], result
        assert by_label['B']['track_id'] == ids['B'], result
        assert by_label['B']['nvdcf_id'] == native_b
        assert by_label['B']['sam3_misses'] == 1
        assert original.process is not None and original.process.poll() is None
        for frame in range(4, 8):
            result, _ = scheduler.update(image(frame), frame, 'camera', 100, lambda _: ([], {}))
            b = next(p for p in result if p['label'] == 'B')
            assert b['sam3_misses'] == 1 and b['nvdcf_id'] == native_b
            assert abs(b['bbox'][0] - (400 + frame * 2)) < 25, b
        result = refresh(8, [(0, 'A-new')])
        assert next(p for p in result if p['label'] == 'B')['sam3_misses'] == 2
        result = refresh(10, [(0, 'A-new')])
        assert [p['track_id'] for p in result] == [ids['A']], result
        scheduler._reaper.drain()
        assert original.process is None
        assert len(scheduler.active.contexts) == 1
        for misses, frame in enumerate((12, 14, 16), 1):
            result = refresh(frame, [])
            if misses < 3:
                assert result[0]['track_id'] == ids['A']
                assert result[0]['sam3_misses'] == misses
            else:
                assert result == [] and not scheduler.active.contexts
        print(json.dumps(dict(passed=True, partial_handoff=True,
                              retained_native_state=True, miss_limit=3)))
    finally:
        scheduler.close()
    assert all(tracker.process is None for tracker in trackers)


if __name__ == '__main__':
    main()
