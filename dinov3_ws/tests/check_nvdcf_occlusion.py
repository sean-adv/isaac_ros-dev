"""Real GPU: full occlusion preserves owners, SAM3 refresh restores public ID."""
import json

import numpy as np

from dinov3_cases.async_nvdcf import AsyncNvDCFProposals
from dinov3_cases.nvdcf import NvDCF


def main():
    visible = np.full((540, 960, 3), 25, dtype=np.uint8)
    visible[160:260, 100:180] = np.random.default_rng(42).integers(
        30, 255, (100, 80, 3), dtype=np.uint8)
    occluded = np.full_like(visible, 25)
    trackers = []

    def factory():
        tracker = NvDCF()
        trackers.append(tracker)
        return tracker

    # Stand in for SAM3/DINO with exact detections of the synthetic texture.
    def detect(pixels):
        boxes = [] if np.all(pixels == 25) else [dict(
            bbox=[100, 160, 180, 260], detection_score=.95, label='medicine')]
        return boxes, {}

    scheduler = AsyncNvDCFProposals(factory=factory, max_missed_frames=6)

    def step(frame, pixels):
        result = scheduler.update(pixels, frame, 'camera', 100, detect,
                                  wait_for_initial_result=True)
        if scheduler.thread is not None:
            scheduler.thread.join(30)
            assert not scheduler.thread.is_alive()
        return scheduler.update(pixels, frame, 'camera', 100, detect)[0]

    try:
        first = step(0, visible)
        assert len(first) == 1, first
        identity = first[0]['track_id']
        original = trackers[0]
        for frame in range(1, 4):
            assert step(frame, occluded) == []
            assert original.process is not None and original.process.poll() is None
            assert len(scheduler.active.hidden) == 1
            assert scheduler.active.hidden[0]['track_id'] == identity
        recovered_frame = None
        for frame in range(4, 7):
            result = step(frame, visible)
            if result:
                assert result[0]['track_id'] == identity, result
                assert result[0]['label'] == 'medicine'
                assert not scheduler.active.hidden
                recovered_frame = frame
                break
        assert recovered_frame is not None, 'No recovery within the native miss grace'
        # A longer occlusion expires owners even across empty detector handoffs.
        for frame in range(recovered_frame+1, recovered_frame+8):
            assert step(frame, occluded) == []
        assert not scheduler.active.contexts
        assert scheduler.scheduler_metrics()['shadow_tracks'] == 0
        print(json.dumps(dict(passed=True, retained_during_occlusion=True,
                              recovered_same_public_id=True,
                              recovered_frame=recovered_frame, expired_after_grace=True)))
    finally:
        scheduler.close()
    assert all(tracker.process is None for tracker in trackers)


if __name__ == '__main__':
    main()
