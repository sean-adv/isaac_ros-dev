"""Background detection must not stall KCF or apply old-coordinate boxes."""

import threading
from unittest.mock import Mock

import numpy as np
import pytest

from dinov3_cases.async_kcf import AsyncKCFProposals
from dinov3_cases.frame_pool import SharedFramePool


def pixels(n):
    return np.full((80, 100, 3), n, dtype=np.uint8)


def proposals(n=0):
    return [dict(bbox=[10+n, 12, 30+n, 32], area=400,
                 detection_score=.9, detection_label="box")]


class MovingTracker:
    def init(self, image, box):
        self.previous = int(image[0, 0, 0])
        self.box = box

    def update(self, image):
        now = int(image[0, 0, 0])
        x, y, w, h = self.box
        self.box = x + now - self.previous, y, w, h
        self.previous = now
        return True, self.box


def detect(image):
    return proposals(int(image[0, 0, 0])), {"test": True}


def finish(pipeline):
    pipeline.thread.join(3)
    assert not pipeline.thread.is_alive()


def update(pipeline, n, detector=detect, context="camera", interval=2):
    return pipeline.update(pixels(n), n, context, interval, detector)


def seed(pipeline):
    assert update(pipeline, 0) is None
    finish(pipeline)
    assert update(pipeline, 0)[1] == "sam3"


def test_kcf_continues_during_sam3_and_new_trackers_replay_before_handoff():
    pipeline = AsyncKCFProposals(MovingTracker)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def slow(image):
        calls.append(int(image[0, 0, 0]))
        entered.set()
        assert release.wait(3)
        return detect(image)

    try:
        seed(pipeline)
        assert update(pipeline, 1)[0][0]["bbox"][0] == 11
        assert update(pipeline, 2, slow)[1] == "kcf"
        assert entered.wait(2)
        # These complete while the detection is still blocked.
        for n in (3, 4):
            items, source = update(pipeline, n, slow)
            assert source == "kcf" and items[0]["bbox"][0] == 10+n
        assert calls == [2]  # No queued SAM3 jobs.
        release.set()
        finish(pipeline)
        items, source = update(pipeline, 5)
        assert source == "sam3"
        assert items[0]["bbox"][0] == 15  # Not frame 2's x=12.
        assert pipeline.last_metrics == {"test": True}
    finally:
        release.set()
        pipeline.close()


def test_old_context_result_is_discarded_and_new_context_starts_empty():
    pipeline = AsyncKCFProposals(MovingTracker)
    entered, release = threading.Event(), threading.Event()

    def slow(image):
        entered.set()
        assert release.wait(3)
        return detect(image)

    try:
        assert update(pipeline, 0, slow) is None
        assert entered.wait(2)
        assert update(pipeline, 1, context="upload") is None
        release.set()
        finish(pipeline)
        assert pipeline.ready is None
        assert update(pipeline, 1, context="upload") is None
        finish(pipeline)
        items, source = update(pipeline, 1, context="upload")
        assert source == "sam3" and items[0]["bbox"][0] == 11
    finally:
        release.set()
        pipeline.close()


def test_static_image_applies_result_once_without_repeated_detection():
    pipeline = AsyncKCFProposals(MovingTracker)
    detector = Mock(side_effect=detect)
    try:
        assert update(pipeline, 0, detector) is None
        finish(pipeline)
        assert pipeline.has_result()
        assert update(pipeline, 0, detector)[1] == "sam3"
        assert not pipeline.has_result()
        for _ in range(5):
            update(pipeline, 0, detector)
        assert detector.call_count == 1
    finally:
        pipeline.close()


def test_history_overflow_discards_result_instead_of_skipping_missing_frames():
    pipeline = AsyncKCFProposals(MovingTracker, max_frames=2, max_bytes=pixels(0).nbytes*2)
    entered, release = threading.Event(), threading.Event()

    def slow(image):
        entered.set()
        assert release.wait(3)
        return detect(image)

    try:
        seed(pipeline)
        update(pipeline, 1)
        update(pipeline, 2, slow)
        assert entered.wait(2)
        for n in range(3, 7):
            assert update(pipeline, n, slow)[1] == "kcf"
        assert len(pipeline.history) == 2
        assert pipeline.history_bytes <= pixels(0).nbytes*2
        release.set()
        finish(pipeline)
        assert pipeline.ready is None and pipeline.has_result()
        assert update(pipeline, 7)[1] == "kcf"
        finish(pipeline)
        assert update(pipeline, 7)[0][0]["bbox"][0] == 17
    finally:
        release.set()
        pipeline.close()


def test_shared_frame_is_retained_until_background_detector_finishes():
    pool = SharedFramePool()
    pipeline = AsyncKCFProposals(MovingTracker)
    entered, release = threading.Event(), threading.Event()
    frame = pool.acquire(pixels(0))

    def slow(leased):
        entered.set()
        assert release.wait(3)
        np.testing.assert_array_equal(leased.pixels, pixels(0))
        return proposals(), {}

    try:
        pipeline.update(frame.pixels, 0, 0, 2, slow, frame=frame)
        assert entered.wait(2)
        frame.release()
        assert pool.slots[0][1] == 1
        release.set()
        finish(pipeline)
        assert pool.slots[0][1] == 0
    finally:
        release.set()
        pipeline.close()
        frame.release()
        pool.close()


def test_background_error_surfaces_and_can_be_retried():
    pipeline = AsyncKCFProposals(MovingTracker)
    try:
        update(pipeline, 0, Mock(side_effect=RuntimeError("detector failed")))
        finish(pipeline)
        with pytest.raises(RuntimeError, match="detector failed"):
            update(pipeline, 0)
        assert update(pipeline, 0) is None
        finish(pipeline)
        assert update(pipeline, 0)[1] == "sam3"
    finally:
        pipeline.close()


def test_interval_one_keeps_synchronous_detection_without_kcf():
    factory = Mock(side_effect=AssertionError("KCF must be disabled"))
    pipeline = AsyncKCFProposals(factory)
    try:
        assert update(pipeline, 0, interval=1)[1] == "sam3"
        assert pipeline.thread is None
        factory.assert_not_called()
    finally:
        pipeline.close()


def test_slow_background_replay_does_not_hold_the_main_tracking_lock():
    replay_entered, release_replay = threading.Event(), threading.Event()
    detect_entered, release_detect = threading.Event(), threading.Event()

    class SlowReplayTracker(MovingTracker):
        def update(self, image):
            if threading.current_thread().name == "sam3-kcf-replay":
                replay_entered.set()
                assert release_replay.wait(3)
            return super().update(image)

    pipeline = AsyncKCFProposals(SlowReplayTracker)

    def slow(image):
        detect_entered.set()
        assert release_detect.wait(3)
        return detect(image)

    try:
        seed(pipeline)
        update(pipeline, 1)
        update(pipeline, 2, slow)
        assert detect_entered.wait(2)
        update(pipeline, 3, slow)
        release_detect.set()
        assert replay_entered.wait(2)
        assert pipeline.lock.acquire(timeout=.5), "Replay must not hold the history lock"
        pipeline.lock.release()
        items, source = update(pipeline, 4, slow)
        assert source == "kcf" and items[0]["bbox"][0] == 14
        release_replay.set()
        finish(pipeline)
        assert update(pipeline, 4)[1] == "sam3"
    finally:
        release_detect.set()
        release_replay.set()
        pipeline.close()


def test_failed_background_replay_is_discarded():
    entered, release = threading.Event(), threading.Event()

    class LostReplayTracker(MovingTracker):
        def update(self, image):
            if threading.current_thread().name == "sam3-kcf-replay":
                return False, (0, 0, 0, 0)
            return super().update(image)

    pipeline = AsyncKCFProposals(LostReplayTracker)

    def slow(image):
        entered.set()
        assert release.wait(3)
        return detect(image)

    try:
        seed(pipeline)
        update(pipeline, 1)
        update(pipeline, 2, slow)
        assert entered.wait(2)
        update(pipeline, 3, slow)
        release.set()
        finish(pipeline)
        assert pipeline.ready is None and pipeline.force_detection
    finally:
        release.set()
        pipeline.close()


def test_empty_detection_clears_active_tracks_on_handoff():
    pipeline = AsyncKCFProposals(MovingTracker)
    try:
        seed(pipeline)
        update(pipeline, 1)
        update(pipeline, 2, lambda _: ([], {}))
        finish(pipeline)
        assert update(pipeline, 2) == ([], "sam3")
        assert pipeline.active.trackers == []
    finally:
        pipeline.close()



@pytest.mark.parametrize('value', [0, -1, True, 1.5, '8'])
def test_invalid_replay_frame_limit(value):
    with pytest.raises(ValueError, match='max_replay_frames'):
        AsyncKCFProposals(max_replay_frames=value)


@pytest.mark.parametrize('value', [0, -1, True, float('nan'), float('inf'), '150'])
def test_invalid_replay_time_budget(value):
    with pytest.raises(ValueError, match='replay_budget_ms'):
        AsyncKCFProposals(replay_budget_ms=value)


def test_initial_detection_does_not_accumulate_arrivals_or_replay_them():
    entered, release = threading.Event(), threading.Event()
    replay_updates = []

    class CountReplayTracker(MovingTracker):
        def update(self, image):
            if threading.current_thread().name == 'sam3-kcf-replay':
                replay_updates.append(int(image[0, 0, 0]))
            return super().update(image)

    pipeline = AsyncKCFProposals(CountReplayTracker, max_frames=2)

    def slow(image):
        entered.set()
        assert release.wait(3)
        return detect(image)

    try:
        assert update(pipeline, 0, slow, interval=100) is None
        assert entered.wait(2)
        initial_serial = pipeline.serial
        assert pipeline.waiting_for_initial_detection('camera', 100)
        assert not pipeline.waiting_for_initial_detection('other-camera', 100)
        for n in range(1, 101):
            assert update(pipeline, n, slow, interval=100) is None
        assert pipeline.serial == initial_serial
        assert len(pipeline.history) == 1
        assert pipeline.frames_since_submit == 0
        assert pipeline.last_key == 0
        release.set()
        finish(pipeline)
        assert not pipeline.waiting_for_initial_detection('camera', 100)
        assert replay_updates == []
        # The seeded model actually tracks the current image, rather than
        # returning frame 0's coordinates as if they came from frame 5.
        items, source = update(pipeline, 5, interval=100)
        assert source == 'sam3' and items[0]['bbox'][0] == 15
        assert pipeline.serial == initial_serial + 1
        assert pipeline.scheduler_metrics()['replay_frames'] == 0
    finally:
        release.set()
        pipeline.close()


def test_replay_frame_budget_preserves_foreground_and_waits_for_normal_interval():
    entered, release = threading.Event(), threading.Event()
    pipeline = AsyncKCFProposals(MovingTracker, max_replay_frames=2)
    detector = Mock()

    def slow(image):
        entered.set()
        assert release.wait(3)
        return detect(image)

    detector.side_effect = slow
    try:
        update(pipeline, 0, interval=100)
        finish(pipeline)
        update(pipeline, 0, interval=100)
        active = pipeline.active
        pipeline.force_detection = True
        update(pipeline, 1, detector, interval=100)
        assert entered.wait(2)
        for n in (2, 3, 4):
            assert update(pipeline, n, detector, interval=100)[0][0]['bbox'][0] == 10+n
        release.set()
        finish(pipeline)
        assert pipeline.ready is None
        assert pipeline.active is active
        assert not pipeline.force_detection
        assert pipeline.scheduler_metrics()['discard_reason'] == 'frame_budget'
        assert pipeline.scheduler_metrics()['replay_frames'] == 2
        assert update(pipeline, 5, detector, interval=100)[0][0]['bbox'][0] == 15
        assert detector.call_count == 1
    finally:
        release.set()
        pipeline.close()


def test_replay_time_budget_stops_before_next_update(monkeypatch):
    from types import SimpleNamespace
    from dinov3_cases import async_kcf
    clock = [0.0]
    monkeypatch.setattr(async_kcf, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    entered, release = threading.Event(), threading.Event()

    class TimedTracker(MovingTracker):
        def update(self, image):
            if threading.current_thread().name == 'sam3-kcf-replay':
                clock[0] += .060
            return super().update(image)

    pipeline = AsyncKCFProposals(TimedTracker, replay_budget_ms=50)

    def slow(image):
        entered.set()
        assert release.wait(3)
        return detect(image)

    try:
        seed(pipeline)
        update(pipeline, 1)
        update(pipeline, 2, slow)
        assert entered.wait(2)
        update(pipeline, 3, slow)
        update(pipeline, 4, slow)
        release.set()
        finish(pipeline)
        assert pipeline.ready is None
        metrics = pipeline.scheduler_metrics()
        assert metrics['discard_reason'] == 'time_budget'
        assert metrics['replay_frames'] == 1
        assert metrics['replay_ms'] == pytest.approx(60)
        assert not pipeline.force_detection
        assert pipeline.active.proposals[0]['bbox'][0] == 14
    finally:
        release.set()
        pipeline.close()


def test_new_arrivals_during_replay_cannot_extend_job_past_frame_budget():
    entered, release = threading.Event(), threading.Event()
    replay_entered, replay_release = threading.Event(), threading.Event()
    replayed = []

    class BlockFirstReplay(MovingTracker):
        def update(self, image):
            if threading.current_thread().name == 'sam3-kcf-replay':
                replayed.append(int(image[0, 0, 0]))
                if len(replayed) == 1:
                    replay_entered.set()
                    assert replay_release.wait(3)
            return super().update(image)

    pipeline = AsyncKCFProposals(BlockFirstReplay, max_replay_frames=2, replay_budget_ms=2000)

    def slow(image):
        entered.set()
        assert release.wait(3)
        return detect(image)

    try:
        seed(pipeline)
        update(pipeline, 1)
        update(pipeline, 2, slow)
        assert entered.wait(2)
        update(pipeline, 3, slow)
        release.set()
        assert replay_entered.wait(2)
        for n in (4, 5, 6):
            items, source = update(pipeline, n, slow)
            assert source == 'kcf' and items[0]['bbox'][0] == 10+n
        replay_release.set()
        finish(pipeline)
        assert replayed == [3, 4]
        assert pipeline.ready is None
        assert pipeline.scheduler_metrics()['discard_reason'] == 'frame_budget'
        assert pipeline.active.proposals[0]['bbox'][0] == 16
    finally:
        release.set()
        replay_release.set()
        pipeline.close()


def test_context_change_cancels_remaining_replay_batch():
    detect_entered, detect_release = threading.Event(), threading.Event()
    replay_entered, replay_release = threading.Event(), threading.Event()
    replayed = []

    class BlockFirstReplay(MovingTracker):
        def update(self, image):
            if threading.current_thread().name == 'sam3-kcf-replay':
                replayed.append(int(image[0, 0, 0]))
                replay_entered.set()
                assert replay_release.wait(3)
            return super().update(image)

    pipeline = AsyncKCFProposals(BlockFirstReplay, replay_budget_ms=2000)

    def slow(image):
        detect_entered.set()
        assert detect_release.wait(3)
        return detect(image)

    try:
        seed(pipeline)
        update(pipeline, 1)
        update(pipeline, 2, slow)
        assert detect_entered.wait(2)
        update(pipeline, 3, slow)
        update(pipeline, 4, slow)
        detect_release.set()
        assert replay_entered.wait(2)
        assert update(pipeline, 5, context='new-source') is None
        replay_release.set()
        finish(pipeline)
        assert replayed == [3]
        assert pipeline.ready is None
        assert not pipeline.active.initialized
        assert pipeline.last_metrics == {}
        assert update(pipeline, 6, context='new-source') is None
        finish(pipeline)
        assert update(pipeline, 6, context='new-source')[0][0]['bbox'][0] == 16
    finally:
        detect_release.set()
        replay_release.set()
        pipeline.close()


def test_failed_first_current_frame_tracking_requests_fresh_detection():
    class LostOnLargeJumpTracker(MovingTracker):
        def update(self, image):
            if int(image[0, 0, 0]) - self.previous > 10:
                return False, (0, 0, 0, 0)
            return super().update(image)

    pipeline = AsyncKCFProposals(LostOnLargeJumpTracker)
    try:
        update(pipeline, 0, interval=100)
        finish(pipeline)
        # Force a first-update tracking failure after a large motion jump;
        # never publish the seed frame's old coordinates on the new image.
        items, source = update(pipeline, 20, interval=100)
        assert items == []
        assert pipeline.thread is not None
        finish(pipeline)
        items, source = update(pipeline, 20, interval=100)
        assert source == 'sam3' and items[0]['bbox'][0] == 30
    finally:
        pipeline.close()


def test_roi_edge_requests_background_detection_before_interval_without_dropping_box():
    pipeline = AsyncKCFProposals(MovingTracker)
    entered, release = threading.Event(), threading.Event()

    def slow(image):
        entered.set()
        assert release.wait(3)
        return detect(image)

    try:
        assert update(pipeline, 0, interval=100) is None
        finish(pipeline)
        update(pipeline, 0, interval=100)
        items, source = update(pipeline, 8, slow, interval=100)
        assert source == 'kcf' and items[0]['bbox'] == [18, 12, 38, 32]
        assert entered.wait(2)
        assert pipeline.frames_since_submit == 0
        release.set()
        finish(pipeline)
        items, source = update(pipeline, 8, interval=100)
        assert source == 'sam3' and items[0]['bbox'] == [18, 12, 38, 32]
        assert pipeline.active.rois == [(8, 2, 48, 42)]
    finally:
        release.set()
        pipeline.close()


@pytest.mark.parametrize('lose_all', [False, True])
def test_tracking_loss_keeps_valid_boxes_and_requests_sam3_before_interval(lose_all):
    pipeline = AsyncKCFProposals(MovingTracker, roi_padding=None)
    entered, release = threading.Event(), threading.Event()
    original = proposals() + [dict(proposals()[0], bbox=[50, 12, 70, 32])]

    def initial(image):
        return original, {}

    def slow(image):
        entered.set()
        assert release.wait(3)
        return original, {}

    try:
        assert update(pipeline, 0, initial, interval=30) is None
        finish(pipeline)
        assert len(update(pipeline, 0, initial, interval=30)[0]) == 2
        active = pipeline.active
        active.trackers[0].update = Mock(return_value=(False, (0, 0, 0, 0)))
        if lose_all:
            active.trackers[1].update = Mock(return_value=(False, (0, 0, 0, 0)))
        items, _ = update(pipeline, 1, slow, interval=30)
        assert entered.wait(2)  # Refresh on frame 1, not after 30 frames.
        assert pipeline.active is active
        if lose_all:
            assert items == []
        else:
            assert [item['bbox'] for item in items] == [[51, 12, 71, 32]]
            # Still valid on the next frame while detection remains blocked.
            items, source = update(pipeline, 2, slow, interval=30)
            assert source == 'kcf'
            assert [item['bbox'] for item in items] == [[52, 12, 72, 32]]
        release.set()
        finish(pipeline)
        recovered, source = update(pipeline, 2, initial, interval=30)
        assert source == 'sam3' and len(recovered) == 2
    finally:
        release.set()
        pipeline.close()


def test_synchronous_detection_never_copies_replay_frames():
    class NoReplayCopy(np.ndarray):
        def copy(self, *args, **kwargs):
            raise AssertionError("Synchronous detection must not copy replay frames")

    pipeline = AsyncKCFProposals(MovingTracker)
    detector = Mock(side_effect=detect)
    try:
        for n in (0, 0, 1):
            image = pixels(n).view(NoReplayCopy)
            items, source = pipeline.update(image, n, "camera", 1, detector)
            assert source == "sam3"
            assert items[0]["bbox"] == proposals(n)[0]["bbox"]
        assert detector.call_count == 2  # Same-frame reads still reuse detection.
        assert not pipeline.history and pipeline.history_bytes == 0
        # Changing to asynchronous mode must still seed/replay from owned pixels.
        assert update(pipeline, 2) is None
        finish(pipeline)
        assert update(pipeline, 3)[0][0]["bbox"] == proposals(3)[0]["bbox"]
        assert pipeline.history
        assert update(pipeline, 4, interval=1)[0][0]["bbox"] == proposals(4)[0]["bbox"]
        assert not pipeline.history and pipeline.history_bytes == 0
    finally:
        pipeline.close()
