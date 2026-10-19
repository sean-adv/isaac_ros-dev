"""Detection-frame candidate initialization, bounded catch-up and atomic handoff."""

import threading
from types import SimpleNamespace
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


def seed(pipeline, interval=2):
    assert update(pipeline, 0, interval=interval) is None
    finish(pipeline)
    assert update(pipeline, 0, interval=interval)[1] == "sam3"


@pytest.fixture
def candidate_init():
    gate = SimpleNamespace(armed=False, entered=threading.Event(), release=threading.Event())

    class GatedTracker(MovingTracker):
        def init(self, image, box):
            super().init(image, box)
            if gate.armed and threading.current_thread().name == "sam3-kcf-replay":
                gate.entered.set()
                assert gate.release.wait(3), "Candidate initialization was never released"

    gate.factory = GatedTracker
    yield gate
    gate.release.set()


def test_kcf_continues_during_sam3_and_candidate_seeds_detection_frame():
    initialized = []

    class RecordTracker(MovingTracker):
        def init(self, image, box):
            initialized.append(int(image[0, 0, 0]))
            super().init(image, box)

    pipeline = AsyncKCFProposals(RecordTracker)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def slow(image):
        calls.append(int(image[0, 0, 0]))
        entered.set()
        assert release.wait(3)
        return detect(image)

    try:
        seed(pipeline, interval=100)
        update(pipeline, 1, interval=100)
        pipeline.force_detection = True
        assert update(pipeline, 2, slow, interval=100)[1] == "kcf"
        assert entered.wait(2)
        for n in (3, 4):
            items, source = update(pipeline, n, slow, interval=100)
            assert source == "kcf" and items[0]["bbox"][0] == 10+n
        assert calls == [2]
        release.set()
        finish(pipeline)
        assert initialized == [0, 2]  # Boxes and seed image both come from frame 2.
        metrics = pipeline.scheduler_metrics()
        assert metrics['seed_frame_key'] == 2
        assert metrics['seed_skipped_frames'] == 0
        assert metrics['replay_frames'] == 1
        items, source = update(pipeline, 4, interval=100)
        assert source == "sam3" and items[0]["bbox"][0] == 14
        assert pipeline.last_metrics == {"test": True}
        # Motion during detection is included before handoff.
        assert update(pipeline, 5, interval=100)[0][0]["bbox"][0] == 15
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


def test_slow_sam3_skips_to_latest_frame_without_detector_backlog():
    pipeline = AsyncKCFProposals(MovingTracker)
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
        assert pipeline.latest[0] == pipeline.serial
        np.testing.assert_array_equal(pipeline.latest[1], pixels(6))
        release.set()
        finish(pipeline)
        assert pipeline.ready is not None
        metrics = pipeline.scheduler_metrics()
        assert metrics['seed_skipped_frames'] == 0
        assert metrics['replay_frames'] == 1
        assert metrics['discard_reason'] is None
        assert update(pipeline, 6)[0][0]['bbox'][0] == 16
    finally:
        release.set()
        pipeline.close()


def test_arrivals_during_initialization_skip_to_latest(candidate_init):
    updated = []

    class RecordingTracker(candidate_init.factory):
        def update(self, image):
            if threading.current_thread().name == 'sam3-kcf-replay':
                updated.append(int(image[0, 0, 0]))
            return super().update(image)

    pipeline = AsyncKCFProposals(RecordingTracker)
    try:
        seed(pipeline)
        candidate_init.armed = True
        update(pipeline, 1)
        update(pipeline, 2)
        assert candidate_init.entered.wait(2)
        for n in (3, 4, 5):
            assert update(pipeline, n)[1] == 'kcf'
        np.testing.assert_array_equal(pipeline.latest[1], pixels(5))
        candidate_init.release.set()
        finish(pipeline)
        assert pipeline.ready is not None
        assert updated == [5]
        schedule = pipeline.scheduler_metrics()
        assert schedule['discard_reason'] is None
        assert schedule['replay_frames'] == 1
        assert schedule['catchup_skipped_frames'] == 2
    finally:
        candidate_init.release.set()
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


def test_slow_background_replay_does_not_hold_the_main_tracking_lock(candidate_init):
    replay_entered, release_replay = threading.Event(), threading.Event()
    detect_entered, release_detect = threading.Event(), threading.Event()

    class SlowReplayTracker(candidate_init.factory):
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
        candidate_init.armed = True
        release_detect.set()
        assert candidate_init.entered.wait(2)
        update(pipeline, 3, slow)
        candidate_init.release.set()
        assert replay_entered.wait(2)
        assert pipeline.lock.acquire(timeout=.5), "Replay must not hold the scheduler lock"
        pipeline.lock.release()
        items, source = update(pipeline, 4, slow)
        assert source == "kcf" and items[0]["bbox"][0] == 14
        release_replay.set()
        finish(pipeline)
        assert update(pipeline, 4)[1] == "sam3"
    finally:
        release_detect.set()
        candidate_init.release.set()
        release_replay.set()
        pipeline.close()


def test_failed_background_replay_is_discarded(candidate_init):
    entered, release = threading.Event(), threading.Event()

    class LostReplayTracker(candidate_init.factory):
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
        candidate_init.armed = True
        release.set()
        assert candidate_init.entered.wait(2)
        update(pipeline, 3, slow)
        candidate_init.release.set()
        finish(pipeline)
        assert pipeline.ready is None and pipeline.force_detection
    finally:
        release.set()
        candidate_init.release.set()
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


def test_slow_detection_skips_arbitrarily_many_intermediate_frames():
    entered, release = threading.Event(), threading.Event()
    initialized, replay_updates = [], []

    class CountReplayTracker(MovingTracker):
        def init(self, image, box):
            initialized.append(int(image[0, 0, 0]))
            super().init(image, box)

        def update(self, image):
            if threading.current_thread().name == 'sam3-kcf-replay':
                replay_updates.append(int(image[0, 0, 0]))
            return super().update(pixels(0))  # Many arrivals with a stationary object.

    pipeline = AsyncKCFProposals(CountReplayTracker)

    def slow(image):
        entered.set()
        assert release.wait(3)
        return detect(image)

    try:
        assert update(pipeline, 0, slow, interval=100) is None
        assert entered.wait(2)
        for n in range(1, 101):
            assert update(pipeline, n, slow, interval=100) is None
        assert pipeline.latest[0] == pipeline.serial
        assert pipeline.frames_since_submit == 0
        assert pipeline.last_key == 100
        release.set()
        finish(pipeline)
        assert not hasattr(pipeline, 'history')
        assert initialized == [0]
        assert replay_updates == [100]
        assert pipeline.ready is not None
        metrics = pipeline.scheduler_metrics()
        assert metrics['discard_reason'] is None
        assert metrics['catchup_skipped_frames'] == 99
        assert metrics['replay_frames'] == 1
        items, source = update(pipeline, 100, interval=100)
        assert source == 'sam3'
        assert items[0]['bbox'][0] == 10
    finally:
        release.set()
        pipeline.close()


def test_catchup_call_budget_preserves_foreground_and_waits_for_normal_interval(candidate_init):
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    calls = []

    class GatedUpdates(candidate_init.factory):
        def update(self, image):
            if threading.current_thread().name == 'sam3-kcf-replay':
                index = len(calls)
                calls.append(int(image[0, 0, 0]))
                entered[index].set()
                assert release[index].wait(3)
            return super().update(image)

    pipeline = AsyncKCFProposals(GatedUpdates, max_replay_frames=2, replay_budget_ms=2000)
    detector = Mock(side_effect=detect)
    try:
        seed(pipeline, interval=100)
        active = pipeline.active
        candidate_init.armed = True
        pipeline.force_detection = True
        update(pipeline, 1, detector, interval=100)
        assert candidate_init.entered.wait(2)
        update(pipeline, 2, detector, interval=100)
        candidate_init.release.set()
        assert entered[0].wait(2)
        update(pipeline, 3, detector, interval=100)
        release[0].set()
        assert entered[1].wait(2)
        update(pipeline, 4, detector, interval=100)
        release[1].set()
        finish(pipeline)
        assert pipeline.ready is None
        assert pipeline.active is active
        assert not pipeline.force_detection
        assert calls == [2, 3]
        assert pipeline.scheduler_metrics()['discard_reason'] == 'frame_budget'
        assert pipeline.scheduler_metrics()['replay_frames'] == 2
        assert update(pipeline, 5, detector, interval=100)[0][0]['bbox'][0] == 15
        assert detector.call_count == 1
    finally:
        candidate_init.release.set()
        for event in release:
            event.set()
        pipeline.close()


def test_replay_time_budget_stops_before_next_update(monkeypatch, candidate_init):
    from types import SimpleNamespace
    from dinov3_cases import async_kcf
    clock = [0.0]
    monkeypatch.setattr(async_kcf, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    entered, release = threading.Event(), threading.Event()
    update_entered, update_release = threading.Event(), threading.Event()

    class TimedTracker(candidate_init.factory):
        def update(self, image):
            if threading.current_thread().name == 'sam3-kcf-replay':
                update_entered.set()
                assert update_release.wait(3)
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
        candidate_init.armed = True
        release.set()
        assert candidate_init.entered.wait(2)
        update(pipeline, 3, slow)
        update(pipeline, 4, slow)
        candidate_init.release.set()
        assert update_entered.wait(2)
        update(pipeline, 5, slow)
        update_release.set()
        finish(pipeline)
        assert pipeline.ready is None
        metrics = pipeline.scheduler_metrics()
        assert metrics['discard_reason'] == 'time_budget'
        assert metrics['replay_frames'] == 1
        assert metrics['replay_ms'] == pytest.approx(60)
        assert not pipeline.force_detection
        assert pipeline.active.proposals[0]['bbox'][0] == 15
    finally:
        release.set()
        candidate_init.release.set()
        update_release.set()
        pipeline.close()


def test_new_arrivals_during_catchup_skip_to_newest_each_time(candidate_init):
    entered, release = threading.Event(), threading.Event()
    replay_entered, replay_release = threading.Event(), threading.Event()
    replayed = []

    class BlockFirstReplay(candidate_init.factory):
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
        candidate_init.armed = True
        release.set()
        assert candidate_init.entered.wait(2)
        update(pipeline, 3, slow)
        candidate_init.release.set()
        assert replay_entered.wait(2)
        for n in (4, 5, 6):
            items, source = update(pipeline, n, slow)
            assert source == 'kcf' and items[0]['bbox'][0] == 10+n
        replay_release.set()
        finish(pipeline)
        assert replayed == [3, 6]
        assert pipeline.ready is not None
        metrics = pipeline.scheduler_metrics()
        assert metrics['discard_reason'] is None
        assert metrics['replay_frames'] == 2
        assert metrics['catchup_skipped_frames'] == 2
        assert update(pipeline, 6)[0][0]['bbox'][0] == 16
    finally:
        release.set()
        candidate_init.release.set()
        replay_release.set()
        pipeline.close()


def test_context_change_cancels_remaining_replay_batch(candidate_init):
    detect_entered, detect_release = threading.Event(), threading.Event()
    replay_entered, replay_release = threading.Event(), threading.Event()
    replayed = []

    class BlockFirstReplay(candidate_init.factory):
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
        candidate_init.armed = True
        detect_release.set()
        assert candidate_init.entered.wait(2)
        update(pipeline, 3, slow)
        update(pipeline, 4, slow)
        candidate_init.release.set()
        assert replay_entered.wait(2)
        assert update(pipeline, 5, context='new-source') is None
        replay_release.set()
        finish(pipeline)
        assert replayed == [4]
        assert pipeline.ready is None
        assert not pipeline.active.initialized
        assert pipeline.last_metrics == {}
        assert update(pipeline, 6, context='new-source') is None
        finish(pipeline)
        assert update(pipeline, 6, context='new-source')[0][0]['bbox'][0] == 16
    finally:
        detect_release.set()
        candidate_init.release.set()
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
        assert pipeline.latest is None
        # Changing to asynchronous mode must still seed/replay from owned pixels.
        assert update(pipeline, 2) is None
        finish(pipeline)
        assert update(pipeline, 3)[0][0]["bbox"] == proposals(3)[0]["bbox"]
        assert pipeline.latest is not None
        assert update(pipeline, 4, interval=1)[0][0]["bbox"] == proposals(4)[0]["bbox"]
        assert pipeline.latest is None
    finally:
        pipeline.close()


def test_seed_snapshot_survives_input_reuse_and_new_arrivals():
    entered, release = threading.Event(), threading.Event()
    initialized = []

    class RecordTracker(MovingTracker):
        def init(self, image, box):
            initialized.append(int(image[0, 0, 0]))
            super().init(image, box)

    pipeline = AsyncKCFProposals(RecordTracker)
    original = pixels(2)

    def slow(image):
        entered.set()
        assert release.wait(3)
        np.testing.assert_array_equal(image, pixels(2))
        return detect(image)

    try:
        assert pipeline.update(original, 2, "camera", 100, slow) is None
        assert entered.wait(2)
        original[:] = 99  # Producer reuses its buffer while SAM3 is running.
        for n in (3, 4, 5):
            assert update(pipeline, n, slow, interval=100) is None
        release.set()
        finish(pipeline)
        assert initialized == [2]
        items, source = update(pipeline, 5, interval=100)
        assert source == "sam3"
        assert items[0]["bbox"] == proposals(5)[0]["bbox"]
        assert pipeline.scheduler_metrics()["replay_frames"] == 1
    finally:
        release.set()
        pipeline.close()


def test_slow_dino_shares_sam3_schedule_and_hands_off_labels_atomically():
    pipeline = AsyncKCFProposals(MovingTracker, roi_padding=None)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def recognize(image, boxes, key):
        calls.append((key, threading.current_thread().name))
        if key == 2:
            entered.set()
            assert release.wait(3)
        return [dict(p, label=f'medicine-{key}') for p in boxes], {'frame_id': key}

    def tick(n):
        return pipeline.update(pixels(n), n, 'camera', 2, detect, recognize=recognize)

    try:
        assert tick(0) is None
        finish(pipeline)
        assert tick(0)[0][0]['label'] == 'medicine-0'
        tick(1)
        items, source = tick(2)
        assert entered.wait(2)
        assert source == 'kcf' and items[0]['label'] == 'medicine-0'
        for n in (3, 4):
            items, source = tick(n)
            assert source == 'kcf' and items[0]['bbox'] == proposals(n)[0]['bbox']
            assert items[0]['label'] == 'medicine-0'
            assert pipeline.last_recognition == {'frame_id': 0}
        release.set()
        finish(pipeline)
        # Do not schedule another job while checking the completed handoff.
        pipeline.frames_since_submit = 0
        items, source = tick(4)
        assert source == 'sam3' and items[0]['label'] == 'medicine-2'
        assert items[0]['bbox'] == proposals(4)[0]['bbox']
        assert pipeline.last_recognition == {'frame_id': 2}
        assert calls == [(0, 'sam3-kcf-replay'), (2, 'sam3-kcf-replay')]
    finally:
        release.set()
        pipeline.close()


def test_cancelled_dino_result_cannot_publish_into_new_context():
    pipeline = AsyncKCFProposals(MovingTracker)
    entered, release = threading.Event(), threading.Event()

    def recognize(image, boxes, key):
        entered.set()
        assert release.wait(3)
        return [dict(p, label='old') for p in boxes], {'frame_id': key}

    try:
        pipeline.update(pixels(0), 0, 'old', 30, detect, recognize=recognize)
        assert entered.wait(2)
        assert pipeline.update(pixels(1), 1, 'new', 30, detect) is None
        release.set()
        finish(pipeline)
        assert pipeline.ready is None and pipeline.last_recognition is None
        assert not pipeline.active.initialized
    finally:
        release.set()
        pipeline.close()


def test_partial_loss_preserves_survivor_label_without_dino_rerun():
    pipeline = AsyncKCFProposals(MovingTracker, roi_padding=None)
    calls = []

    def detect_two(image):
        return proposals() + [dict(proposals()[0], bbox=[50, 12, 70, 32])], {}

    def recognize(image, boxes, key):
        calls.append(key)
        return [dict(p, label=label) for p, label in zip(boxes, ['red', 'blue'])], key

    try:
        pipeline.update(pixels(0), 0, 'camera', 30, detect_two, recognize=recognize)
        finish(pipeline)
        pipeline.update(pixels(0), 0, 'camera', 30, detect_two, recognize=recognize)
        pipeline.active.trackers[0].update = Mock(return_value=(False, (0, 0, 0, 0)))
        entered, release = threading.Event(), threading.Event()

        def slow(image):
            entered.set()
            assert release.wait(3)
            return detect_two(image)

        items, source = pipeline.update(pixels(1), 1, 'camera', 30, slow, recognize=recognize)
        assert entered.wait(2)
        assert source == 'kcf' and [p['label'] for p in items] == ['blue']
        assert calls == [0]
    finally:
        release.set()
        pipeline.close()


@pytest.mark.parametrize('context', ['startup', 'settings-changed'])
def test_first_result_wait_keeps_seed_frame_and_recognition_together(context):
    pipeline = AsyncKCFProposals(MovingTracker, max_replay_frames=2)
    entered, release = threading.Event(), threading.Event()
    results = []
    payload = {'label': 'medicine'}

    def slow_recognize(image, boxes, key):
        entered.set()
        assert release.wait(3)
        assert key == 10
        np.testing.assert_array_equal(image, pixels(10))
        return boxes, payload

    def run():
        results.append(pipeline.update(
            pixels(10), 10, context, 30, detect,
            recognize=slow_recognize, wait_for_initial_result=True))

    worker = threading.Thread(target=run)
    try:
        if context == 'settings-changed':
            seed(pipeline, interval=30)
        worker.start()
        assert entered.wait(2)
        # A camera thread can keep receiving frames independently; the inference
        # caller waits without admitting them into the first candidate's history.
        assert worker.is_alive()
        assert pipeline.lock.acquire(timeout=.5)
        pipeline.lock.release()
        release.set()
        worker.join(3)
        assert not worker.is_alive()
        assert results[0][1] == 'sam3'
        assert results[0][0][0]['bbox'][0] == 20
        assert pipeline.last_recognition is payload
        assert pipeline.scheduler_metrics()['discard_reason'] is None
        assert pipeline.scheduler_metrics()['replay_frames'] == 0
        # Subsequent frames use KCF and preserve the initialized context.
        assert pipeline.update(pixels(11), 11, context, 30, detect,
                               wait_for_initial_result=True)[1] == 'kcf'
    finally:
        release.set()
        worker.join(3)
        pipeline.close()


def test_initial_replay_timeout_remains_retryable(monkeypatch):
    from dinov3_cases import async_kcf
    clock = [0.0]
    monkeypatch.setattr(async_kcf, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    entered, release = threading.Event(), threading.Event()

    class TimedTracker(MovingTracker):
        def update(self, image):
            if threading.current_thread().name == "sam3-kcf-replay":
                update(pipeline, 3, slow)
            clock[0] += .1
            return super().update(image)

    def slow(image):
        entered.set()
        assert release.wait(3)
        return detect(image)

    pipeline = AsyncKCFProposals(TimedTracker, replay_budget_ms=50)
    try:
        update(pipeline, 0, slow)
        assert entered.wait(2)
        update(pipeline, 1, slow)
        update(pipeline, 2, slow)
        release.set()
        finish(pipeline)
        assert pipeline.scheduler_metrics()['discard_reason'] == 'time_budget'
        assert not pipeline.active.initialized
        assert pipeline.has_result()
        assert pipeline.force_detection
        result = pipeline.update(pixels(3), 3, 'camera', 2, detect,
                                 wait_for_initial_result=True)
        assert result[1] == 'sam3'
    finally:
        release.set()
        pipeline.close()


def test_roi_limit_survives_background_handoff_and_context_reset():
    shapes = []

    class StationaryTracker:
        def init(self, image, box):
            shapes.append(image.shape)
            self.box = box

        def update(self, image):
            shapes.append(image.shape)
            return True, self.box

    image = np.zeros((800, 1200, 3), dtype=np.uint8)
    boxes = [dict(bbox=[400, 200, 600, 400], area=40000, label='known')]
    recognized_shapes = []

    def recognize(seed, items, key):
        recognized_shapes.append(seed.shape)
        return items, {'frame_id': key}

    pipeline = AsyncKCFProposals(StationaryTracker, roi_padding=.5, max_roi_side=200)
    try:
        for context in ('camera-a', 'camera-b'):
            result = pipeline.update(image, (context, 0), context, 30,
                                     lambda frame: (boxes, {}), recognize=recognize,
                                     wait_for_initial_result=True)
            assert result[0][0]['bbox'] == boxes[0]['bbox']
            assert pipeline.active.max_roi_side == 200
            result = pipeline.update(image, (context, 1), context, 30,
                                     lambda frame: (boxes, {}))
            assert result[1] == 'kcf'
            assert result[0][0]['bbox'] == boxes[0]['bbox']
            assert result[0][0]['label'] == 'known'
        assert shapes and all(shape == (200, 200, 3) for shape in shapes)
        assert recognized_shapes == [(800, 1200, 3)] * 2
    finally:
        pipeline.close()
