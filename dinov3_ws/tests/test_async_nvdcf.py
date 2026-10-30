"""Timing and identity contracts without requiring DeepStream on a test host."""
import threading

import numpy as np
import pytest

from dinov3_cases.async_nvdcf import AsyncNvDCFProposals


class FakeNvDCF:
    def __init__(self):
        self.proposals = []
        self.metadata = {}
        self.frame_key = None
        self.closed = False
        self.calls = []

    def update(self, pixels, key, detections=None):
        self.calls.append((key, detections))
        if detections is not None:
            self.metadata = {i: dict(p) for i, p in enumerate(detections)}
        self.frame_key = key
        self.proposals = [dict(p, nvdcf_id=i, bbox=[key, 0, key+10, 10])
                          for i, p in self.metadata.items()]
        return self.proposals

    def close(self):
        self.closed = True


@pytest.fixture
def pixels():
    return np.zeros((32, 32, 3), dtype=np.uint8)


def detect(_):
    return [dict(bbox=[0, 0, 10, 10], detection_score=.9, label='medicine')], {'test': True}


def test_live_timing_ignores_seed_and_duplicate_rgb(pixels):
    from dinov3_cases.timing import ModelTimings

    timings = ModelTimings()
    scheduler = AsyncNvDCFProposals(factory=FakeNvDCF, model_timings=timings)
    try:
        scheduler.update(pixels, (0, 1), 'camera', 100, detect, wait_for_initial_result=True)
        assert not timings.snapshot()['nvdcf']['valid']
        scheduler.update(pixels, (0, 2), 'camera', 100, detect)
        sample = timings.snapshot()['nvdcf']
        assert sample['valid'] and sample['source_frame_id'] == 2
        scheduler.update(pixels, (0, 2), 'camera', 100, detect)
        assert timings.snapshot()['nvdcf']['last_ms'] == sample['last_ms']
        assert not timings.snapshot()['nvdcf']['busy']
    finally:
        scheduler.close()


def test_seed_recognition_and_catchup_handoff_use_current_frame(pixels):
    started, release = threading.Event(), threading.Event()
    trackers = []

    def factory():
        tracker = FakeNvDCF()
        trackers.append(tracker)
        return tracker

    def delayed(image):
        started.set()
        assert release.wait(3)
        return detect(image)

    def recognize(image, boxes, key):
        assert key == (1, 1)
        return [dict(p, label='classified') for p in boxes], {'frame_id': 1}

    scheduler = AsyncNvDCFProposals(factory=factory)
    try:
        assert scheduler.update(pixels, (1, 1), 'camera', 30, delayed, recognize=recognize) is None
        assert started.wait(3)
        assert scheduler.update(pixels, (1, 2), 'camera', 30, delayed) is None
        release.set()
        scheduler.thread.join(3)
        result, source = scheduler.update(pixels, (1, 3), 'camera', 30, delayed)
        assert source == 'sam3'
        assert result[0]['bbox'] == [3, 0, 13, 10]
        assert result[0]['label'] == 'classified'
        assert [key for key, _ in trackers[0].calls] == [1, 2, 3]
        public_id = result[0]['track_id']
        result, source = scheduler.update(pixels, (1, 4), 'camera', 30, detect)
        assert source == 'nvdcf' and result[0]['track_id'] == public_id
        assert result[0]['bbox'] == [4, 0, 14, 10]
    finally:
        release.set()
        scheduler.close()
    scheduler._reaper.drain()
    assert all(t.closed for t in trackers)


def test_refresh_preserves_public_id_and_empty_detection_counts_misses(pixels):
    scheduler = AsyncNvDCFProposals(factory=FakeNvDCF)
    try:
        first, _ = scheduler.update(pixels, 1, 'camera', 100, detect, wait_for_initial_result=True)
        identity = first[0]['track_id']
        scheduler.force_detection = True
        scheduler.update(pixels, 2, 'camera', 100, detect)
        scheduler.thread.join(3)
        result, source = scheduler.update(pixels, 3, 'camera', 100, detect)
        assert source == 'sam3' and result[0]['track_id'] == identity
        scheduler.force_detection = True
        scheduler.update(pixels, 4, 'camera', 100, lambda _: ([], {}))
        scheduler.thread.join(3)
        result, source = scheduler.update(pixels, 5, 'camera', 100, detect)
        assert source == 'sam3' and result[0]['track_id'] == identity
        assert result[0]['sam3_misses'] == 1
        for frame in (6, 8):
            result, _ = refresh(scheduler, pixels, frame, [])
        assert result == []
    finally:
        scheduler.close()


def test_context_reset_discards_inflight_detector_result(pixels):
    started, release = threading.Event(), threading.Event()
    def delayed(image):
        started.set()
        assert release.wait(3)
        return detect(image)
    scheduler = AsyncNvDCFProposals(factory=FakeNvDCF)
    try:
        scheduler.update(pixels, (1, 1), 'old', 30, delayed)
        assert started.wait(3)
        scheduler.update(pixels, (2, 1), 'new', 30, detect)
        release.set()
        scheduler.thread.join(3)
        assert scheduler.ready is None
        result, _ = scheduler.update(pixels, (2, 2), 'new', 30, detect, wait_for_initial_result=True)
        assert result[0]['label'] == 'medicine'
        assert scheduler.update(pixels, (1, 1), 'new', 30, detect) is None
    finally:
        release.set()
        scheduler.close()


def test_synchronous_mode_does_not_start_native_backend(pixels):
    scheduler = AsyncNvDCFProposals(factory=lambda: pytest.fail('native tracker started in sync mode'))
    try:
        result, source = scheduler.update(pixels, 1, 'camera', 1, detect)
        assert source == 'sam3' and result[0]['label'] == 'medicine'
        first_id = result[0]['track_id']
        result, _ = scheduler.update(pixels, 2, 'other', 1, detect)
        assert result[0]['track_id'] != first_id
    finally:
        scheduler.close()


class SpatialNvDCF(FakeNvDCF):
    """Move each seed independently; native IDs restart at zero per context."""
    def update(self, pixels, key, detections=None):
        assert not self.closed
        self.calls.append((key, detections))
        if detections is not None:
            self.metadata = {i: dict(p) for i, p in enumerate(detections)}
        delta = key - self.frame_key if self.frame_key is not None else 0
        self.proposals = []
        for native_id, item in self.metadata.items():
            x0, y0, x1, y1 = item['bbox']
            output = dict(item, nvdcf_id=native_id, bbox=[x0+delta, y0, x1+delta, y1])
            self.proposals.append(output)
            self.metadata[native_id] = dict(output)
        self.frame_key = key
        return self.proposals


def box(x, label='medicine', size=10):
    return dict(bbox=[x, 0, x+size, size], detection_score=.9, label=label)


def refresh(scheduler, pixels, frame, boxes):
    scheduler.force_detection = True
    detector = lambda _: (boxes, {})
    scheduler.update(pixels, frame, 'camera', 100, detector)
    scheduler.thread.join(3)
    assert not scheduler.thread.is_alive()
    return scheduler.update(pixels, frame, 'camera', 100, detector)


def seed_spatial(pixels, tracker_class=SpatialNvDCF, **kwargs):
    trackers = []
    def factory():
        tracker = tracker_class()
        trackers.append(tracker)
        return tracker
    scheduler = AsyncNvDCFProposals(factory=factory, **kwargs)
    result, _ = scheduler.update(pixels, 1, 'camera', 100,
                                lambda _: ([box(0, 'A'), box(100, 'B')], {}),
                                wait_for_initial_result=True)
    return scheduler, trackers, result


def test_partial_handoff_retains_native_state_and_independent_ids(pixels):
    scheduler, trackers, first = seed_spatial(pixels)
    a_id, b_id = [p['track_id'] for p in first]
    try:
        result, _ = refresh(scheduler, pixels, 2, [box(1, 'A-new')])
        assert [(p['track_id'], p['label']) for p in result] == [(a_id, 'A-new'), (b_id, 'B')]
        assert [p['nvdcf_id'] for p in result] == [0, 1]
        assert result[1]['sam3_misses'] == 1
        scheduler._reaper.drain()
        assert not trackers[0].closed
        assert len(trackers[0].metadata) == 1  # Replaced A cannot reappear.
        result, _ = scheduler.update(pixels, 3, 'camera', 100, detect)
        assert [p['bbox'][0] for p in result] == [2, 102]
        assert result[1]['sam3_misses'] == 1  # Camera frames are not detector misses.
        # Match B in the original context and add C, whose native ID collides
        # with retained A's native ID in a different context.
        result, _ = refresh(scheduler, pixels, 4, [box(103, 'B-new'), box(200, 'C')])
        assert [(p['track_id'], p['label']) for p in result[:2]] == [(b_id, 'B-new'), (3, 'C')]
        assert result[2]['track_id'] == a_id
        assert result[0]['nvdcf_id'] == result[2]['nvdcf_id'] == 0
        scheduler._reaper.drain()
        assert trackers[0].closed and not trackers[1].closed
        assert result[0]['sam3_misses'] == 0
        # Reacquire all objects: obsolete contexts must close, with IDs preserved.
        result, _ = refresh(scheduler, pixels, 5, [box(4, 'A'), box(104, 'B'), box(201, 'C')])
        assert [p['track_id'] for p in result] == [a_id, b_id, 3]
        assert all(p['sam3_misses'] == 0 for p in result)
        scheduler._reaper.drain()
        assert all(t.closed for t in trackers[:-1])
        assert len(scheduler.active.contexts) == 1
    finally:
        scheduler.close()
    scheduler._reaper.drain()
    assert all(t.closed for t in trackers)


def test_empty_results_retire_only_at_configured_miss_limit(pixels):
    scheduler, trackers, first = seed_spatial(pixels, max_missed_detections=2)
    try:
        result, _ = refresh(scheduler, pixels, 2, [])
        assert [p['track_id'] for p in result] == [p['track_id'] for p in first]
        assert all(p['sam3_misses'] == 1 and p['track_status'] == 'coasting' for p in result)
        scheduler._reaper.drain()
        assert trackers[1].closed and not trackers[0].closed
        # Reprocessing the same frame never adds a miss or revives an empty context.
        for _ in range(3):
            result, _ = scheduler.update(pixels, 2, 'camera', 100, detect)
            assert all(p['sam3_misses'] == 1 for p in result)
        result, _ = refresh(scheduler, pixels, 3, [])
        assert result == [] and not scheduler.active.contexts
        scheduler._reaper.drain()
        assert all(t.closed for t in trackers)
    finally:
        scheduler.close()


def test_unknown_matched_candidate_cannot_leave_old_known_track(pixels):
    scheduler, trackers, first = seed_spatial(pixels)
    try:
        result, _ = refresh(scheduler, pixels, 2, [box(1, 'unknown')])
        assert [p['track_id'] for p in result] == [first[1]['track_id']]
        assert result[0]['label'] == 'B'
        result, _ = scheduler.update(pixels, 3, 'camera', 100, detect)
        assert [p['label'] for p in result] == ['B']
    finally:
        scheduler.close()


def test_containment_filters_after_merging_retained_tracks(pixels):
    scheduler, trackers, first = seed_spatial(pixels, containment_threshold=.8)
    try:
        # New box is >4x A's area, so association rejects it, but containment
        # must remove A after the candidate and retained boxes have merged.
        result, _ = refresh(scheduler, pixels, 2, [box(0, 'large', size=30)])
        assert {p['label'] for p in result} == {'large', 'B'}
        assert first[0]['track_id'] not in scheduler.identities.recent
        result, _ = scheduler.update(pixels, 3, 'camera', 100, detect)
        assert {p['label'] for p in result} == {'large', 'B'}
    finally:
        scheduler.close()


def test_reset_closes_all_retained_contexts_and_clears_misses(pixels):
    scheduler, trackers, first = seed_spatial(pixels)
    try:
        refresh(scheduler, pixels, 2, [box(1)])
        assert len(scheduler.active.contexts) == 2
        result, _ = scheduler.update(pixels, 3, 'new-camera', 100,
                                    lambda _: ([box(0)], {}), wait_for_initial_result=True)
        scheduler._reaper.drain()
        assert all(t.closed for t in trackers[:-1])
        assert result[0]['sam3_misses'] == 0
        assert result[0]['track_id'] not in {p['track_id'] for p in first}
    finally:
        scheduler.close()


def test_stalled_detection_does_not_count_misses(pixels):
    scheduler, trackers, first = seed_spatial(pixels)
    started, release = threading.Event(), threading.Event()
    def stalled(_):
        started.set()
        assert release.wait(3)
        return [], {}
    try:
        scheduler.force_detection = True
        scheduler.update(pixels, 2, 'camera', 100, stalled)
        assert started.wait(3)
        for frame in range(3, 8):
            result, _ = scheduler.update(pixels, frame, 'camera', 100, detect)
            assert all(p['sam3_misses'] == 0 for p in result)
        release.set()
        scheduler.thread.join(3)
        result, _ = scheduler.update(pixels, 8, 'camera', 100, detect)
        assert all(p['sam3_misses'] == 1 for p in result)
    finally:
        release.set()
        scheduler.close()


class OccludedNvDCF(SpatialNvDCF):
    """Model native shadow output: metadata survives an absent output record."""
    def __init__(self):
        super().__init__()
        self.missing = {}

    def update(self, pixels, key, detections=None):
        super().update(pixels, key, detections)
        self.proposals = [p for p in self.proposals
                          if p['nvdcf_id'] not in self.missing.get(key, set())]
        return self.proposals


def test_native_shadow_recovers_same_id_label_and_lifecycle(pixels):
    scheduler, trackers, first = seed_spatial(pixels, tracker_class=OccludedNvDCF,
                                             max_missed_frames=2)
    try:
        trackers[0].missing = {2: {0}, 3: {0}}
        scheduler.busy = True  # Model a pending detector while native output recovers.
        for frame in (2, 3):
            result, _ = scheduler.update(pixels, frame, 'camera', 100, detect)
            assert [p['label'] for p in result] == ['B']
            assert len(scheduler.active.hidden) == 1
            assert scheduler.active.hidden[0]['track_id'] == first[0]['track_id']
            assert set(trackers[0].metadata) == {0, 1}
            scheduler._reaper.drain()
            assert not trackers[0].closed
        result, _ = scheduler.update(pixels, 4, 'camera', 100, detect)
        assert [(p['track_id'], p['nvdcf_id'], p['label']) for p in result] == [
            (p['track_id'], p['nvdcf_id'], p['label']) for p in first]
        assert result[0]['sam3_hits'] == first[0]['sam3_hits']
        assert result[0]['sam3_misses'] == 0
        assert not scheduler.active.hidden
    finally:
        scheduler.close()


@pytest.mark.parametrize('grace', [0, 2])
def test_shadow_expires_after_grace_unique_frames_and_cannot_revive(pixels, grace):
    scheduler, trackers, first = seed_spatial(pixels, tracker_class=OccludedNvDCF,
                                             max_missed_frames=grace)
    try:
        trackers[0].missing = {frame: {0, 1} for frame in range(2, grace+3)}
        scheduler.busy = True
        for frame in range(2, grace+2):
            for _ in range(3):
                result, _ = scheduler.update(pixels, frame, 'camera', 100, detect)
                assert result == []
                assert len(scheduler.active.hidden) == 2
                scheduler._reaper.drain()
                assert not trackers[0].closed
        result, _ = scheduler.update(pixels, grace+2, 'camera', 100, detect)
        scheduler._reaper.drain()
        assert result == [] and trackers[0].closed
        assert not scheduler.active.contexts
        assert not scheduler.lifecycle.states
        assert not scheduler.identities.recent
        result, _ = scheduler.update(pixels, grace+3, 'camera', 100, detect)
        assert result == []  # Expired metadata cannot reappear.
    finally:
        scheduler.close()


def test_hidden_owner_survives_empty_handoffs_without_resetting_frame_age(pixels):
    scheduler, trackers, first = seed_spatial(pixels, tracker_class=OccludedNvDCF,
                                             max_missed_frames=3,
                                             max_missed_detections=1)
    try:
        trackers[0].missing = {frame: {0} for frame in range(2, 6)}
        # Empty SAM3 results must not delete occluded A via the separate SAM3
        # miss limit, or reset its last-seen serial on every new generation.
        for frame in (2, 3, 4):
            result, _ = refresh(scheduler, pixels, frame, [])
            assert result == []
            assert len(scheduler.active.hidden) == 1
            assert scheduler.active.hidden[0]['track_id'] == first[0]['track_id']
            assert scheduler.active.last_seen == {first[0]['track_id']: 1}
            assert scheduler.lifecycle.states[first[0]['track_id']]['misses'] == 0
        scheduler.busy = True
        result, _ = scheduler.update(pixels, 5, 'camera', 100, detect)
        scheduler._reaper.drain()
        assert result == [] and trackers[0].closed
    finally:
        scheduler.close()


def test_sam3_reacquires_hidden_owner_once_and_retires_old_native_handle(pixels):
    scheduler, trackers, first = seed_spatial(pixels, tracker_class=OccludedNvDCF)
    try:
        trackers[0].missing = {2: {0}, 3: {0}}
        result, _ = refresh(scheduler, pixels, 2, [box(1, 'A-new')])
        a = next(p for p in result if p['label'] == 'A-new')
        assert a['track_id'] == first[0]['track_id']
        assert len(result) == 2 and not scheduler.active.hidden
        assert set(trackers[0].metadata) == {1}
        result, _ = scheduler.update(pixels, 3, 'camera', 100, detect)
        assert len({p['track_id'] for p in result}) == len(result) == 2
        assert [p['label'] for p in result] == ['A-new', 'B']
    finally:
        scheduler.close()


def test_partial_loss_requests_detection_before_normal_interval(pixels):
    scheduler, trackers, first = seed_spatial(pixels, tracker_class=OccludedNvDCF)
    entered, release = threading.Event(), threading.Event()
    def stalled(_):
        entered.set()
        assert release.wait(3)
        return [], {}
    try:
        trackers[0].missing = {2: {0}}
        result, _ = scheduler.update(pixels, 2, 'camera', 100, stalled)
        assert [p['label'] for p in result] == ['B']
        assert entered.wait(3)
        assert scheduler.busy and len(scheduler.active.hidden) == 1
    finally:
        release.set()
        scheduler.close()


def test_context_reset_and_close_release_hidden_contexts(pixels):
    scheduler, trackers, first = seed_spatial(pixels, tracker_class=OccludedNvDCF)
    try:
        trackers[0].missing = {2: {0, 1}}
        scheduler.busy = True
        scheduler.update(pixels, 2, 'camera', 100, detect)
        assert len(scheduler.active.hidden) == 2
        scheduler.busy = False
        result, _ = scheduler.update(pixels, 3, 'new-camera', 100,
                                    lambda _: ([box(0)], {}), wait_for_initial_result=True)
        scheduler._reaper.drain()
        assert trackers[0].closed and not scheduler.active.hidden
        assert result[0]['track_id'] not in {p['track_id'] for p in first}
    finally:
        scheduler.close()
    scheduler._reaper.drain()
    assert all(t.closed for t in trackers)


@pytest.mark.parametrize('value', [-1, True, 1.5])
def test_invalid_shadow_grace_is_rejected(value):
    with pytest.raises(ValueError, match='nvdcf_max_missed_frames'):
        AsyncNvDCFProposals(max_missed_frames=value)


def test_blocked_cleanup_does_not_stall_handoff_or_live_tracking(pixels):
    entered, release, committed = threading.Event(), threading.Event(), threading.Event()
    trackers = []

    class SlowClose(SpatialNvDCF):
        def close(self):
            entered.set()
            assert release.wait(5)
            super().close()

    def factory():
        tracker = SlowClose() if not trackers else SpatialNvDCF()
        trackers.append(tracker)
        return tracker

    scheduler = AsyncNvDCFProposals(factory=factory)
    commit = None
    try:
        first, _ = scheduler.update(pixels, 1, 'camera', 100,
                                    lambda _: ([box(0, 'old')], {}),
                                    wait_for_initial_result=True)
        identity = first[0]['track_id']
        scheduler.force_detection = True
        scheduler.update(pixels, 2, 'camera', 100, lambda _: ([box(1, 'new')], {}))
        scheduler.thread.join(3)
        outputs = []

        def handoff():
            outputs.append(scheduler.update(pixels, 3, 'camera', 100, detect))
            committed.set()

        commit = threading.Thread(target=handoff)
        commit.start()
        assert entered.wait(2)
        assert committed.wait(1), 'retired context shutdown blocked the handoff'
        result, source = outputs[0]
        assert source == 'sam3'
        assert [(p['track_id'], p['label'], p['bbox']) for p in result] == [
            (identity, 'new', [2, 0, 12, 10])]
        assert not trackers[0].closed
        result, source = scheduler.update(pixels, 4, 'camera', 100, detect)
        assert source == 'nvdcf'
        assert (result[0]['track_id'], result[0]['label'], result[0]['bbox']) == (
            identity, 'new', [3, 0, 13, 10])
    finally:
        release.set()
        if commit is not None:
            commit.join(3)
        scheduler.close()
    assert all(t.closed for t in trackers)
    assert scheduler._reaper.thread is None


def test_shutdown_drains_cleanup_even_when_a_retired_close_fails(pixels, caplog):
    from dinov3_cases.async_nvdcf import _TrackerReaper
    reaper = _TrackerReaper()
    entered, release, done = threading.Event(), threading.Event(), threading.Event()

    class Broken:
        def close(self):
            raise RuntimeError('cleanup failure')

    class Blocking(FakeNvDCF):
        def close(self):
            entered.set()
            assert release.wait(5)
            super().close()

    tracker = Blocking()
    reaper.retire(Broken())
    reaper.retire(tracker)
    closer = threading.Thread(target=lambda: (reaper.close(), done.set()))
    try:
        assert entered.wait(2)
        closer.start()
        assert not done.wait(.05), 'shutdown must wait for native cleanup'
    finally:
        release.set()
        if closer.ident is not None:
            closer.join(3)
        else:
            reaper.close()
    assert tracker.closed and done.is_set()
    assert reaper.thread is None
    assert 'cleanup failure' in caplog.text


def test_deferred_geometry_tracks_before_dino_and_labels_never_rewind_boxes(pixels):
    entered, release = threading.Event(), threading.Event()
    scheduler = AsyncNvDCFProposals(factory=SpatialNvDCF, defer_recognition=True)

    def recognize(image, proposals, key):
        entered.set()
        assert release.wait(5)
        return [dict(p, label='classified', score=.95) for p in proposals], {'frame_id': key}

    try:
        first, source = scheduler.update(pixels, 1, 'camera', 100,
                                        lambda _: ([box(0)], {}), recognize=recognize,
                                        wait_for_initial_result=True)
        assert entered.wait(2) and scheduler.busy
        assert source == 'sam3' and first[0]['recognition_pending']
        identity = first[0]['track_id']
        tracked, source = scheduler.update(pixels, 2, 'camera', 100, detect)
        assert source == 'nvdcf' and tracked[0]['bbox'] == [1, 0, 11, 10]
        release.set()
        scheduler.thread.join(3)
        assert scheduler.has_result()
        result, _ = scheduler.update(pixels, 3, 'camera', 100, detect)
        assert (result[0]['track_id'], result[0]['label'], result[0]['bbox']) == (
            identity, 'classified', [2, 0, 12, 10])
        assert result[0]['recognition_pending'] is False
        result, _ = scheduler.update(pixels, 4, 'camera', 100, detect)
        assert result[0]['label'] == 'classified'  # Native metadata also changed.
    finally:
        release.set()
        scheduler.close()


def test_deferred_refresh_keeps_old_label_then_updates_same_identity(pixels):
    scheduler = AsyncNvDCFProposals(factory=SpatialNvDCF, defer_recognition=True)
    entered, release = threading.Event(), threading.Event()
    try:
        scheduler.update(pixels, 1, 'camera', 100, lambda _: ([box(0)], {}),
                         recognize=lambda image, p, key: ([dict(x, label='old') for x in p], {}),
                         wait_for_initial_result=True)
        scheduler.thread.join(3)
        first, _ = scheduler.update(pixels, 1, 'camera', 100, detect)
        identity = first[0]['track_id']

        def recognize(image, p, key):
            entered.set()
            assert release.wait(5)
            return [dict(x, label='new') for x in p], {}

        scheduler.force_detection = True
        scheduler.update(pixels, 2, 'camera', 100, lambda _: ([box(1)], {}), recognize=recognize)
        assert entered.wait(2)
        pending, _ = scheduler.update(pixels, 3, 'camera', 100, detect)
        assert (pending[0]['track_id'], pending[0]['label']) == (identity, 'old')
        assert pending[0]['recognition_pending']
        release.set()
        scheduler.thread.join(3)
        result, _ = scheduler.update(pixels, 4, 'camera', 100, detect)
        assert (result[0]['track_id'], result[0]['label']) == (identity, 'new')
    finally:
        release.set()
        scheduler.close()


def test_deferred_labels_ignore_removed_objects_and_survivor_list_positions(pixels):
    scheduler = AsyncNvDCFProposals(factory=SpatialNvDCF, defer_recognition=True)
    entered, release = threading.Event(), threading.Event()
    try:
        def recognize(image, p, key):
            entered.set()
            assert release.wait(5)
            return [dict(p[0], label='A'), dict(p[1], label='B')], {}

        first, _ = scheduler.update(pixels, 1, 'camera', 100,
                                    lambda _: ([box(0), box(100)], {}), recognize=recognize,
                                    wait_for_initial_result=True)
        assert entered.wait(2)
        b_id = first[1]['track_id']
        scheduler.active.keep([first[1]])
        release.set()
        scheduler.thread.join(3)
        result, _ = scheduler.update(pixels, 2, 'camera', 100, detect)
        assert [(p['track_id'], p['label']) for p in result] == [(b_id, 'B')]
    finally:
        release.set()
        scheduler.close()


def test_deferred_context_reset_discards_stale_classification(pixels):
    scheduler = AsyncNvDCFProposals(factory=SpatialNvDCF, defer_recognition=True)
    entered, release = threading.Event(), threading.Event()
    try:
        def recognize(image, p, key):
            entered.set()
            assert release.wait(5)
            return [dict(x, label='stale') for x in p], {}

        scheduler.update(pixels, 1, 'old', 100, lambda _: ([box(0)], {}),
                         recognize=recognize, wait_for_initial_result=True)
        assert entered.wait(2)
        assert scheduler.update(pixels, 2, 'new', 100, detect) is None
        release.set()
        scheduler.thread.join(3)
        assert scheduler.recognized is None
        result, _ = scheduler.update(pixels, 3, 'new', 100,
                                     lambda _: ([box(0, 'fresh')], {}), wait_for_initial_result=True)
        assert result[0]['label'] == 'fresh'
    finally:
        release.set()
        scheduler.close()


def test_deferred_dino_error_keeps_geometry_and_allows_retry(pixels):
    scheduler = AsyncNvDCFProposals(factory=SpatialNvDCF, defer_recognition=True)
    try:
        def broken(*args):
            raise RuntimeError('classification failed')
        scheduler.update(pixels, 1, 'camera', 100, lambda _: ([box(0)], {}),
                         recognize=broken, wait_for_initial_result=True)
        scheduler.thread.join(3)
        result, source = scheduler.update(pixels, 2, 'camera', 100, detect)
        assert result and source == 'nvdcf'
        assert scheduler.scheduler_metrics()['recognition_error'] == 'classification failed'
        assert scheduler.error is None
    finally:
        scheduler.close()
