"""Appearance changes retire successful KCF tracks before SAM3 miss expiry."""

import numpy as np
import pytest

from dinov3_cases.async_kcf import AsyncKCFProposals
from dinov3_cases.kcf import KCFProposals
from dinov3_cases.kcf_lab import validate_lab_settings


class StationaryTracker:
    def init(self, image, box):
        self.box = box

    def update(self, image):
        return True, self.box


def scene(changed=False):
    image = np.full((100, 160, 3), 128, dtype=np.uint8)
    image[20:60, 20:60] = (0, 0, 255) if changed else (255, 0, 0)
    image[20:60, 90:130] = (0, 255, 0)
    return image


def proposals():
    return [dict(bbox=b, detection_score=.9) for b in
            ([20, 20, 60, 60], [90, 20, 130, 60])]


def pipeline(**kwargs):
    return KCFProposals(StationaryTracker, roi_padding=.3, max_roi_side=32,
                        lab_similarity_threshold=.35, lab_change_confirmations=3, **kwargs)


def step(p, seq, changed=False):
    return p.update(scene(changed), seq, 'camera', 1000, proposals,
                    retain_valid_tracks=True)


def test_persistent_change_drops_only_changed_track_despite_kcf_success():
    p = pipeline()
    step(p, 0)
    reference = p.trackers[0].reference.copy()
    items, _ = step(p, 1, True)
    assert len(items) == 2
    assert items[0]['kcf_lab_similarity'] < .35
    assert items[0]['kcf_lab_changed_frames'] == 1
    assert items[1]['kcf_lab_similarity'] == pytest.approx(1)
    # Re-reading the same frame must not count as another confirmation.
    step(p, 1, True)
    assert p.trackers[0].lab_changed_frames == 1
    step(p, 2, True)
    np.testing.assert_array_equal(p.trackers[0].reference, reference)
    items, source = step(p, 3, True)
    assert source == 'kcf'
    assert [item['bbox'] for item in items] == [[90, 20, 130, 60]]
    assert p.refresh_requested
    assert len(p.trackers) == len(p.rois) == 1


def test_transient_change_resets_on_recovery_and_new_sam_seed_resets_reference():
    p = pipeline()
    step(p, 0)
    step(p, 1, True)
    items, _ = step(p, 2)
    assert items[0]['kcf_lab_changed_frames'] == 0
    assert items[0]['kcf_lab_similarity'] == pytest.approx(1)
    step(p, 3, True)
    assert p.trackers[0].lab_changed_frames == 1
    # Changed context represents a new detection; accept the new appearance.
    p.update(scene(True), 4, 'new-camera', 1000, proposals, retain_valid_tracks=True)
    items, _ = p.update(scene(True), 5, 'new-camera', 1000, proposals,
                        retain_valid_tracks=True)
    assert items[0]['kcf_lab_similarity'] == pytest.approx(1)
    assert items[0]['kcf_lab_changed_frames'] == 0


def test_disabled_check_retains_changed_region():
    p = KCFProposals(StationaryTracker, lab_similarity_threshold=0)
    for seq in range(6):
        items, _ = step(p, seq, changed=seq > 0)
    assert len(items) == 2
    assert 'kcf_lab_similarity' not in items[0]


def test_lab_survives_async_handoff_and_retires_before_miss_limit():
    p = AsyncKCFProposals(StationaryTracker, max_missed_detections=10,
                          lab_similarity_threshold=.35, lab_change_confirmations=3)
    detector = lambda image: (proposals(), {})
    missing = lambda image: ([], {})
    def update(seq, changed=False, detect=detector):
        return p.update(scene(changed), seq, 'camera', 1000, detect)
    try:
        update(0)
        p.thread.join(3)
        assert not p.thread.is_alive()
        assert len(update(0)[0]) == 2
        # A committed SAM3 miss retains live trackers and their appearance state.
        p.force_detection = True
        update(1, True, missing)
        p.thread.join(3)
        assert not p.thread.is_alive()
        items, _ = update(1, True, missing)
        assert len(items) == 2
        assert items[0]['sam3_misses'] == 1
        assert p.active.trackers[0].lab_changed_frames == 1
        update(2, True, missing)
        items, _ = update(3, True, missing)
        assert [item['bbox'] for item in items] == [[90, 20, 130, 60]]
        assert items[0]['sam3_misses'] < 10
        assert p.busy or p.ready is not None
    finally:
        p.close()


@pytest.mark.parametrize('threshold', [-1, 1.1, float('nan'), float('inf'), True, '0.3'])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError, match='similarity_threshold'):
        validate_lab_settings(threshold, 3)


@pytest.mark.parametrize('confirmations', [0, -1, True, 1.5])
def test_invalid_confirmations(confirmations):
    with pytest.raises(ValueError, match='change_confirmations'):
        validate_lab_settings(.35, confirmations)
