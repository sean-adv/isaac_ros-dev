"""BBox smoothing, geometric history matching and frame-local results."""

import pytest

from dinov3_cases.tracking import BBoxSmoother, TrackIdentities, TrackLifecycle


def test_bbox_smoothing_averages_last_three_raw_frames():
    smoother = BBoxSmoother()
    # 0, 6, 12 average to 6; fourth frame evicts 0, averaging 6, 12, 24 to 14.
    expected = [0, 3, 6, 14]
    for seq, (x, mean) in enumerate(zip((0, 6, 12, 24), expected)):
        raw = [box(x)]
        result = smoother.update(raw, seq, "camera")
        assert result[0]["bbox"] == [mean, 0, mean + 100, 100]
        assert result[0]["area"] == 10000
        assert raw[0]["bbox"][0] == x
        assert "track_id" not in result[0]


def test_bbox_smoothing_matches_objects_when_sam_reorders_detections():
    smoother = BBoxSmoother()
    smoother.update([box(0), box(300)], 1, "camera")
    result = smoother.update([box(310, "other"), box(10, "unknown")], 2, "camera")
    assert [item["bbox"][0] for item in result] == [305, 5]
    assert [item["label"] for item in result] == ["other", "unknown"]


def test_bbox_smoothing_repeated_frame_does_not_bias_average_or_expose_cache():
    smoother = BBoxSmoother()
    smoother.update([box(0)], 1, "camera")
    result = smoother.update([box(6)], 2, "camera")
    result[0]["bbox"][0] = 999
    assert smoother.update([box(6)], 2, "camera")[0]["bbox"][0] == 3
    assert smoother.update([box(12)], 3, "camera")[0]["bbox"][0] == 6


def test_bbox_smoothing_clears_missing_and_context_changed_objects():
    smoother = BBoxSmoother()
    smoother.update([box(0)], 1, "camera")
    assert smoother.update([], 2, "camera") == []
    assert smoother.update([box(20)], 3, "camera")[0]["bbox"][0] == 20
    assert smoother.update([box(30)], 3, "upload")[0]["bbox"][0] == 30
    assert smoother.update([box(1000)], 4, "upload")[0]["bbox"][0] == 1000


def test_bbox_smoothing_one_frame_disables_averaging():
    smoother = BBoxSmoother(1)
    smoother.update([box(0)], 1, "camera")
    assert smoother.update([box(10)], 2, "camera")[0]["bbox"] == box(10)["bbox"]


def test_bbox_smoothing_partial_loss_and_new_box_do_not_mix_histories():
    smoother = BBoxSmoother()
    smoother.update([box(0), box(300)], 1, "camera")
    result = smoother.update([box(310), box(600)], 2, "camera")
    assert [item["bbox"][0] for item in result] == [305, 600]
    result = smoother.update([box(610), box(320)], 3, "camera")
    assert [item["bbox"][0] for item in result] == [605, 310]


def test_bbox_smoothing_does_not_share_one_history_between_two_boxes():
    smoother = BBoxSmoother()
    smoother.update([box(0)], 1, "camera")
    result = smoother.update([box(2), box(20)], 2, "camera")
    assert [item["bbox"][0] for item in result] == [1, 20]


def test_bbox_smoothing_rounds_outwards_and_updates_area():
    smoother = BBoxSmoother()
    smoother.update([dict(bbox=[0, 0, 100, 100])], 1, "camera")
    result = smoother.update([dict(bbox=[1, 1, 102, 102])], 2, "camera")[0]
    assert result["bbox"] == [0, 0, 101, 101]
    assert result["area"] == 10201


@pytest.mark.parametrize("window", [0, -1, True, 1.5, "3", None])
def test_bbox_smoothing_rejects_invalid_window(window):
    with pytest.raises(ValueError, match="bbox_smoothing_frames"):
        BBoxSmoother(window)


def box(x=0, label="medicine"):
    return dict(bbox=[x, 0, x + 100, 100], label=label)


def test_new_track_uses_same_miss_allowance_as_established_tracks():
    lifecycle = TrackLifecycle()
    item = dict(box(), track_id=1)
    first, _ = lifecycle.update([item], {1}, now=0)
    assert first[0]['track_status'] == 'confirmed'
    # KCF callbacks cannot count as supporting detections.
    for _ in range(10):
        tracked, _ = lifecycle.update(first, now=.5)
        assert tracked[0]['sam3_hits'] == 1
    for now in (1, 2):
        coasting, retired = lifecycle.update(first, set(), now=now)
        assert not retired and coasting[0]['sam3_misses'] == now
        assert coasting[0]['track_status'] == 'coasting'
    assert lifecycle.update(first, set(), now=3) == ([], {1})


def test_confirmed_track_tolerates_isolated_miss_then_retires_after_three():
    lifecycle = TrackLifecycle()
    item = dict(box(), track_id=1)
    lifecycle.update([item], {1}, now=0)
    confirmed, _ = lifecycle.update([item], {1}, now=1)
    assert confirmed[0]['track_status'] == 'confirmed'
    coasting, _ = lifecycle.update([item], set(), now=2)
    assert coasting[0]['track_status'] == 'coasting'
    assert coasting[0]['sam3_misses'] == 1
    # Repeated camera frames do not increment misses or renew detector age.
    for _ in range(10):
        tracked, _ = lifecycle.update(coasting, now=2.5)
        assert tracked[0]['sam3_misses'] == 1
        assert tracked[0]['sam3_age_seconds'] == 1.5
    restored, _ = lifecycle.update([item], {1}, now=3)
    assert restored[0]['sam3_misses'] == 0
    assert restored[0]['track_status'] == 'confirmed'
    assert lifecycle.update([item], set(), now=4)[0]
    assert lifecycle.update([item], set(), now=5)[0]
    assert lifecycle.update([item], set(), now=6) == ([], {1})


def test_detector_timeout_retires_track_even_without_new_detector_results():
    lifecycle = TrackLifecycle()
    item = dict(box(), track_id=1)
    lifecycle.update([item], {1}, detected_at=0, now=.4)
    assert lifecycle.update([item], now=4.99)[0]
    assert lifecycle.update([item], now=5) == ([], {1})


def test_stale_detection_result_cannot_start_a_fresh_track_lifetime():
    lifecycle = TrackLifecycle()
    assert lifecycle.update([dict(box(), track_id=1)], {1}, detected_at=0, now=6) == ([], {1})


def test_track_lifetime_is_independent_for_each_object_and_unknown_is_allowed():
    lifecycle = TrackLifecycle()
    items = [dict(box(), track_id=1), dict(box(300, 'unknown'), track_id=2)]
    lifecycle.update(items, {1, 2}, now=0)
    lifecycle.update(items, {1, 2}, now=.5)
    lifecycle.update(items, {2}, now=1)
    lifecycle.update(items, {2}, now=2)
    remaining, retired = lifecycle.update(items, {2}, now=3)
    assert retired == {1}
    assert [item['track_id'] for item in remaining] == [2]
    assert remaining[0]['label'] == 'unknown'
    assert remaining[0]['sam3_misses'] == 0


@pytest.mark.parametrize('options', [
    {'max_missed_detections': True}, {'max_missed_detections': -1},
    {'max_missed_detections': 0}, {'max_missed_detections': 1.5},
    {'max_unconfirmed_seconds': False}, {'max_unconfirmed_seconds': 0},
    {'max_unconfirmed_seconds': float('inf')}, {'max_unconfirmed_seconds': float('nan')},
])
def test_invalid_lifecycle_limits_are_rejected(options):
    with pytest.raises(ValueError, match='kcf_'):
        TrackLifecycle(**options)


def test_lost_identity_expires_without_displaying_stale_boxes(monkeypatch):
    now = [0.0]
    monkeypatch.setattr('dinov3_cases.tracking.time.monotonic', lambda: now[0])
    identities = TrackIdentities()
    first, _ = identities.assign([box()], [])
    now[0] = .5
    assert identities.assign([], [])[0] == []
    recovered, _ = identities.assign([box(10)], [])
    assert recovered[0]['track_id'] == first[0]['track_id']
    now[0] = 2.0
    fresh, _ = identities.assign([box(10)], [])
    assert fresh[0]['track_id'] != first[0]['track_id']


def test_live_identity_has_priority_over_nearby_lost_identity():
    identities = TrackIdentities()
    first, _ = identities.assign([box(0), box(40)], [])
    # A missing neighbor is closer to the detector correction, but the live
    # track must get first refusal rather than having its ID stolen.
    result, _ = identities.assign([box(5)], [first[1]])
    assert result[0]['track_id'] == first[1]['track_id']


def test_smoothing_uses_object_id_and_does_not_inherit_neighbor_history():
    smoother = BBoxSmoother()
    smoother.update([dict(box(0), track_id=1), dict(box(300), track_id=2)], 1, 'camera')
    items = smoother.update([dict(box(310), track_id=2), dict(box(10), track_id=3)],
                            2, 'camera')
    assert [item['bbox'][0] for item in items] == [305, 10]


def test_same_frame_identity_change_replaces_smoothing_history():
    smoother = BBoxSmoother()
    smoother.update([dict(box(0), track_id=1)], 1, 'camera')
    smoother.update([dict(box(10), track_id=1)], 2, 'camera')
    items = smoother.update([dict(box(10), track_id=2)], 2, 'camera')
    assert items[0]['bbox'][0] == 10


def test_ros_2d_and_3d_use_same_tracking_id():
    from std_msgs.msg import Header
    from dinov3_cases.grounded import recognition_message
    from dinov3_cases.depth_bbox import bboxes3d_message

    item = box()
    item.update(track_id=42, score=.9, bbox3d=dict(
        position=[0., 0., 1.], size=[.1, .1, .1],
        corner_sample_pixels=[[0, 0], [99, 0], [99, 99], [0, 99]],
        corners=[[-.05, -.05, 1.], [.05, -.05, 1.],
                 [.05, .05, 1.], [-.05, .05, 1.]],
        nearest_pixel=[0, 0], nearest_point=[-.05, -.05, 1.]))
    assert recognition_message([item], Header()).detections[0].id == "42"
    assert bboxes3d_message([item], Header()).detections[0].id == "42"


def test_same_frame_handoff_preserves_added_removed_and_reordered_boxes():
    smoother = BBoxSmoother()
    smoother.update([box(0), box(300)], 1, 'camera')
    # The previous version reused two cached boxes and silently lost the third.
    result = smoother.update([box(300), box(0), box(600)], 1, 'camera')
    assert [item['bbox'] for item in result] == [box(x)['bbox'] for x in (300, 0, 600)]
    result = smoother.update([box(600)], 1, 'camera')
    assert [item['bbox'] for item in result] == [box(600)['bbox']]
    assert smoother.update([], 1, 'camera') == []
    assert smoother.update([box(900)], 1, 'camera')[0]['bbox'] == box(900)['bbox']


def test_same_frame_correction_replaces_sample_without_biasing_average():
    smoother = BBoxSmoother()
    smoother.update([box(0)], 1, 'camera')
    smoother.update([box(6)], 2, 'camera')
    assert smoother.update([box(12)], 2, 'camera')[0]['bbox'][0] == 6
    assert smoother.update([box(12)], 2, 'camera')[0]['bbox'][0] == 6
    assert smoother.update([box(24)], 3, 'camera')[0]['bbox'][0] == 12
