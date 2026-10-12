"""Identity persistence, ROI matching and exact missed-frame boundaries."""

import pytest

from dinov3_cases.tracking import ROITracker


def box(x=0, label="medicine"):
    return dict(bbox=[x, 0, x + 100, 100], label=label)


def test_ids_follow_boxes_when_order_and_labels_change():
    tracker = ROITracker()
    first = tracker.update([box(0), box(300)])
    second = tracker.update([box(310, "unknown"), box(10, "other")])
    assert [d["track_id"] for d in second] == [first[1]["track_id"], first[0]["track_id"]]


def test_expanded_roi_handles_motion_without_original_overlap():
    tracker = ROITracker()
    first = tracker.update([box()])[0]["track_id"]
    assert tracker.update([box(105)])[0]["track_id"] == first
    assert tracker.update([box(1000)])[0]["track_id"] != first


def test_five_missing_frames_keep_id_and_six_expire_it():
    tracker = ROITracker()
    original = tracker.update([box()])[0]["track_id"]
    for _ in range(5):
        assert tracker.update([]) == []
    assert tracker.update([box(10)])[0]["track_id"] == original
    for _ in range(6):
        tracker.update([])
    assert tracker.update([box(10)])[0]["track_id"] != original


def test_one_track_cannot_match_two_detections():
    tracker = ROITracker()
    original = tracker.update([box()])[0]["track_id"]
    items = tracker.update([box(1), box(10)])
    ids = [d["track_id"] for d in items]
    assert len(set(ids)) == 2
    assert original in ids


def test_size_changes_keep_identity():
    tracker = ROITracker()
    original = tracker.update([box()])[0]["track_id"]
    assert tracker.update([dict(bbox=[10, 10, 90, 90])])[0]["track_id"] == original


def test_reprocessing_same_frame_does_not_age_missing_tracks():
    tracker = ROITracker()
    original = tracker.update([box()], frame_key=1)[0]["track_id"]
    for key in range(2, 7):
        for _ in range(3):
            tracker.update([], frame_key=key)
    assert tracker.update([box()], frame_key=7)[0]["track_id"] == original


def test_reset_forgets_boxes_without_reusing_ids():
    tracker = ROITracker()
    first = tracker.update([box()])[0]["track_id"]
    tracker.reset()
    assert tracker.update([box()])[0]["track_id"] > first


def test_bbox_requires_more_than_ten_detection_frames():
    tracker = ROITracker()
    for frame in range(1, 12):
        item = tracker.update([box()], frame_key=frame)[0]
        assert item["track_detected_frames"] == frame
        assert item["track_total_frames"] == frame
        assert item["track_hit_ratio"] == 1.0
        assert item["bbox_visible"] is (frame > 10)


def test_ratio_includes_misses_and_exact_eighty_percent_is_hidden():
    tracker = ROITracker(max_missed=10)
    for frame in range(1, 12):
        item = tracker.update([box()], frame_key=frame)[0]
    original = item["track_id"]
    assert item["bbox_visible"]
    for frame in range(12, 15):
        assert tracker.update([], frame_key=frame) == []
    item = tracker.update([box()], frame_key=15)[0]
    assert item["track_id"] == original
    assert item["track_detected_frames"] == 12
    assert item["track_total_frames"] == 15
    assert item["track_hit_ratio"] == .8
    assert not item["bbox_visible"]
    item = tracker.update([box()], frame_key=16)[0]
    assert item["track_hit_ratio"] == 13 / 16
    assert item["bbox_visible"]


def test_reprocessing_does_not_inflate_hits_or_total_frames():
    tracker = ROITracker()
    for _ in range(20):
        item = tracker.update([box()], frame_key=1)[0]
        assert item["track_detected_frames"] == item["track_total_frames"] == 1
        assert not item["bbox_visible"]
    tracker.update([], frame_key=2)
    # Detection on a rerun of a previously missed frame counts at most once.
    for _ in range(20):
        item = tracker.update([box()], frame_key=2)[0]
        assert item["track_detected_frames"] == item["track_total_frames"] == 2


def test_each_track_counts_from_its_own_first_frame_and_reset_clears_counts():
    tracker = ROITracker()
    for frame in range(1, 20):
        tracker.update([box()], frame_key=frame)
    items = tracker.update([box(), box(1000)], frame_key=20)
    assert items[0]["bbox_visible"]
    assert items[1]["track_detected_frames"] == items[1]["track_total_frames"] == 1
    assert not items[1]["bbox_visible"]
    original = items[0]["track_id"]
    tracker.reset()
    item = tracker.update([box()], frame_key=21)[0]
    assert item["track_id"] > original
    assert item["track_detected_frames"] == item["track_total_frames"] == 1
    assert not item["bbox_visible"]


def test_custom_bbox_thresholds():
    tracker = ROITracker(bbox_min_hit_ratio=.5, bbox_min_detections=0)
    assert tracker.update([box()], frame_key=1)[0]["bbox_visible"]
    tracker.update([], frame_key=2)
    tracker.update([], frame_key=3)
    item = tracker.update([box()], frame_key=4)[0]
    assert item["track_hit_ratio"] == .5
    assert not item["bbox_visible"]


@pytest.mark.parametrize("options", [
    dict(max_missed=-1), dict(max_missed=1.5), dict(iou_threshold=0),
    dict(iou_threshold=float("nan")), dict(roi_padding=-1),
    dict(bbox_min_hit_ratio=-.1), dict(bbox_min_hit_ratio=1.1),
    dict(bbox_min_hit_ratio=float("nan")), dict(bbox_min_hit_ratio=True),
    dict(bbox_min_detections=-1), dict(bbox_min_detections=10.5),
    dict(bbox_min_detections=True),
])
def test_invalid_settings(options):
    with pytest.raises(ValueError):
        ROITracker(**options)


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
