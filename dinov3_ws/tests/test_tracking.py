"""BBox smoothing, geometric history matching and frame-local results."""

import pytest

from dinov3_cases.tracking import BBoxSmoother


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
