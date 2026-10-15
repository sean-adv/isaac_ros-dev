"""Robust foreground bounds, calibration and the standard ROS detection contract."""

import cv2
import numpy as np
import pytest
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header

from dinov3_cases.depth_bbox import estimate_bbox3d, attach_bboxes3d, bboxes3d_message


def calibration():
    return CameraInfo(width=160, height=120,
                      k=[200., 0., 80., 0., 200., 60., 0., 0., 1.],
                      distortion_model="plumb_bob", d=[0.] * 5)


def scene():
    depth = np.full((120, 160), 2.)
    depth[40:80, 35:125] = 1.
    return depth


BOX = [25, 30, 135, 90]


def test_flat_surface_uses_component_quantiles_without_inventing_thickness():
    data = estimate_bbox3d(scene(), calibration(), BOX)
    np.testing.assert_allclose(data["position"], [-.0025, -.0025, 1.])
    # 90 x 40 samples: 2% cuts one X column and no Y row.
    np.testing.assert_allclose(data["size"], [.435, .195, 0.])
    assert data["point_count"] == 3600
    assert data["seed_depth"] == 1.
    assert data["method"] == "central_depth_component_mean_z"


def test_quantile_zero_gives_full_visible_bounds():
    data = estimate_bbox3d(scene(), calibration(), BOX, quantile=0.)
    np.testing.assert_allclose(data["size"], [.445, .195, 0.])


def test_near_outlier_does_not_choose_foreground_or_change_box():
    depth = scene()
    depth[60, 80] = .3
    data = estimate_bbox3d(depth, calibration(), BOX)
    np.testing.assert_allclose(data["position"], [-.0025, -.0025, 1.])
    assert data["seed_depth"] == data["nearest_depth"] == 1.
    assert data["point_count"] == 3599


def test_in_band_outlier_is_diagnostic_but_does_not_inflate_thickness():
    depth = scene()
    depth[60, 80] = .95
    data = estimate_bbox3d(depth, calibration(), BOX)
    assert data["nearest_depth"] == .95
    assert data["nearest_pixel"] == [80, 60]
    np.testing.assert_allclose(data["depth_range"], [1., 1.])



def test_z_uses_all_component_depths_mean_instead_of_quantile_midpoint():
    depth = scene()
    depth[40:44, 35:125] = .95
    expected_z = float(np.mean(depth[40:80, 35:125]))
    for quantile in (.02, .2):
        data = estimate_bbox3d(depth, calibration(), BOX, quantile=quantile)
        assert data["position"][2] == pytest.approx(expected_z)
        assert not np.isclose(data["position"][2], np.mean(data["depth_range"]))
        assert data["point_count"] == 3600


def test_disconnected_same_depth_region_is_excluded():
    depth = scene()
    depth[31:36, 26:134] = 1.
    data = estimate_bbox3d(depth, calibration(), BOX, quantile=0.)
    assert data["point_count"] == 3600
    np.testing.assert_allclose(data["size"], [.445, .195, 0.])


def test_round_object_bounds_include_extrema_outside_diagonal_rays():
    v, u = np.indices((120, 160))
    depth = np.where((u-80)**2 + (v-60)**2 <= 30**2, 1., 2.)
    data = estimate_bbox3d(depth, calibration(), [40, 20, 121, 101])
    # A four-diagonal-ray box is only about .21 m wide; robust cloud bounds
    # recover most of the .30 m diameter despite trimming the extreme tips.
    assert .26 <= data["size"][0] <= .30
    assert .26 <= data["size"][1] <= .30
    np.testing.assert_allclose(data["position"], [0., 0., 1.])


def test_depth_holes_on_diagonals_do_not_reject_valid_component():
    depth = scene()
    for offset in range(15):
        depth[40+offset, 35+offset] = np.nan
    depth[60, 80] = np.inf
    with np.errstate(invalid="raise"):
        data = estimate_bbox3d(depth, calibration(), BOX)
    assert data["point_count"] == 3584
    assert data["size"][0] > .4


def test_sloped_surface_can_span_both_sides_of_seed_depth():
    depth = scene()
    depth[40:80, 35:125] = np.linspace(.93, 1.07, 90)[None, :]
    data = estimate_bbox3d(depth, calibration(), BOX)
    assert data["size"][2] > .13  # No longer capped at 8 cm from nearest Z.
    assert data["seed_depth"] == pytest.approx(1., abs=.002)


@pytest.mark.parametrize("value", [0., np.nan, np.inf, .01, 6.])
def test_missing_depth_does_not_invent_distance(value):
    items = [dict(label="medicine", score=.9, bbox=BOX, bbox3d={"stale": True})]
    attach_bboxes3d(items, np.full((120, 160), value), calibration())
    assert items[0]["bbox3d_status"] == "no_valid_bbox_depth"
    assert "bbox3d" not in items[0]
    assert not bboxes3d_message(items, Header()).detections


def test_missing_center_and_small_components_fail_explicitly():
    depth = scene()
    depth[50:71, 65:96] = 0.
    with pytest.raises(ValueError, match="no_valid_center_depth"):
        estimate_bbox3d(depth, calibration(), BOX)
    depth[:] = 0.
    depth[58:63, 78:83] = 1.
    with pytest.raises(ValueError, match="insufficient_depth_points"):
        estimate_bbox3d(depth, calibration(), BOX)


def test_clips_bbox_and_rejects_empty_or_invalid_intrinsics():
    depth = np.ones((120, 160))
    data = estimate_bbox3d(depth, calibration(), [-20, -10, 180, 130], quantile=0.)
    np.testing.assert_allclose(data["size"], [.795, .595, 0.])
    with pytest.raises(ValueError, match="empty_bbox"):
        estimate_bbox3d(depth, calibration(), [200, 10, 220, 30])
    info = calibration()
    info.k[0] = 0.
    with pytest.raises(ValueError, match="invalid_camera_intrinsics"):
        estimate_bbox3d(depth, info, BOX)


@pytest.mark.parametrize("model, coefficients", [
    ("plumb_bob", [.1, -.03, .001, .002, 0.]),
    ("equidistant", [.01, 0., 0., 0.]),
])
def test_diagnostic_samples_reproject_to_actual_component_pixels(model, coefficients):
    info = calibration()
    info.distortion_model, info.d = model, coefficients
    data = estimate_bbox3d(scene(), info, BOX)
    points = np.asarray(data["corners"]).reshape(-1, 1, 3)
    k, d, zero = np.asarray(info.k).reshape(3, 3), np.asarray(info.d), np.zeros(3)
    project = cv2.fisheye.projectPoints if model == "equidistant" else cv2.projectPoints
    pixels, _ = project(points, zero, zero, k, d)
    np.testing.assert_allclose(pixels.reshape(-1, 2), data["corner_sample_pixels"], atol=1e-5)


def test_standard_ros_serialization_preserves_pose_class_and_bounds():
    from rclpy.serialization import serialize_message, deserialize_message
    from vision_msgs.msg import Detection3DArray

    depth = scene()
    depth[60, 80] = .95
    items = [dict(label=label, score=.9, track_id=i, bbox=BOX)
             for i, label in enumerate(["medicine", "unknown"], 1)]
    attach_bboxes3d(items, depth, calibration())
    header = Header(frame_id="camera_color_optical_frame")
    message = bboxes3d_message(items, header)
    assert isinstance(message, Detection3DArray)
    received = deserialize_message(serialize_message(message), Detection3DArray)
    assert received.header == header
    assert received == message
    assert len(received.detections) == 1
    detection = received.detections[0]
    assert detection.header == header
    assert detection.id == "1"
    assert detection.bbox.center.position.z == pytest.approx((3599. + .95) / 3600)
    assert detection.bbox.center.orientation.w == 1.
    assert detection.bbox.size.x == pytest.approx(.435)
    assert detection.bbox.size.z == 0.
    assert len(detection.results) == 1
    result = detection.results[0]
    assert result.hypothesis.class_id == "medicine"
    assert result.hypothesis.score == pytest.approx(.9)
    assert result.pose.pose == detection.bbox.center
    assert list(result.pose.covariance) == [0.] * 36
    data = items[0]["bbox3d"]
    np.testing.assert_allclose(
        [result.pose.pose.position.x, result.pose.pose.position.y, result.pose.pose.position.z],
        data["position"])
    np.testing.assert_allclose(
        [detection.bbox.size.x, detection.bbox.size.y, detection.bbox.size.z], data["size"])
    assert "pose" not in items[0] and "pose_status" not in items[0]
    assert not bboxes3d_message([items[1]], header).detections
    assert "bbox3d" in items[1]
    assert not bboxes3d_message([], header).detections


@pytest.mark.parametrize("option, value, error", [
    *[("depth_band", v, "invalid_bbox_depth_band") for v in (0., -1., np.nan, np.inf)],
    *[("quantile", v, "invalid_bbox_quantile") for v in (-.01, .5, np.nan, np.inf)],
    *[("min_points", v, "invalid_bbox_min_points") for v in (0, 4, 5.5)],
])
def test_invalid_segmentation_options(option, value, error):
    with pytest.raises(ValueError, match=error):
        estimate_bbox3d(scene(), calibration(), BOX, **{option: value})
