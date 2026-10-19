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
    assert data["method"] == "central_depth_component_plane_rectangle"
    assert data["orientation_status"] == "ok"
    np.testing.assert_allclose(data["rotation"], np.eye(3))


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



def test_center_uses_local_quantile_midpoint_consistently_with_size():
    depth = scene()
    depth[40:44, 35:125] = .95
    for quantile in (.02, .2):
        data = estimate_bbox3d(depth, calibration(), BOX, quantile=quantile)
        assert data["position"][2] == pytest.approx(np.mean(data["depth_range"]))
        assert data["size"][2] == pytest.approx(np.ptp(data["depth_range"]))
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
    np.testing.assert_allclose(data["position"], [0., 0., 1.], atol=1e-12)


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
    assert np.ptp(data["depth_range"]) > .13
    assert data["size"][2] < .01  # Camera-Z slope is no longer box thickness.
    assert abs(data["rotation"][0][2]) > .2
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
    assert len(received.detections) == 2
    detection = received.detections[0]
    assert detection.header == header
    assert detection.id == "1"
    assert detection.bbox.center.position.z > 0.
    assert detection.bbox.center.orientation.w == 1.
    assert detection.bbox.size.x > .435
    assert detection.bbox.size.z > 0.
    assert items[0]["bbox3d"]["fit_status"] == "matched"
    assert items[0]["bbox3d"]["measured_size"] == pytest.approx([.435, .195, 0.])
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
    unknown = received.detections[1]
    assert unknown.id == "2"
    assert unknown.results[0].hypothesis.class_id == "unknown"
    assert len(bboxes3d_message([items[1]], header).detections) == 1
    for item in items:
        item["bbox_visible"] = False
    assert not bboxes3d_message(items, header).detections
    assert "bbox3d" in items[1]
    assert not bboxes3d_message([], header).detections


@pytest.mark.parametrize("option, value, error", [
    *[("plane_threshold", v, "invalid_plane_threshold") for v in (0., -1., np.nan, np.inf)],
    *[("depth_band", v, "invalid_bbox_depth_band") for v in (0., -1., np.nan, np.inf)],
    *[("quantile", v, "invalid_bbox_quantile") for v in (-.01, .5, np.nan, np.inf)],
    *[("min_points", v, "invalid_bbox_min_points") for v in (0, 4, 5.5)],
])
def test_invalid_segmentation_options(option, value, error):
    with pytest.raises(ValueError, match=error):
        estimate_bbox3d(scene(), calibration(), BOX, **{option: value})


@pytest.mark.parametrize("model, coefficients", [
    ("plumb_bob", [.1, -.03, .001, .002, 0.]),
    ("rational_polynomial", [.1, -.03, .001, .002, 0., .001, 0., 0.]),
    ("equidistant", [.01, .001, 0., 0.]),
])
@pytest.mark.parametrize("dtype, scale", [(np.uint16, .001), (np.float32, 1.)])
def test_roi_conversion_and_cached_rays_preserve_all_output_fields(model, coefficients, dtype, scale):
    from dinov3_cases.depth_bbox import ProjectionRayCache
    info = calibration()
    info.distortion_model, info.d = model, coefficients
    raw = (scene() / scale).astype(dtype)
    raw[60, 80] = 0
    original = raw.copy()
    baseline = estimate_bbox3d(raw.astype(np.float64) * scale, info, BOX)
    cache = ProjectionRayCache()
    for box in (BOX, [-10, -5, 170, 125], [30.4, 35.8, 130.2, 85.1]):
        expected = estimate_bbox3d(raw.astype(np.float64) * scale, info, box)
        actual = estimate_bbox3d(raw, info, box, depth_scale=scale, ray_cache=cache)
        assert actual.keys() == expected.keys()
        for key in expected:
            if isinstance(expected[key], str):
                assert actual[key] == expected[key]
            else:
                np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-12)
    assert baseline["point_count"] == 3599
    np.testing.assert_array_equal(raw, original)


def test_ray_cache_reuses_map_and_rebuilds_for_mutated_calibration():
    from dinov3_cases.depth_bbox import ProjectionRayCache
    cache = ProjectionRayCache()
    info = calibration()
    def run():
        cached = estimate_bbox3d(scene(), info, BOX, ray_cache=cache)
        assert cached == estimate_bbox3d(scene(), info, BOX)
        return cache._rays
    previous = run()
    assert run() is previous
    for change in (lambda: info.k.__setitem__(0, 210.),
                   lambda: info.d.__setitem__(0, .05),
                   lambda: setattr(info, "distortion_model", "rational_polynomial")):
        change()
        current = run()
        assert current is not previous
        previous = current
    resized = calibration()
    resized.width, resized.height = 80, 60
    data = estimate_bbox3d(np.ones((60, 80)), resized, [10, 10, 60, 50], ray_cache=cache)
    assert cache._rays.shape == (60, 80, 2)
    assert cache._rays is not previous
    assert data == estimate_bbox3d(np.ones((60, 80)), resized, [10, 10, 60, 50])
    resized.distortion_model = "unsupported"
    with pytest.raises(ValueError, match="unsupported_distortion_model"):
        estimate_bbox3d(np.ones((60, 80)), resized, [10, 10, 60, 50], ray_cache=cache)


def test_roi_conversion_preserves_holes_and_depth_band_boundaries():
    from dinov3_cases.depth_bbox import ProjectionRayCache
    raw = scene().astype(np.float32)
    raw[45, 40:44] = [np.nan, np.inf, 0., .92]
    expected = estimate_bbox3d(raw.astype(np.float64), calibration(), BOX)
    actual = estimate_bbox3d(raw, calibration(), BOX, ray_cache=ProjectionRayCache())
    assert actual == expected


def tilted_rectangle(roll, tilt):
    """Ray/plane intersection of a known .4 x .16 m rectangle at Z=1 m."""
    rz, _ = cv2.Rodrigues(np.array([0., 0., roll]))
    ry, _ = cv2.Rodrigues(np.array([0., tilt, 0.]))
    rotation = rz @ ry
    v, u = np.indices((120, 160))
    rays = np.stack(((u - 80) / 200., (v - 60) / 200., np.ones_like(u)), axis=-1)
    center = np.array([0., 0., 1.])
    normal = rotation[:, 2]
    z = np.dot(center, normal) / (rays @ normal)
    local = (rays * z[..., None] - center) @ rotation
    mask = (np.abs(local[..., 0]) <= .2) & (np.abs(local[..., 1]) <= .08)
    return np.where(mask, z, 2.), rotation, mask


@pytest.mark.parametrize("roll, tilt", [(0., 0.), (.55, .25), (-.7, -.3), (1.4, .2)])
def test_recovers_long_edge_normal_and_bounds_of_tilted_rectangle(roll, tilt):
    from dinov3_cases.depth_bbox import bbox_vertices, _rotation_quaternion
    depth, expected, mask = tilted_rectangle(roll, tilt)
    # Missing samples and submillimeter noise, plus in-band depth outliers.
    rng = np.random.default_rng(7)
    depth[mask] += rng.normal(0., .0004, mask.sum())
    depth[mask & (rng.random(depth.shape) < .01)] = 0.
    depth[mask & (rng.random(depth.shape) < .01)] -= .035
    data = estimate_bbox3d(depth, calibration(), [15, 15, 145, 105], quantile=.02)
    rotation = np.array(data["rotation"])
    assert data["orientation_status"] == "ok"
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.)
    assert np.dot(rotation[:, 0], expected[:, 0]) > .999
    assert np.dot(rotation[:, 2], expected[:, 2]) > .999
    # A 2% trim at each end retains about 96% of each edge.
    np.testing.assert_allclose(data["size"][:2], np.array([.4, .16]) * .96, atol=.008)
    assert data["size"][2] < .004
    np.testing.assert_allclose(data["position"], [0., 0., 1.], atol=.003)
    local_vertices = (bbox_vertices(data) - data["position"]) @ expected
    np.testing.assert_allclose(np.ptp(local_vertices, axis=0)[:2],
                               np.array([.4, .16]) * .96, atol=.01)
    q = np.array(_rotation_quaternion(rotation))
    assert np.linalg.norm(q) == pytest.approx(1.)
    angle = 2 * np.arctan2(np.linalg.norm(q[:3]), q[3])
    recovered, _ = cv2.Rodrigues(q[:3] / np.linalg.norm(q[:3]) * angle)
    np.testing.assert_allclose(recovered, rotation, atol=1e-10)


def test_nonplanar_component_falls_back_explicitly_without_losing_box():
    depth = scene()
    depth[40:80, 35:125] = np.random.default_rng(3).uniform(.9, 1.1, (40, 90))
    data = estimate_bbox3d(depth, calibration(), BOX, depth_band=.2)
    assert data["orientation_status"] == "camera_aligned_fallback:no_dominant_plane"
    assert data["method"] == "central_depth_component_mean_z"
    np.testing.assert_allclose(data["rotation"], np.eye(3))
    assert data["position"][2] == pytest.approx(depth[40:80, 35:125].mean())


def test_square_surface_marks_in_plane_ambiguity():
    depth = np.full((120, 160), 2.)
    depth[40:80, 60:100] = 1.
    data = estimate_bbox3d(depth, calibration(), BOX)
    assert data["orientation_status"] == "ambiguous_in_plane"


def test_ros_quaternion_and_wireframe_share_nonidentity_rotation():
    from dinov3_cases.depth_bbox import bbox_vertices
    from rclpy.serialization import serialize_message, deserialize_message
    from vision_msgs.msg import Detection3DArray
    depth, expected, _ = tilted_rectangle(.6, .25)
    items = [dict(label="medicine", score=.9, bbox=[15, 15, 145, 105])]
    attach_bboxes3d(items, depth, calibration())
    message = bboxes3d_message(items, Header(frame_id="camera"))
    message = deserialize_message(serialize_message(message), Detection3DArray)
    detection = message.detections[0]
    q = detection.bbox.center.orientation
    xyz = np.array([q.x, q.y, q.z])
    rotation, _ = cv2.Rodrigues(xyz / np.linalg.norm(xyz) *
                              (2 * np.arctan2(np.linalg.norm(xyz), q.w)))
    data = items[0]["bbox3d"]
    np.testing.assert_allclose(rotation, expected, atol=.015)
    vertices = (bbox_vertices(data) - data["position"]) @ rotation
    np.testing.assert_allclose(np.ptp(vertices, axis=0), data["size"], atol=1e-10)
    assert detection.results[0].pose.pose == detection.bbox.center


@pytest.mark.parametrize("axis", np.eye(3))
def test_quaternion_handles_half_turn(axis):
    from dinov3_cases.depth_bbox import _rotation_quaternion
    rotation, _ = cv2.Rodrigues(axis * np.pi)
    q = np.array(_rotation_quaternion(rotation))
    assert abs(np.dot(q[:3], axis)) == pytest.approx(1.)
    assert abs(q[3]) < 1e-12


def test_open3d_failure_keeps_explicit_camera_aligned_fallback(monkeypatch):
    from types import SimpleNamespace
    from dinov3_cases import depth_bbox

    def fail(**options):
        raise RuntimeError('native segmentation failed')

    monkeypatch.setattr(depth_bbox.o3d.geometry, 'PointCloud',
                        lambda points: SimpleNamespace(segment_plane=fail))
    data = estimate_bbox3d(scene(), calibration(), BOX)
    assert data['orientation_status'] == 'camera_aligned_fallback:plane_segmentation_failed'
    np.testing.assert_allclose(data['rotation'], np.eye(3))
    np.testing.assert_allclose(data['size'], [.435, .195, 0.])


def test_open3d_degenerate_plane_does_not_produce_invalid_orientation():
    from dinov3_cases.depth_bbox import _surface_rotation
    points = np.column_stack((np.linspace(-.2, .2, 100), np.zeros(100), np.ones(100)))
    with pytest.raises(ValueError, match='degenerate_plane'):
        _surface_rotation(points, .005, 80)
