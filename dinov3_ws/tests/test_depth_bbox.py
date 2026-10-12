"""Measured corner depths, ROI nearest point and ROS bbox contract."""

import numpy as np
import pytest
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header

from dinov3_cases.depth_bbox import estimate_bbox3d, attach_bboxes3d, bboxes3d_message


def calibration():
    return CameraInfo(width=160, height=120,
                      k=[200., 0., 80., 0., 200., 60., 0., 0., 1.],
                      distortion_model="plumb_bob", d=[0.] * 5)


def test_flat_depth_has_no_invented_thickness():
    data = estimate_bbox3d(np.ones((120, 160)), calibration(), [35, 40, 125, 80])
    np.testing.assert_allclose(data["position"], [-.0025, -.0025, 1.])
    np.testing.assert_allclose(data["size"], [.445, .195, 0.])
    np.testing.assert_allclose(data["corners"], [
        [-.225, -.1, 1.], [.22, -.1, 1.], [.22, .095, 1.], [-.225, .095, 1.]])


def test_each_corner_uses_own_depth_and_nearest_point_defines_front():
    depth = np.full((120, 160), 2.)
    depth[40, 35], depth[40, 124], depth[79, 124], depth[79, 35] = 1.2, 1.4, 1.6, 1.8
    depth[60, 80] = .8
    data = estimate_bbox3d(depth, calibration(), [35, 40, 125, 80], depth_band=5., edge_min_jump=5.)
    np.testing.assert_allclose(data["corner_depths"], [1.2, 1.4, 1.6, 1.8])
    np.testing.assert_allclose(data["nearest_point"], [0., 0., .8])
    np.testing.assert_allclose(data["depth_range"], [.8, 1.8])
    np.testing.assert_allclose(data["position"], [-.0265, .0155, 1.3])
    np.testing.assert_allclose(data["size"], [.757, .311, 1.])
    assert data["nearest_depth"] == .8


def test_nearest_depth_ignores_invalid_and_out_of_range_values():
    depth = np.ones((120, 160))
    depth[59:61, 79:82] = [[0., np.nan, np.inf], [.01, 6., .7]]
    data = estimate_bbox3d(depth, calibration(), [35, 40, 125, 80], depth_band=5., edge_min_jump=5.)
    assert data["nearest_depth"] == .7
    assert data["size"][2] == pytest.approx(.3)


def test_four_corners_walk_inward_and_unproject_actual_sample_pixels():
    depth = np.zeros((120, 160))
    samples = [(43, 23), (117, 23), (117, 97), (43, 97)]
    zs = [1., 1.2, 1.4, 1.6]
    for (u, v), z in zip(samples, zs):
        depth[v, u] = z
    data = estimate_bbox3d(depth, calibration(), [40, 20, 121, 101], depth_band=5., edge_min_jump=5.)
    assert data["corner_sample_pixels"] == [list(p) for p in samples]
    assert data["corner_pixels"] == [[40, 20], [120, 20], [120, 100], [40, 100]]
    assert data["corner_depths"] == zs
    expected = [[(u-80)*z/200, (v-60)*z/200, z] for (u, v), z in zip(samples, zs)]
    np.testing.assert_allclose(data["corners"], expected)


def test_corner_search_uses_nearest_pixel_not_smallest_depth_or_off_line_pixel():
    depth = np.ones((120, 160))
    depth[20, 40] = 0.
    depth[21, 41] = np.nan
    depth[22, 42] = .01
    depth[23, 43] = 6.
    depth[24, 44] = 1.5  # First valid along the line.
    depth[25, 45] = .3  # Closer Z but farther from the original corner.
    depth[20, 41] = .2  # Neighbor outside the line must not be selected.
    data = estimate_bbox3d(depth, calibration(), [40, 20, 121, 101], depth_band=5., edge_min_jump=5.)
    assert data["corner_sample_pixels"][0] == [44, 24]
    assert data["corner_depths"][0] == 1.5
    assert data["nearest_depth"] == .2


def test_corner_line_without_valid_depth_rejects_object_even_with_valid_neighbors():
    depth = np.ones((120, 160))
    for offset in range(41):
        depth[20+offset, 40+offset] = 0.
    items = [dict(label="medicine", score=.9, bbox=[40, 20, 121, 101])]
    attach_bboxes3d(items, depth, calibration())
    assert items[0]["bbox3d_status"] == "no_reasonable_corner_depth:0"
    assert "bbox3d" not in items[0]


def test_line_search_includes_center_and_clipped_image_boundaries():
    # Top-left, top-right and bottom-left first become valid at the center.
    depth = np.zeros((120, 160))
    depth[60, 80] = 1.
    depth[119, 159] = 1.
    with pytest.raises(ValueError, match="empty_projected_bbox"):
        # All four resolve to one point; a volume cannot be inferred.
        estimate_bbox3d(depth, calibration(), [40, 20, 121, 101], depth_band=5., edge_min_jump=5.)
    data = estimate_bbox3d(depth, calibration(), [-10, -10, 180, 130])
    assert data["corner_sample_pixels"][:2] == [[80, 60], [80, 60]]
    assert data["corner_sample_pixels"][2] == [159, 119]
    assert data["corner_sample_pixels"][3] == [80, 60]


@pytest.mark.parametrize("value", [0., np.nan, np.inf, .01, 6.])
def test_missing_depth_does_not_invent_distance(value):
    items = [dict(label="medicine", score=.9, bbox=[35, 40, 125, 80])]
    attach_bboxes3d(items, np.full((120, 160), value), calibration())
    assert items[0]["bbox3d_status"] == "no_valid_bbox_depth"
    assert "bbox3d" not in items[0]
    assert not bboxes3d_message(items, Header()).detections


def test_ros_bbox_filters_unknown_and_includes_points():
    items = [dict(label=label, score=.9, track_id=i, bbox=[35, 40, 125, 80])
             for i, label in enumerate(["medicine", "unknown"], 1)]
    attach_bboxes3d(items, np.ones((120, 160)), calibration())
    header = Header(frame_id="camera_color_optical_frame")
    message = bboxes3d_message(items, header)
    assert len(message.detections) == 1
    for item, detection in zip(items, message.detections):
        assert detection.header == header
        assert detection.id == str(item["track_id"])
        assert detection.bbox.center.position.z == 1.
        assert detection.bbox.center.orientation.w == 1.
        assert detection.bbox.size.x == pytest.approx(.445)
        assert detection.bbox.size.y == pytest.approx(.195)
        assert detection.bbox.size.z == 0.
        assert detection.hypothesis.class_id == item["label"]
        assert not hasattr(detection, "results")
        assert "pose" not in item and "pose_status" not in item
    assert not bboxes3d_message([items[1]], header).detections
    assert "bbox3d" in items[1]  # Internal results remain available.
    assert not bboxes3d_message([], header).detections


def test_clips_bbox_and_rejects_empty_or_invalid_intrinsics():
    depth = np.ones((120, 160))
    data = estimate_bbox3d(depth, calibration(), [-20, -10, 180, 130])
    np.testing.assert_allclose(data["size"], [.795, .595, 0.])
    with pytest.raises(ValueError, match="empty_bbox"):
        estimate_bbox3d(depth, calibration(), [200, 10, 220, 30])
    info = calibration()
    info.k[0] = 0.
    with pytest.raises(ValueError, match="invalid_camera_intrinsics"):
        estimate_bbox3d(depth, info, [35, 40, 125, 80])


def test_fisheye_unprojection_contains_all_sampled_points():
    info = calibration()
    info.distortion_model = "equidistant"
    info.d = [0.] * 4
    depth = np.full((120, 160), 1.5)
    depth[60, 100] = .8
    data = estimate_bbox3d(depth, info, [90, 30, 140, 90], depth_band=5., edge_min_jump=5.)
    points = np.asarray(data["corners"] + [data["nearest_point"]])
    offsets = np.abs(points - data["position"])
    assert np.all(offsets <= np.asarray(data["size"]) / 2 + 1e-12)
    assert data["size"][2] == pytest.approx(.7)


def test_ros_points_preserve_original_corners_actual_samples_and_all_box_vertices():
    from rclpy.serialization import serialize_message, deserialize_message

    depth = np.zeros((120, 160))
    samples = [(43, 23), (117, 23), (117, 97), (43, 97)]
    for (u, v), z in zip(samples, [1., 1.2, 1.4, 1.6]):
        depth[v, u] = z
    depth[60, 80] = .8
    item = dict(label="medicine", score=.9, track_id=42, bbox=[40, 20, 121, 101])
    attach_bboxes3d([item], depth, calibration(), depth_band=5., edge_min_jump=5.)
    message = bboxes3d_message([item], Header(frame_id="camera"))
    # Exercise generated ROS typesupport and wire serialization of fixed arrays.
    received = deserialize_message(serialize_message(message), type(message))
    assert received == message
    detection = received.detections[0]
    assert [[p.x, p.y] for p in detection.bbox_2d_corners] == [
        [40, 20], [121, 20], [121, 101], [40, 101]]
    assert [[p.x, p.y] for p in detection.sampled_2d_points] == [list(p) for p in samples]
    np.testing.assert_allclose([[p.x, p.y, p.z] for p in detection.sampled_3d_points],
                               item["bbox3d"]["corners"])
    assert [detection.nearest_2d_point.x, detection.nearest_2d_point.y] == [80, 60]
    assert detection.nearest_3d_point.z == .8
    vertices = np.array([[p.x, p.y, p.z] for p in detection.bbox_3d_corners])
    lower = np.asarray(item["bbox3d"]["position"]) - np.asarray(item["bbox3d"]["size"]) / 2
    upper = np.asarray(item["bbox3d"]["position"]) + np.asarray(item["bbox3d"]["size"]) / 2
    np.testing.assert_allclose(vertices.min(axis=0), lower)
    np.testing.assert_allclose(vertices.max(axis=0), upper)
    np.testing.assert_allclose(vertices[:4, 2], lower[2])
    np.testing.assert_allclose(vertices[4:, 2], upper[2])
    assert len(np.unique(vertices, axis=0)) == 8


def test_corner_search_skips_valid_far_background_that_previously_shifted_box():
    depth = np.full((120, 160), 1.371)
    depth[30:91, 50:111] = .31
    depth[60, 80] = .296
    data = estimate_bbox3d(depth, calibration(), [40, 20, 121, 101])
    assert data["corner_sample_pixels"] == [[50, 30], [110, 30], [110, 90], [50, 90]]
    np.testing.assert_allclose(data["corner_depths"], [.31] * 4)
    np.testing.assert_allclose(data["position"], [0., 0., .303])
    np.testing.assert_allclose(data["size"], [.093, .093, .014])
    assert data["foreground_depth_band"] == .08


def test_background_only_corner_ray_rejects_box_instead_of_dragging_it_back():
    depth = np.full((120, 160), 1.371)
    depth[25, 80] = .3  # Nearest point is outside the four rays.
    with pytest.raises(ValueError, match="no_reasonable_corner_depth:0"):
        estimate_bbox3d(depth, calibration(), [40, 20, 121, 101])


@pytest.mark.parametrize("band", [0., -1., np.nan, np.inf])
def test_invalid_foreground_depth_band(band):
    with pytest.raises(ValueError, match="invalid_bbox_depth_band"):
        estimate_bbox3d(np.ones((120, 160)), calibration(), [40, 20, 121, 101], depth_band=band)


def test_center_outward_selects_center_side_of_largest_depth_jump():
    # Two outward steps: .300 -> .320 (not the edge), then .320 -> 1.371.
    v, u = np.indices((120, 160))
    radius = np.maximum(np.abs(u-80), np.abs(v-60))
    depth = np.where(radius <= 10, .3, np.where(radius <= 25, .32, 1.371))
    data = estimate_bbox3d(depth, calibration(), [40, 20, 121, 101])
    assert data["corner_sample_pixels"] == [[55, 35], [105, 35], [105, 85], [55, 85]]
    np.testing.assert_allclose(data["corner_depths"], [.32]*4)
    np.testing.assert_allclose(data["corner_edge_jumps"], [1.051]*4)
    assert data["corner_sampling_methods"] == ["max_depth_jump"]*4


def test_depth_hole_is_not_treated_as_a_huge_edge():
    depth = np.full((120, 160), .3)
    depth[40, 60] = np.nan
    depth[39, 59] = 1.371  # Isolated valid far pixel beyond the hole.
    # The other side of that isolated pixel is a valid jump toward near depth,
    # but its center-side depth is background, so it is not an object edge.
    data = estimate_bbox3d(depth, calibration(), [40, 20, 121, 101])
    assert data["corner_sample_pixels"][0] == [40, 20]
    assert data["corner_sampling_methods"][0] == "reasonable_depth_fallback"


def test_flat_surface_falls_back_without_inventing_edge():
    data = estimate_bbox3d(np.full((120, 160), .3), calibration(), [40, 20, 121, 101])
    assert data["corner_sample_pixels"] == [[40, 20], [120, 20], [120, 100], [40, 100]]
    assert data["corner_sampling_methods"] == ["reasonable_depth_fallback"]*4
    assert data["corner_edge_jumps"] == [0.]*4


@pytest.mark.parametrize("jump", [0., -1., np.nan, np.inf])
def test_invalid_minimum_edge_jump(jump):
    with pytest.raises(ValueError, match="invalid_bbox_edge_min_jump"):
        estimate_bbox3d(np.ones((120, 160)), calibration(), [40, 20, 121, 101], edge_min_jump=jump)


def test_invalid_neighbor_depths_are_excluded_before_subtraction():
    depth = np.full((120, 160), .3)
    depth[40, 60] = np.inf
    depth[39, 59] = np.inf
    with np.errstate(invalid="raise"):
        data = estimate_bbox3d(depth, calibration(), [40, 20, 121, 101])
    assert data["corner_sampling_methods"][0] == "reasonable_depth_fallback"
