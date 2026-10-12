"""Geometric pose accuracy, invalid depth rejection and RGB-D synchronization."""

from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock, patch
import threading

import numpy as np
import pytest
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from builtin_interfaces.msg import Time

from dinov3_cases.depth_pose import (
    estimate_surface_pose, attach_surface_poses, poses_message, matched_inputs,
    rotation_quaternion, draw_pose_axes,
)
from dinov3_cases.node import CaseNode


def header(seconds=10.0, frame="camera_color_optical_frame"):
    return Header(stamp=Time(sec=int(seconds), nanosec=round((seconds % 1) * 1e9)), frame_id=frame)


def calibration():
    return CameraInfo(header=header(), width=160, height=120,
                      k=[200., 0., 80., 0., 200., 60., 0., 0., 1.],
                      distortion_model="plumb_bob", d=[0.] * 5)


def plane(tilt=0.):
    v, u = np.indices((120, 160))
    # Plane z = 1 + tilt*x, intersects camera rays.
    return 1. / (1. - tilt * (u - 80.) / 200.)


def test_tilted_plane_has_measured_normal_and_metric_translation():
    info = calibration()
    data = estimate_surface_pose(plane(0.4), info, [35, 40, 125, 80], depth_band=0.2)
    qx, qy, qz, qw = data["quaternion"]
    assert np.linalg.norm(data["quaternion"]) == pytest.approx(1.)
    normal = np.array([2*(qx*qz + qw*qy), 2*(qy*qz - qw*qx), 1-2*(qx*qx+qy*qy)])
    expected = np.array([-0.4, 0., 1.]) / np.sqrt(1.16)
    np.testing.assert_allclose(normal, expected, atol=1e-7)
    assert data["position"][2] == pytest.approx(1., abs=0.01)
    assert abs(data["position"][0]) < 0.01
    assert data["point_count"] >= 80
    assert max(data["size"]) < 0.6


def test_background_and_invalid_pixels_are_removed():
    depth = np.full((120, 160), 2.0)
    depth[40:80, 35:125] = 1.0
    depth[42:45, 38:48] = np.nan
    depth[65:68, 105:115] = 0
    pose = estimate_surface_pose(depth, calibration(), [20, 20, 140, 100])
    assert pose["position"][2] == pytest.approx(1.)
    assert max(pose["size"]) < 0.5


@pytest.mark.parametrize("depth, bbox, reason", [
    (np.zeros((120, 160)), [35, 40, 125, 80], "no_valid_center_depth"),
    (np.full((120, 160), np.inf), [35, 40, 125, 80], "no_valid_center_depth"),
    (plane(), [0, 0, 3, 3], "insufficient_depth_points"),
    (plane(), [50, 30, 110, 90], "ambiguous_surface_axes"),
])
def test_unobservable_poses_are_rejected(depth, bbox, reason):
    with pytest.raises(ValueError, match=reason):
        estimate_surface_pose(depth, calibration(), bbox)


def test_unknown_is_never_estimated_and_ros_contains_label_and_pose():
    items = [dict(label=label, score=0.9, bbox=[35, 40, 125, 80])
             for label in ["medicine_a", "unknown"]]
    attach_surface_poses(items, plane(), calibration())
    assert items[0]["pose_status"] == "ok"
    assert "pose" not in items[1]
    message = poses_message(items, header())
    assert message.header == header()
    assert len(message.detections) == 1
    result = message.detections[0].results[0]
    assert result.hypothesis.class_id == "medicine_a"
    assert result.pose.pose.position.z == pytest.approx(1.)
    assert result.pose.pose == message.detections[0].bbox.center
    assert message.detections[0].id == ""
    assert len(poses_message([], header()).detections) == 0


def test_sync_rejects_stale_unregistered_and_wrong_size_depth():
    info = calibration()
    valid = Image(header=header(10.02), width=160, height=120)
    frames = [Image(header=header(10.001, "camera_depth_optical_frame"), width=160, height=120),
              Image(header=header(10.0), width=80, height=60),
              Image(header=header(9.8), width=160, height=120), valid]
    assert matched_inputs(header(), 160, 120, frames, [info], .05) == (valid, info)
    assert matched_inputs(header(), 160, 120, frames[:-1], [info], .05) is None
    assert matched_inputs(header(), 160, 120, frames, [], .05) is None
    assert matched_inputs(header(0), 160, 120, frames, [info], .05) is None


@pytest.mark.parametrize("encoding, scale", [("16UC1", 1000), ("32FC1", 1)])
def test_node_converts_depth_units_and_keeps_original_snapshot(encoding, scale):
    values = dict(pose_min_depth=.1, pose_max_depth=5., pose_depth_band=.08,
                  pose_min_points=80, pose_axis_ratio=1.15, pose_enabled=True,
                  pose_sync_tolerance=.05)
    node = SimpleNamespace(param=values.__getitem__, bridge=Mock(), stop=threading.Event(),
                           depth_changed=threading.Condition(), depth_frames=deque(),
                           depth_infos=deque([calibration()]))
    frame = Image(header=header(), width=160, height=120)
    depth = Image(header=header(), width=160, height=120, encoding=encoding)
    node.depth_frames.append(depth)
    inputs = CaseNode.depth_inputs_for(node, frame)
    node.depth_frames.clear()  # Long inference may outlive the ring buffer.
    node.bridge.imgmsg_to_cv2.return_value = (plane() * scale).astype(
        np.uint16 if encoding == "16UC1" else np.float32)
    items = [dict(label="medicine", score=.9, bbox=[35, 40, 125, 80])]
    CaseNode.estimate_poses(node, items, inputs)
    assert items[0]["pose"]["position"][2] == pytest.approx(1.)
    CaseNode.estimate_poses(node, [dict(label="medicine")], None)


def test_unit_quaternion_handles_half_turn():
    q = rotation_quaternion(np.diag([1., -1., -1.]))
    np.testing.assert_allclose(abs(q[0]), 1., atol=1e-12)
    np.testing.assert_allclose(q[1:], 0., atol=1e-12)


@pytest.mark.parametrize("angle", [0., np.pi / 2])
def test_pose_axes_follow_object_rotation_and_metric_projection(angle):
    import cv2

    image = np.zeros((120, 160, 3), dtype=np.uint8)
    pose = dict(position=[0., 0., 1.],
                quaternion=[0., 0., np.sin(angle / 2), np.cos(angle / 2)])
    with patch("dinov3_cases.depth_pose.cv2.arrowedLine", wraps=cv2.arrowedLine) as draw:
        draw_pose_axes(image, pose, calibration())
    colored = [call.args for call in draw.call_args_list if call.args[3] != (0, 0, 0)]
    assert [args[3] for args in colored] == [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    assert all(args[1] == (80, 60) for args in colored)
    # 8 cm at 1 m projects to 16 pixels with focal length 200.
    expected = [(96, 60), (80, 76), (80, 60)] if angle == 0 else [
        (80, 76), (64, 60), (80, 60)]
    assert [args[2] for args in colored] == expected
    assert image.any()


def test_pose_axes_use_fisheye_projection():
    import cv2

    info = calibration()
    info.distortion_model = "equidistant"
    info.d = [0.] * 4
    pose = dict(position=[.2, .1, 1.], quaternion=[0., 0., 0., 1.])
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    with patch("dinov3_cases.depth_pose.cv2.fisheye.projectPoints",
               wraps=cv2.fisheye.projectPoints) as project:
        draw_pose_axes(image, pose, info)
    project.assert_called_once()
    assert image.any()


@pytest.mark.parametrize("position, quaternion", [
    ([0., 0., -1.], [0., 0., 0., 1.]),
    ([0., 0., 0.], [0., 0., 0., 1.]),
    ([np.nan, 0., 1.], [0., 0., 0., 1.]),
    ([0., 0., 1.], [0., 0., 0., 0.]),
])
def test_pose_axes_skip_unprojectable_poses(position, quaternion):
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    draw_pose_axes(image, dict(position=position, quaternion=quaternion), calibration())
    assert not image.any()
