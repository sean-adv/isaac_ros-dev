"""RGB-D synchronization, depth units and inference snapshot lifetime."""

from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock
import threading

import numpy as np
import pytest
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from builtin_interfaces.msg import Time

from dinov3_cases.depth_sync import matched_inputs
from dinov3_cases.node import CaseNode


def header(seconds=10.0, frame="camera_color_optical_frame"):
    return Header(stamp=Time(sec=int(seconds), nanosec=round((seconds % 1) * 1e9)), frame_id=frame)


def calibration():
    return CameraInfo(header=header(), width=160, height=120,
                      k=[200., 0., 80., 0., 200., 60., 0., 0., 1.],
                      distortion_model="plumb_bob", d=[0.] * 5)


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
    values = dict(pose_min_depth=.1, pose_max_depth=5., pose_enabled=True,
                  pose_sync_tolerance=.05, bbox3d_depth_band=.08, bbox3d_min_points=80, bbox3d_quantile=.02)
    from dinov3_cases.depth_bbox import ProjectionRayCache
    node = SimpleNamespace(depth_ray_cache=ProjectionRayCache(),
                           param=values.__getitem__, bridge=Mock(), stop=threading.Event(),
                           depth_changed=threading.Condition(), depth_frames=deque(),
                           depth_infos=deque([calibration()]))
    frame = Image(header=header(), width=160, height=120)
    depth = Image(header=header(), width=160, height=120, encoding=encoding)
    node.depth_frames.append(depth)
    inputs = CaseNode.depth_inputs_for(node, frame)
    node.depth_frames.clear()  # Long inference may outlive the ring buffer.
    node.bridge.imgmsg_to_cv2.return_value = np.full((120, 160), scale).astype(
        np.uint16 if encoding == "16UC1" else np.float32)
    items = [dict(label="medicine", score=.9, bbox=[35, 40, 125, 80])]
    CaseNode.estimate_bboxes3d(node, items, inputs)
    assert items[0]["bbox3d"]["position"][2] == pytest.approx(1.)
    CaseNode.estimate_bboxes3d(node, [dict(label="medicine")], None)
