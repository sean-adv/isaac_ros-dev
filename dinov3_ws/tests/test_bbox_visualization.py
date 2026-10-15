"""Calibrated wireframe projection and independent web overlay selection."""
from types import SimpleNamespace
from unittest.mock import patch
import threading

import cv2
import numpy as np
import pytest
from sensor_msgs.msg import CameraInfo

from dinov3_cases.visualization import draw_bbox3d, render_bbox_overlay
from dinov3_cases.web import create_app


def calibration(model="plumb_bob"):
    return CameraInfo(width=160, height=120,
                      k=[200., 0., 80., 0., 200., 60., 0., 0., 1.],
                      distortion_model=model, d=[0.] * (4 if model == "equidistant" else 5))


def box():
    return dict(position=[0., 0., 1.], size=[.2, .1, .2])


def item():
    return dict(label="medicine", track_id=1, detection_label="box", bbox=[40, 20, 121, 101],
                bbox_visible=True, bbox3d=box())


def test_projects_twelve_edges_with_perspective_and_metric_dimensions():
    image = np.zeros((120, 160, 3), np.uint8)
    with patch("dinov3_cases.visualization.cv2.line", wraps=cv2.line) as line:
        assert draw_bbox3d(image, box(), calibration(), (60, 220, 80))
    assert line.call_count == 12
    segments = [(call.args[1], call.args[2]) for call in line.call_args_list]
    assert segments[0] == ((58, 49), (102, 49))  # Front face at Z=.9.
    assert segments[4] == ((62, 51), (98, 51))  # Back face at Z=1.1.
    assert segments[8] == ((58, 49), (62, 51))
    assert image.any()


def test_fisheye_projection_and_clipped_edges():
    image = np.zeros((120, 160, 3), np.uint8)
    with patch("dinov3_cases.visualization.cv2.fisheye.projectPoints",
               wraps=cv2.fisheye.projectPoints) as project:
        assert draw_bbox3d(image, box(), calibration("equidistant"), (60, 220, 80))
    assert project.call_count == 1
    assert draw_bbox3d(image, dict(position=[.3, 0., 1.], size=[.8, .4, .2]),
                       calibration(), (60, 220, 80))


@pytest.mark.parametrize("data", [dict(position=[0., 0., -.1], size=[.2]*3),
                                  dict(position=[np.nan, 0., 1.], size=[.2]*3)])
def test_does_not_project_invalid_or_behind_camera_box(data):
    image = np.zeros((120, 160, 3), np.uint8)
    assert not draw_bbox3d(image, data, calibration(), (60, 220, 80))
    assert not image.any()


def test_modes_share_input_without_drawing_2d_fallback_in_3d():
    image = np.zeros((120, 160, 3), np.uint8)
    detection = item()
    two = render_bbox_overlay(image, [detection], calibration(), "2d")
    three = render_bbox_overlay(image, [detection], calibration(), "3d")
    assert two.any() and three.any() and not np.array_equal(two, three)
    assert not image.any()
    assert np.array_equal(render_bbox_overlay(image, [detection], None, "3d"), image)
    del detection["bbox3d"]
    assert np.array_equal(render_bbox_overlay(image, [detection], calibration(), "3d"), image)
    detection = item()
    detection["label"] = "unknown"
    assert np.array_equal(render_bbox_overlay(image, [detection], calibration(), "3d"), image)
    detection = item()
    detection["bbox_visible"] = False
    assert np.array_equal(render_bbox_overlay(image, [detection], calibration(), "3d"), image)


def test_web_modes_serve_separate_cached_images_and_validate_selection(tmp_path):
    lock = threading.Lock()
    node = SimpleNamespace(root=tmp_path, lock=lock, stream_changed=threading.Condition(lock),
                           stop=threading.Event(), jpeg=b"two-dimensional", jpeg3d=b"three-dimensional")
    app = create_app(node)
    client = app.test_client()
    assert client.get('/overlay.jpg').data == b"two-dimensional"
    assert client.get('/overlay.jpg?bbox=2d').data == b"two-dimensional"
    assert client.get('/overlay.jpg?bbox=3d').data == b"three-dimensional"
    assert client.get('/overlay.jpg?bbox=wrong').status_code == 400
    assert client.get('/overlay.mjpg?bbox=wrong').status_code == 400
    with app.test_request_context('/overlay.mjpg?bbox=3d'):
        response = app.view_functions['overlay_stream']()
        frames = iter(response.response)
        assert b"three-dimensional" in next(frames)
        frames.close()
    html = client.get('/').data
    assert b'id="bbox-view"' in html and b'value="3d"' in html


def test_frame_local_detection_renders_without_track_id():
    image = np.zeros((200, 300, 3), np.uint8)
    detection = dict(bbox=[60, 80, 180, 180], label="unknown",
                     detection_label="person", bbox_visible=True)
    rendered = render_bbox_overlay(image, [detection])
    assert not np.array_equal(rendered, image)
    assert "track_id" not in detection
