"""Calibrated wireframe projection and configuration-selected web overlays."""
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
    detection["bbox_visible"] = False
    assert np.array_equal(render_bbox_overlay(image, [detection], calibration(), "3d"), image)


@pytest.mark.parametrize("mode", ["2d", "3d"])
def test_web_serves_only_configured_overlay_and_has_no_mode_switch(tmp_path, mode):
    lock = threading.Lock()
    cached = (mode + "-configured").encode()
    node = SimpleNamespace(root=tmp_path, lock=lock, stream_changed=threading.Condition(lock),
                           stop=threading.Event(), jpeg=cached, bbox_visualization_mode=mode)
    app = create_app(node)
    client = app.test_client()
    assert client.get('/overlay.jpg').data == cached
    # Old browser URLs cannot select another mode or trigger another render.
    for query in ('2d', '3d', 'wrong'):
        assert client.get('/overlay.jpg?bbox=' + query).data == cached
    with app.test_request_context('/overlay.mjpg?bbox=2d'):
        response = app.view_functions['overlay_stream']()
        frames = iter(response.response)
        assert cached in next(frames)
        frames.close()
    html = client.get('/').data
    assert b'id="bbox-view"' not in html
    assert (b'Showing projected 3D' if mode == '3d' else b'Showing original 2D') in html


def test_frame_local_detection_renders_without_track_id():
    image = np.zeros((200, 300, 3), np.uint8)
    detection = dict(bbox=[60, 80, 180, 180], label="unknown",
                     detection_label="person", bbox_visible=True)
    rendered = render_bbox_overlay(image, [detection])
    assert not np.array_equal(rendered, image)
    assert "track_id" not in detection


@pytest.mark.parametrize("label", ["medicine", "unknown"])
@pytest.mark.parametrize("visible", [True, False])
def test_2d_and_3d_share_visibility_for_all_labels(label, visible):
    image = np.zeros((120, 160, 3), np.uint8)
    detection = item()
    detection.update(label=label, bbox_visible=visible)
    for mode in ("2d", "3d"):
        rendered = render_bbox_overlay(image, [detection], calibration(), mode)
        assert bool(rendered.any()) == visible
        if visible:
            color = (0, 190, 255) if label == "unknown" else (60, 220, 80)
            assert np.any(np.all(rendered == color, axis=2))


@pytest.mark.parametrize("model", ["plumb_bob", "equidistant"])
def test_browser_projection_scales_pixels_without_changing_metric_data(model):
    from copy import deepcopy
    from dinov3_cases.visualization import render_display_overlay

    image = np.zeros((120, 160, 3), np.uint8)
    detection, info = item(), calibration(model)
    info.d = [.1, -.02, .001, .002] + ([] if model == "equidistant" else [.005])
    original_detection, original_info = deepcopy(detection), deepcopy(info)
    project_name = ("dinov3_cases.visualization.cv2.fisheye.projectPoints"
                    if model == "equidistant" else "dinov3_cases.visualization.cv2.projectPoints")
    project = cv2.fisheye.projectPoints if model == "equidistant" else cv2.projectPoints
    projections = []

    def capture(*args, **kwargs):
        result = project(*args, **kwargs)
        projections.append(result[0].reshape(-1, 2).copy())
        return result

    with patch(project_name, side_effect=capture), \
            patch("dinov3_cases.visualization.render_bbox_overlay",
                  wraps=render_bbox_overlay) as render:
        render_bbox_overlay(image, [detection], info, "3d")
        display = render_display_overlay(image, [detection], info, "3d")
    assert display.shape == (540, 960, 3)
    assert display.any() and not image.any()
    np.testing.assert_allclose(projections[1], projections[0] * [6., 4.5])
    scaled_item = render.call_args.args[1][0]
    assert scaled_item["bbox"] == [240, 90, 721, 451]
    assert scaled_item["bbox3d"] == original_detection["bbox3d"]
    assert detection == original_detection and info == original_info


def test_browser_2d_coordinates_and_resize_before_color_conversion():
    from dinov3_cases.visualization import render_display_overlay

    rgb = np.full((1080, 1920, 3), [200, 80, 30], dtype=np.uint8)
    detection = item()
    with patch("dinov3_cases.visualization.cv2.cvtColor", wraps=cv2.cvtColor) as convert, \
            patch("dinov3_cases.visualization.cv2.rectangle", wraps=cv2.rectangle) as rectangle:
        display = render_display_overlay(rgb, [detection], mode="2d")
    assert convert.call_args.args[0].shape == (540, 960, 3)
    assert rectangle.call_args.args[1:3] == ((20, 10), (60, 50))
    assert tuple(display[-1, -1]) == (30, 80, 200)
    assert tuple(rgb[-1, -1]) == (200, 80, 30)


def test_wireframe_projects_rotated_local_axes():
    image = np.zeros((120, 160, 3), np.uint8)
    data = dict(position=[0., 0., 1.], size=[.2, .1, .2],
                rotation=[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    with patch("dinov3_cases.visualization.cv2.line", wraps=cv2.line) as line:
        assert draw_bbox3d(image, data, calibration(), (60, 220, 80))
    segments = [(call.args[1], call.args[2]) for call in line.call_args_list]
    # Local X edge is vertical after a 90-degree rotation about camera Z.
    assert segments[0] == ((91, 38), (91, 82))
    assert segments[4] == ((89, 42), (89, 78))
    assert line.call_count == 12
