"""The main page sends one adaptive stream and encodes only its needed view."""
import threading
import time
from types import SimpleNamespace

import pytest

from dinov3_cases.node import CaseNode
from dinov3_cases.web import create_app


def node_for(tmp_path, mode='3d'):
    lock = threading.Lock()
    return SimpleNamespace(
        root=tmp_path, lock=lock, stream_changed=threading.Condition(lock),
        stop=threading.Event(), source='camera', bbox_visualization_mode=mode,
        jpeg=None, preview_jpeg=b'raw-camera', overlay_at=None, overlay_has_boxes=False,
        detections=[], stream_viewers={}, stream_requested_at={},
    )


def set_boxes(node):
    node.detections = [dict(bbox_visible=True, bbox3d={'position': [0, 0, 1]})]
    node.overlay_has_boxes = True
    node.overlay_at = time.monotonic()
    node.jpeg = b'bbox-frame'


def test_single_connection_switches_camera_boxes_camera_and_cleans_up(tmp_path):
    node = node_for(tmp_path)
    app = create_app(node)
    with app.test_request_context('/view.mjpg'):
        response = app.view_functions['display_stream']()
        frames = iter(response.response)
        assert b'raw-camera' in next(frames)
        assert node.stream_viewers == {'display': 1}
        assert CaseNode.wants_stream(node, 'preview_jpeg')
        assert not CaseNode.wants_stream(node, 'jpeg')
        with node.stream_changed:
            set_boxes(node)
            node.stream_changed.notify_all()
        assert b'bbox-frame' in next(frames)
        assert CaseNode.wants_stream(node, 'jpeg')
        assert not CaseNode.wants_stream(node, 'preview_jpeg')
        with node.stream_changed:
            node.detections = []
            node.preview_jpeg = b'camera-after-loss'
            node.stream_changed.notify_all()
        assert b'camera-after-loss' in next(frames)
        assert CaseNode.wants_stream(node, 'preview_jpeg')
        assert not CaseNode.wants_stream(node, 'jpeg')
        frames.close()
        assert node.stream_viewers['display'] == 0


@pytest.mark.parametrize('reason', ['stale', 'hidden', 'missing_3d', 'not_rendered'])
def test_unusable_box_result_returns_to_camera(tmp_path, reason):
    node = node_for(tmp_path)
    set_boxes(node)
    if reason == 'stale':
        node.overlay_at -= 4
    elif reason == 'hidden':
        node.detections[0]['bbox_visible'] = False
    elif reason == 'missing_3d':
        del node.detections[0]['bbox3d']
    else:
        node.overlay_has_boxes = False
    app = create_app(node)
    with app.test_request_context('/view.mjpg'):
        frames = iter(app.view_functions['display_stream']().response)
        assert b'raw-camera' in next(frames)
        assert CaseNode.wants_stream(node, 'preview_jpeg')
        frames.close()


def test_2d_boxes_do_not_require_depth(tmp_path):
    node = node_for(tmp_path, '2d')
    set_boxes(node)
    del node.detections[0]['bbox3d']
    app = create_app(node)
    with app.test_request_context('/view.mjpg'):
        frames = iter(app.view_functions['display_stream']().response)
        assert b'bbox-frame' in next(frames)
        assert not CaseNode.wants_stream(node, 'preview_jpeg')
        frames.close()


def test_uploaded_image_without_boxes_is_preserved_and_never_expires(tmp_path):
    node = node_for(tmp_path)
    node.source = 'upload'
    node.jpeg = b'uploaded-target'
    node.overlay_at = time.monotonic()-100
    app = create_app(node)
    with app.test_request_context('/view.mjpg'):
        frames = iter(app.view_functions['display_stream']().response)
        assert b'uploaded-target' in next(frames)
        assert CaseNode.wants_stream(node, 'jpeg')
        assert not CaseNode.wants_stream(node, 'preview_jpeg')
        frames.close()


def test_explicit_camera_viewer_can_still_request_preview(tmp_path):
    node = node_for(tmp_path)
    set_boxes(node)
    node.stream_viewers = {'display': 1, 'preview_jpeg': 1}
    assert CaseNode.wants_stream(node, 'preview_jpeg')
    node.stream_viewers['preview_jpeg'] = 0
    assert not CaseNode.wants_stream(node, 'preview_jpeg')


def test_homepage_has_one_live_stream_and_keeps_camera_fallback_visible(tmp_path):
    node = node_for(tmp_path)
    client = create_app(node).test_client()
    html = client.get('/').get_data(as_text=True)
    assert html.count('src="/view.mjpg"') == 1
    assert 'src="/camera.mjpg"' not in html
    assert 'src="/overlay.mjpg"' not in html
    js = client.get('/static/cases.js').get_data(as_text=True)
    assert "el('preview')" not in js
    assert "style.display = d.overlay_ready" not in js


@pytest.mark.parametrize('boxes', [False, True])
def test_workers_encode_only_the_selected_view_for_main_page(tmp_path, boxes):
    from unittest.mock import Mock, patch
    import numpy as np

    node = node_for(tmp_path)
    node.stream_viewers['display'] = 1
    if boxes:
        set_boxes(node)
    frame = Mock(pixels=np.zeros((8, 8, 3), np.uint8))
    node.generation = 0
    node.pending_render = (frame, node.detections, None, 0, 7, None, 1.)
    node.performance = {'completed_at': 1.}
    node.render_changed = threading.Event()
    node.pub = Mock()
    node.recognition_debug_pub = Mock()
    node.pub.get_subscription_count.return_value = 0
    node.recognition_debug_pub.get_subscription_count.return_value = 0
    node.param = lambda _: 70
    node.get_logger = lambda: Mock()
    node.wants_stream = lambda attr: CaseNode.wants_stream(node, attr)
    node.stop = Mock()
    node.stop.is_set.side_effect = [False, True]
    node.render_changed.set()
    with patch('dinov3_cases.node.render_display_overlay', return_value=frame.pixels), \
            patch('dinov3_cases.node.encode_display_jpeg', return_value=b'new-bbox') as encode:
        CaseNode.run_rendering(node)
    assert encode.call_count == int(boxes)
    frame.release.assert_called_once()
    node.preview_changed = threading.Event()
    node.camera_frame, node.camera_seq = object(), 1
    node.bridge = Mock()
    node.bridge.imgmsg_to_cv2.return_value = frame.pixels
    node.stop.is_set.side_effect = [False, True]
    node.preview_changed.set()
    with patch('dinov3_cases.node.encode_display_jpeg', return_value=b'new-camera') as encode:
        CaseNode.run_preview(node)
    assert encode.call_count == int(not boxes)
