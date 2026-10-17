"""Latency regressions: latest-only streams and start-to-start scheduling."""

from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock, patch
import threading

import pytest

from dinov3_cases.node import CaseNode
from dinov3_cases.web import create_app


def test_rate_limit_subtracts_processing_time():
    node = SimpleNamespace(image_changed=Mock(), stop=Mock(), param=lambda _: 20.0)
    node.stop.is_set.return_value = False
    node.stop.wait.return_value = False
    with patch("dinov3_cases.node.time.monotonic", side_effect=[0.0, 0.0, 0.03, 0.05, 0.15, 0.15]):
        ticks = CaseNode.worker_ticks(node)
        next(ticks)
        next(ticks)
        next(ticks)
    assert [call.args[0] for call in node.stop.wait.call_args_list] == pytest.approx(
        [0.0, 0.02, 0.0]
    )


def test_stream_skips_obsolete_frames_and_terminates(tmp_path):
    lock = threading.Lock()
    node = SimpleNamespace(
        root=tmp_path,
        lock=lock,
        stream_changed=threading.Condition(lock),
        stop=threading.Event(),
        jpeg=b"first",
        preview_jpeg=b"camera",
    )
    app = create_app(node)
    with app.test_request_context("/overlay.mjpg"):
        response = app.view_functions["overlay_stream"]()
        frames = iter(response.response)
        assert b"first" in next(frames)
        with node.stream_changed:
            node.jpeg = b"obsolete"
            node.jpeg = b"latest"
            node.stream_changed.notify_all()
        part = next(frames)
        assert b"latest" in part and b"obsolete" not in part
        assert response.headers["X-Accel-Buffering"] == "no"
        node.stop.set()
        with pytest.raises(StopIteration):
            next(frames)


def test_requested_debug_revision_survives_new_inference(tmp_path):
    lock = threading.Lock()
    old = Mock(reference_cases=[])
    old.render.return_value = ([], {(0, "crop"): b"old-jpeg"})
    node = SimpleNamespace(
        root=tmp_path,
        lock=lock,
        object_debug_revision=1,
        object_debug_snapshot=old,
        object_debug_history=deque(maxlen=3),
    )
    client = create_app(node).test_client()
    assert client.get("/api/object_visualizations").status_code == 200
    for revision in range(2, 20):
        node.object_debug_revision = revision
        node.object_debug_snapshot = Mock()
    assert client.get("/debug/objects/1/0/crop.jpg").data == b"old-jpeg"


def test_blocked_ros_output_does_not_lock_input_and_skips_pending_frames():
    from std_msgs.msg import Header

    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    captured = []

    def publish(message):
        captured.append(message.header.frame_id)
        if len(captured) == 1:
            entered.set()
            assert release.wait(3)
        else:
            finished.set()

    node = SimpleNamespace(
        stop=threading.Event(),
        publish_changed=threading.Event(),
        lock=threading.Lock(),
        generation=0,
        pending_publication=([], Header(frame_id="first"), 0),
        recognition_pub=Mock(),
        bbox3d_pub=Mock(),
        pub=Mock(),
        recognition_debug_pub=Mock(),
        bridge=Mock(),
        get_logger=lambda: Mock(),
    )
    node.bbox3d_pub.get_subscription_count.return_value = 0
    node.recognition_pub.get_subscription_count.return_value = 1
    node.recognition_pub.publish.side_effect = publish
    node.pub.get_subscription_count.return_value = 0
    node.recognition_debug_pub.get_subscription_count.return_value = 0
    worker = threading.Thread(target=CaseNode.run_publications, args=(node,))
    node.publish_changed.set()
    worker.start()
    try:
        assert entered.wait(3)
        assert node.lock.acquire(timeout=1), "DDS publication must not hold the input lock"
        try:
            node.pending_publication = ([], Header(frame_id="obsolete"), 0)
            node.pending_publication = ([], Header(frame_id="latest"), 0)
            node.publish_changed.set()
        finally:
            node.lock.release()
        release.set()
        assert finished.wait(3)
        assert captured == ["first", "latest"]
        node.bridge.cv2_to_imgmsg.assert_not_called()
    finally:
        release.set()
        node.stop.set()
        node.publish_changed.set()
        worker.join(3)
        assert not worker.is_alive()


@pytest.mark.parametrize("mode", ["2d", "3d"])
def test_jpeg_block_does_not_block_bbox_publication_and_frame_is_leased(mode):
    import numpy as np
    from std_msgs.msg import Header
    from dinov3_cases.frame_pool import SharedFramePool

    pool = SharedFramePool()
    frame = pool.acquire(np.zeros((8, 8, 3), np.uint8))
    entered, release, published = threading.Event(), threading.Event(), threading.Event()
    lock = threading.Lock()
    node = SimpleNamespace(
        stop=threading.Event(), lock=lock, generation=0,
        render_changed=threading.Event(), publish_changed=threading.Event(),
        pending_render=(frame.retain(), [], Header(), 0, 7, None, 1.0),
        pending_publication=([], Header(frame_id="bbox"), 0),
        stream_changed=threading.Condition(lock), performance={"completed_at": 1.0},
        jpeg=None, bbox_visualization_mode=mode, stream_viewers={}, stream_requested_at={},
        pub=Mock(), recognition_debug_pub=Mock(), recognition_pub=Mock(), bbox3d_pub=Mock(),
        tag_pub=Mock(), param=lambda _: 80, bridge=Mock(), get_logger=lambda: Mock(),
    )
    node.wants_stream = lambda attr: CaseNode.wants_stream(node, attr)
    for pub in (node.pub, node.recognition_debug_pub, node.bbox3d_pub, node.tag_pub):
        pub.get_subscription_count.return_value = 0
    node.recognition_pub.get_subscription_count.return_value = 1
    node.recognition_pub.publish.side_effect = lambda _: published.set()

    def encode(*args):
        entered.set()
        assert release.wait(3)
        return True, np.array([1, 2], dtype=np.uint8)

    renderer = threading.Thread(target=CaseNode.run_rendering, args=(node,))
    publisher = threading.Thread(target=CaseNode.run_publications, args=(node,))
    try:
        with patch("dinov3_cases.node.cv2.imencode", side_effect=encode) as jpeg_encode, \
                patch("dinov3_cases.node.render_display_overlay",
                      side_effect=lambda pixels, *args: pixels.copy()) as render:
            renderer.start()
            publisher.start()
            node.render_changed.set()
            assert entered.wait(3)
            frame.release()
            assert pool.slots[0][1] == 1  # Renderer still owns the raw frame.
            node.publish_changed.set()
            assert published.wait(1), "bbox must not wait for JPEG"
            release.set()
            with node.stream_changed:
                assert node.stream_changed.wait_for(lambda: node.jpeg is not None, timeout=3)
            assert node.performance["overlay_frame_id"] == 7
            assert render.call_count == 1
            assert render.call_args.args[3] == mode
            assert jpeg_encode.call_count == 1
    finally:
        release.set()
        node.stop.set()
        node.render_changed.set()
        node.publish_changed.set()
        renderer.join(3)
        publisher.join(3)
        frame.release()
        pool.close()
    assert all(slot[0] is None for slot in pool.slots)


@pytest.mark.parametrize("ros_reader", [False, True])
def test_browser_renders_small_canvas_and_ros_full_size_only_with_subscribers(ros_reader):
    import numpy as np
    import cv2
    from std_msgs.msg import Header
    from dinov3_cases.visualization import render_bbox_overlay, render_display_overlay

    lock = threading.Lock()
    frame = Mock(pixels=np.zeros((1080, 1920, 3), np.uint8))
    publisher = Mock()
    publisher.get_subscription_count.return_value = int(ros_reader)
    node = SimpleNamespace(
        stop=Mock(), lock=lock, generation=0,
        render_changed=threading.Event(),
        pending_render=(frame, [], Header(), 0, 7, None, 1.0),
        stream_changed=threading.Condition(lock), performance={"completed_at": 1.0},
        jpeg=None, bbox_visualization_mode="3d", wants_stream=lambda _: True,
        pub=publisher, recognition_debug_pub=Mock(), param=lambda _: 70,
        bridge=Mock(), get_logger=lambda: Mock(),
    )
    node.stop.is_set.side_effect = [False, True]
    node.recognition_debug_pub.get_subscription_count.return_value = 0
    node.render_changed.set()
    with patch("dinov3_cases.node.render_bbox_overlay", wraps=render_bbox_overlay) as full_render, \
            patch("dinov3_cases.node.render_display_overlay", wraps=render_display_overlay) as small_render:
        CaseNode.run_rendering(node)
    assert small_render.call_count == 1
    assert full_render.call_count == int(ros_reader)
    decoded = cv2.imdecode(np.frombuffer(node.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (540, 960, 3)
    if ros_reader:
        assert node.bridge.cv2_to_imgmsg.call_args.args[0].shape == (1080, 1920, 3)
        publisher.publish.assert_called_once()
    else:
        node.bridge.cv2_to_imgmsg.assert_not_called()
    frame.release.assert_called_once()
    node.get_logger().error.assert_not_called()
