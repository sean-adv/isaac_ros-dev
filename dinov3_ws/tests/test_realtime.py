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
        pending_publication=(None, [], Header(frame_id="first"), 0),
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
            node.pending_publication = (None, [], Header(frame_id="obsolete"), 0)
            node.pending_publication = (None, [], Header(frame_id="latest"), 0)
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
