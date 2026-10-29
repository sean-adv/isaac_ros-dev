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
        bridge=Mock(),
        get_logger=lambda: Mock(),
    )
    node.bbox3d_pub.get_subscription_count.return_value = 0
    node.recognition_pub.get_subscription_count.return_value = 1
    node.recognition_pub.publish.side_effect = publish
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


def test_camera_input_fps_counts_arrivals_independently_of_result_fps():
    from dinov3_cases.timing import ResultRate

    node = SimpleNamespace(
        lock=threading.Lock(), camera_seq=0, camera_rate=ResultRate(),
        render_changed=threading.Event(), image_changed=threading.Event(),
        source='camera', frame_seq=0, performance={'result_fps': 20.0},
    )
    frame = SimpleNamespace(header=SimpleNamespace(frame_id='rgb'))
    with patch('dinov3_cases.node.time.monotonic', side_effect=[10., 10.+1/30, 10.+2/30]):
        for _ in range(3):
            CaseNode.on_image(node, frame)
    assert node.camera_fps == pytest.approx(30.)
    assert node.performance['result_fps'] == 20.
    assert node.camera_seq == node.frame_seq == 3
    assert node.frame is frame


def test_pending_overlay_cannot_exhaust_input_slots_during_sam3_and_rendering():
    import numpy as np
    from dinov3_cases.frame_pool import SharedFramePool

    pool = SharedFramePool()
    detector = pool.acquire(np.full((8, 8, 3), 1, np.uint8))
    rendering = pool.acquire(np.full((8, 8, 3), 2, np.uint8))
    queued = pool.acquire(np.full((8, 8, 3), 3, np.uint8))
    node = SimpleNamespace(frame_pool=pool, lock=threading.Lock(), pending_render=(queued,))
    fresh = None
    try:
        fresh = CaseNode.acquire_input_frame(
            node, np.full((8, 8, 3), 4, np.uint8), frame_id=4, timestamp=None,
        )
        assert fresh is not None, 'Queued rendering must yield a slot to current inference'
        assert node.pending_render is None
        assert queued.released
        np.testing.assert_array_equal(fresh.pixels, 4)
        np.testing.assert_array_equal(detector.pixels, 1)
        np.testing.assert_array_equal(rendering.pixels, 2)
        assert not detector.released and not rendering.released
    finally:
        if fresh is not None:
            fresh.release()
        detector.release()
        rendering.release()
        queued.release()
        pool.close()


@pytest.mark.parametrize('current_frame', [4, 5])
def test_ros_publication_timing_follows_its_frame(current_frame):
    from std_msgs.msg import Header

    stop = threading.Event()
    wake = threading.Event()
    wake.set()
    node = SimpleNamespace(
        stop=stop, publish_changed=wake, lock=threading.Lock(), generation=2,
        pending_publication=([], Header(frame_id='camera'), 2),
        pending_publication_timing=dict(frame_id=4, received_at=10., completed_at=19.),
        recognition_pub=Mock(), bbox3d_pub=Mock(), tag_pub=Mock(),
        performance=dict(frame_id=current_frame, completed_at=19.), get_logger=lambda: Mock(),
    )
    node.recognition_pub.get_subscription_count.return_value = 1
    node.recognition_pub.publish.side_effect = lambda _: stop.set()
    node.bbox3d_pub.get_subscription_count.return_value = 0
    node.tag_pub.get_subscription_count.return_value = 0
    with patch('dinov3_cases.node.time.monotonic', side_effect=[20., 21.]):
        CaseNode.run_publications(node)
    assert node.last_publication['callback_to_publish_ms'] == 11000.
    assert node.last_publication['result_to_publish_ms'] == 2000.
    assert node.last_publication['dds_submit_ms'] == 1000.
    assert node.last_publication['frame_id'] == 4
    assert node.pending_publication_timing is None
    assert ('ros_publication' in node.performance) == (current_frame == 4)
