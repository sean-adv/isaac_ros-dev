"""Shared ROS Image rendering, bounded JPEG delivery and HTTP streaming regressions."""
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest
from cv_bridge import CvBridge
from std_msgs.msg import Header

from dinov3_cases.api.image_stream import ImageStream
from dinov3_cases.ros.node import CaseNode
from dinov3_cases.api.visualization import render_display_overlay
from dinov3_cases.api.web import create_app


def render_node(audience=True, boxes=True, source='camera'):
    bridge = CvBridge()
    rgb = np.full((120, 160, 3), [200, 80, 30], np.uint8)
    camera = bridge.cv2_to_imgmsg(rgb, encoding='rgb8')
    camera.header.frame_id = 'color_optical'
    detections = [dict(bbox_visible=True, bbox=[20, 20, 90, 90], label='known',
                       detection_label='box')] if boxes else []
    frame = Mock(pixels=rgb)
    stop = Mock()
    stop.is_set.side_effect = [False, True]
    node = SimpleNamespace(
        stop=stop, lock=threading.Lock(), generation=0, source=source,
        render_changed=threading.Event(), camera_frame=camera, camera_seq=2,
        pending_render=(frame, detections, camera.header, 0, 2, None, time.monotonic()),
        pending_annotated_image=None, annotated_image_changed=threading.Event(),
        annotated_image_pub=Mock(get_subscription_count=Mock(return_value=1)),
        bbox_visualization_mode='2d',
        image_stream=SimpleNamespace(has_readers=bool(audience), needs_frame=False),
        param=lambda _: 30, bridge=bridge, performance={}, get_logger=lambda: Mock(),
    )
    node.render_changed.set()
    return node, frame, None


@pytest.mark.parametrize('web,external', [(False,False),(True,False),(False,True),(True,True)])
@pytest.mark.parametrize('boxes', [False,True])
def test_shared_ros_image_is_drawn_once_with_header_and_only_on_demand(web,external,boxes):
    node,frame,_=render_node(audience=web,boxes=boxes)
    node.annotated_image_pub.get_subscription_count.return_value=1+int(external)
    with patch('dinov3_cases.ros.node.render_display_overlay',wraps=render_display_overlay) as draw:
        CaseNode.run_rendering(node)
    assert draw.call_count==int(web or external)
    frame.release.assert_called_once()
    node.stop.is_set.side_effect=[False,True]
    CaseNode.run_image_publications(node)
    assert node.annotated_image_pub.publish.call_count==int(web or external)
    if web or external:
        message=node.annotated_image_pub.publish.call_args.args[0]
        assert message.encoding=='bgr8'
        assert (message.width,message.height)==(960,540)
        assert message.header==node.camera_frame.header
    node.get_logger().error.assert_not_called()


def test_slow_rviz_does_not_hold_state_lock_and_next_publication_uses_latest_frame():
    node, _, _ = render_node(audience=False)
    node.stop = threading.Event()
    node.annotated_image_pub.get_subscription_count.return_value = 1
    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    received = []

    def job(seq):
        header = Header()
        header.stamp.sec = seq
        return (np.zeros((2, 2, 3), np.uint8), header, 0)

    def publish(message):
        received.append(message.header.stamp.sec)
        if len(received) == 1:
            entered.set()
            assert release.wait(3)
        else:
            node.stop.set()
            done.set()

    node.annotated_image_pub.publish.side_effect = publish
    node.pending_annotated_image = job(1)
    node.annotated_image_changed.set()
    worker = threading.Thread(target=CaseNode.run_image_publications, args=(node,))
    worker.start()
    try:
        assert entered.wait(3)
        assert node.lock.acquire(timeout=1), "DDS publication must not hold the state lock"
        try:
            node.pending_annotated_image = job(2)
            node.pending_annotated_image = job(3)
            node.annotated_image_changed.set()
        finally:
            node.lock.release()
        release.set()
        assert done.wait(3)
        assert received == [1, 3]
    finally:
        release.set()
        node.stop.set()
        node.annotated_image_changed.set()
        worker.join(3)
    assert not worker.is_alive()


@pytest.mark.parametrize('generation,readers', [(1, 1), (0, 0)])
def test_raw_image_worker_discards_old_source_and_unsubscribed_frames(generation, readers):
    node, _, _ = render_node(audience=False)
    node.generation = generation
    node.annotated_image_pub.get_subscription_count.return_value = readers
    node.pending_annotated_image = (np.zeros((2, 2, 3), np.uint8), Header(), 0)
    node.annotated_image_changed.set()
    CaseNode.run_image_publications(node)
    node.annotated_image_pub.publish.assert_not_called()
    assert node.pending_annotated_image is None


@pytest.mark.parametrize('pending_dino,name', [(True, 'unknown'), (False, 'unknown'), (False, 'known')])
@pytest.mark.parametrize('depth_ready', [True, False])
def test_3d_renderer_waits_for_depth_and_never_draws_2d_fallback(pending_dino, name, depth_ready):
    node, frame, _ = render_node()
    node.bbox_visualization_mode = '3d'
    item = node.pending_render[1][0]
    item.update(label=name,
                recognition_pending=pending_dino,
                bbox3d_status='pending_background_measurement')
    if depth_ready:
        item['bbox3d'] = dict(position=[0., 0., 1.], size=[.1, .1, .1])
        job = list(node.pending_render)
        job[5] = SimpleNamespace(k=[100., 0., 80., 0., 100., 60., 0., 0., 1.],
                                 d=[], distortion_model='plumb_bob')
        node.pending_render = tuple(job)
    with patch('dinov3_cases.api.visualization.cv2.rectangle', wraps=cv2.rectangle) as rectangle, \
            patch('dinov3_cases.api.visualization.cv2.putText', wraps=cv2.putText) as text:
        CaseNode.run_rendering(node)
    rectangle.assert_not_called()
    if depth_ready:
        assert text.call_args.args[1] == ('box' if name == 'unknown' else 'known | box')
    else:
        text.assert_not_called()
    assert node.pending_annotated_image is not None
    frame.release.assert_called_once()


@pytest.mark.parametrize('work_ms,expected_wait_ms', [(10,1000/30-10),(50,0)])
def test_renderer_paces_frame_starts_without_delay_after_overrun(work_ms,expected_wait_ms):
    node,frame,_=render_node(boxes=False)
    node.stop.is_set.side_effect=[False,False,True]
    clock=[100.];waits=[]
    node.render_changed=Mock()
    node.render_changed.wait.side_effect=lambda _: setattr(node,'camera_seq',node.camera_seq+1)
    def wait(delay):
        waits.append(delay*1000);clock[0]+=delay;return False
    def draw(*args,**kwargs):
        clock[0]+=work_ms/1000
        return np.zeros((540,960,3),np.uint8)
    node.stop.wait.side_effect=wait
    with patch('dinov3_cases.ros.node.time.monotonic',side_effect=lambda:clock[0]), \
            patch('dinov3_cases.ros.node.render_display_overlay',side_effect=draw):
        CaseNode.run_rendering(node)
    assert waits==pytest.approx([expected_wait_ms] if expected_wait_ms else [])
    frame.release.assert_called_once()


def test_old_generation_render_is_released_and_not_published():
    node,frame,_=render_node()
    node.generation=1;node.camera_frame=None
    CaseNode.run_rendering(node)
    assert node.pending_annotated_image is None
    frame.release.assert_called_once()


def test_new_web_reader_gets_cached_static_image_without_new_input():
    node,_,_=render_node(source='upload')
    node.stop.is_set.side_effect=[False,False,True]
    calls=[]
    node.image_stream.needs_frame=True
    # Both passes queue the static result until a JPEG has reached the web bridge.
    with patch.object(node.annotated_image_changed,'set',side_effect=lambda:calls.append(True)):
        CaseNode.run_rendering(node)
    assert len(calls)==2


def wait_for(predicate):
    deadline=time.monotonic()+3
    while not predicate():
        assert time.monotonic()<deadline
        time.sleep(.005)


def test_multiple_http_readers_share_jpeg_and_receive_latest_without_queues(tmp_path):
    stream=ImageStream(CvBridge())
    clients=[]
    try:
        client=create_app(SimpleNamespace(root=tmp_path,image_stream=stream,image_changed=Mock(),render_changed=Mock())).test_client()
        # A MIME keepalive is emitted before the first camera frame arrives.
        clients=[client.get('/image_annotated.mjpg',buffered=False) for _ in range(2)]
        assert all(r.status_code==200 and r.headers['Cache-Control']=='no-store' for r in clients)
        assert stream.readers==2
        for response in clients:
            assert next(response.response)==b'\r\n'
        with patch('dinov3_cases.api.image_stream.cv2.imencode',wraps=cv2.imencode) as encode:
            message=CvBridge().cv2_to_imgmsg(np.full((540,960,3),[30,80,200],np.uint8),encoding='bgr8')
            stream.update(message)
            wait_for(lambda:stream.jpeg is not None)
            chunks=[next(r.response) for r in clients]
            assert chunks[0]==chunks[1]
            jpeg=chunks[0].split(b'\r\n\r\n',1)[1][:-2]
            decoded=cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
            assert decoded.shape==(540,960,3)
            np.testing.assert_allclose(decoded[100,100],[30,80,200],atol=3)
            assert encode.call_count==1
            # Leave both HTTP readers idle while a newer frame replaces the last one.
            stream.update(CvBridge().cv2_to_imgmsg(np.zeros((540,960,3),np.uint8),encoding='bgr8'))
            wait_for(lambda:stream.sequence==2)
            assert next(clients[0].response)==next(clients[1].response)
        assert client.post('/api/video/offer',json={}).status_code==404
    finally:
        for response in clients:response.close()
        stream.close()
    assert stream.readers==0 and not stream.worker.is_alive()


def test_slow_jpeg_encode_does_not_block_input_and_reset_discards_inflight_frame():
    entered,release=threading.Event(),threading.Event()
    bridge=Mock()
    def convert(message,desired_encoding):
        entered.set();assert release.wait(3)
        return np.zeros((2,2,3),np.uint8)
    bridge.imgmsg_to_cv2.side_effect=convert
    stream=ImageStream(bridge)
    reader=stream.multipart()
    try:
        assert next(reader)==b'\r\n'
        stream.update(object());assert entered.wait(3)
        # update/reset can proceed while conversion is blocked; only the latest remains.
        stream.update(object());stream.update(object())
        stream.reset()
        release.set()
        wait_for(lambda:bridge.imgmsg_to_cv2.call_count==1)
        time.sleep(.05)
        assert stream.jpeg is None and stream.pending is None
    finally:
        release.set();reader.close();stream.close()


def test_no_viewer_does_not_convert_or_encode_ros_images():
    bridge=Mock();stream=ImageStream(bridge)
    try:
        stream.update(object())
        assert stream.pending is None
        bridge.imgmsg_to_cv2.assert_not_called()
    finally:stream.close()


def test_both_pages_use_the_image_stream_and_roi_uses_image_dimensions():
    from pathlib import Path
    root=Path(__file__).parents[1]/'src/dinov3_cases/dinov3_cases/api'
    for name in ('cases.html','images.html'):
        text=(root/'templates'/name).read_text()
        assert '<img ' in text and 'src="/image_annotated.mjpg"' in text
        assert '<video ' not in text
    text=(root/'static/sam3_roi.js').read_text()
    assert 'naturalWidth' in text and 'naturalHeight' in text


def test_real_ros_image_topic_to_http_jpeg_preserves_raw_pixels_and_header():
    import rclpy
    from rclpy.context import Context
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Image as ImageMsg

    context=Context();rclpy.init(context=context)
    node=Node('annotated_image_smoke',context=context)
    executor=SingleThreadedExecutor(context=context);executor.add_node(node)
    stream=ImageStream(CvBridge())
    reader=stream.multipart();received=[]
    qos=QoSProfile(history=HistoryPolicy.KEEP_LAST,depth=1,reliability=ReliabilityPolicy.RELIABLE)
    publisher=node.create_publisher(ImageMsg,'/medicine_recognition/image_annotated',qos)
    def receive(message):
        received.append(message);stream.update(message)
    subscription=node.create_subscription(ImageMsg,'/medicine_recognition/image_annotated',receive,qos)
    try:
        assert next(reader)==b'\r\n'
        pixels=np.full((540,960,3),[30,80,200],np.uint8)
        message=CvBridge().cv2_to_imgmsg(pixels,encoding='bgr8')
        message.header=Header(frame_id='rgb_optical');message.header.stamp.sec=123
        deadline=time.monotonic()+5
        while not received or stream.jpeg is None:
            assert time.monotonic()<deadline
            publisher.publish(message)
            executor.spin_once(timeout_sec=.02)
        assert received[-1].header==message.header
        np.testing.assert_array_equal(CvBridge().imgmsg_to_cv2(received[-1]),pixels)
        chunk=next(reader)
        decoded=cv2.imdecode(np.frombuffer(chunk.split(b'\r\n\r\n',1)[1][:-2],np.uint8),cv2.IMREAD_COLOR)
        assert decoded.shape==pixels.shape
        np.testing.assert_allclose(decoded[100,100],pixels[100,100],atol=3)
    finally:
        reader.close();stream.close();executor.shutdown();node.destroy_node();context.shutdown()
