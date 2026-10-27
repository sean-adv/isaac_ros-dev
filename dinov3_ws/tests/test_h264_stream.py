"""Shared encoding, decoder-safe fan-out, rendering selection, and real WebRTC delivery."""
import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import av
import numpy as np
import pytest
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from cv_bridge import CvBridge
from std_msgs.msg import Header

from dinov3_cases.h264_stream import AccessUnit, H264Encoder, PacketHub, WebRTCVideo
from dinov3_cases.node import CaseNode
from dinov3_cases.visualization import render_display_overlay
from dinov3_cases.web import create_app


def test_encoder_real_decode_and_forced_idr_for_new_reader():
    encoder = H264Encoder('libx264')
    header = Header(frame_id='color_optical')
    header.stamp.sec, header.stamp.nanosec = 123, 456
    pixels = np.full((540, 960, 3), [30, 80, 200], np.uint8)
    units = [encoder.encode(pixels, header, 4, force_keyframe=i == 2) for i in range(3)]
    assert [unit.keyframe for unit in units] == [True, False, True]
    assert units[0].pts < units[1].pts < units[2].pts
    decoder = av.CodecContext.create('h264', 'r')
    for unit in units:
        decoded = decoder.decode(av.Packet(unit.data))
        assert len(decoded) == 1
        assert decoded[0].to_ndarray(format='bgr24').shape == (540, 960, 3)
        assert unit.header is header and unit.generation == 4
    # Every IDR includes SPS/PPS; a newly joined decoder needs no earlier packets.
    fresh = av.CodecContext.create('h264', 'r')
    assert len(fresh.decode(av.Packet(units[2].data))) == 1


@pytest.mark.parametrize('codec,fps,bitrate', [('bad', 30, 1), ('auto', 0, 1),
                                              ('auto', True, 1), ('auto', 31, 1),
                                              ('auto', 30, 0)])
def test_encoder_rejects_invalid_settings(codec, fps, bitrate):
    with pytest.raises(ValueError):
        H264Encoder(codec, fps, bitrate)


def test_auto_falls_back_when_hardware_cannot_encode():
    encoder = H264Encoder('auto')
    original = encoder._open
    def open_codec(codec):
        if codec == 'h264_nvenc':
            raise RuntimeError('no hardware encoder')
        return original(codec)
    with patch.object(encoder, '_open', side_effect=open_codec):
        assert encoder.encode(np.zeros((540, 960, 3), np.uint8), Header(), 0).keyframe
    assert encoder.codec == 'libx264'


def test_slow_readers_drop_entire_gop_and_wait_for_idr():
    hub = PacketHub(capacity=2)
    reader = hub.subscribe()
    def unit(i, key=False):
        return AccessUnit(bytes([i]), i, key, None, 0)
    hub.publish(unit(0))
    assert reader.get(timeout=0) is None
    hub.publish(unit(1, True))
    hub.publish(unit(2))
    hub.keyframe_requested.clear()
    hub.publish(unit(3))  # overflow cannot retain dependent P frames
    assert reader.get(timeout=0) is None
    assert hub.keyframe_requested.is_set()
    hub.publish(unit(4))
    assert reader.get(timeout=0) is None
    key = unit(5, True)
    hub.publish(key)
    assert reader.get(timeout=0) is key
    hub.reset()
    hub.publish(unit(6))
    assert reader.get(timeout=0) is None
    reader.close()
    assert not hub.has_readers


def render_node(audience=True, boxes=True, source='camera'):
    bridge = CvBridge()
    rgb = np.full((120, 160, 3), [200, 80, 30], np.uint8)
    camera = bridge.cv2_to_imgmsg(rgb, encoding='rgb8')
    camera.header.frame_id = 'color_optical'
    detections = [dict(bbox_visible=True, bbox=[20, 20, 90, 90], label='known',
                       detection_label='box')] if boxes else []
    frame = Mock(pixels=rgb)
    hub = PacketHub()
    readers = [hub.subscribe() for _ in range(int(audience))]
    stop = Mock()
    stop.is_set.side_effect = [False, True]
    node = SimpleNamespace(
        stop=stop, lock=threading.Lock(), generation=0, source=source,
        render_changed=threading.Event(), camera_frame=camera, camera_seq=2,
        pending_render=(frame, detections, camera.header, 0, 2, None, time.monotonic()),
        pending_annotated_image=None, annotated_image_changed=threading.Event(),
        annotated_image_pub=Mock(get_subscription_count=Mock(return_value=0)),
        bbox_visualization_mode='2d', video_hub=hub, video_encoder=H264Encoder('libx264'),
        param=lambda _: 30, bridge=bridge, performance={}, get_logger=lambda: Mock(),
    )
    node.render_changed.set()
    return node, frame, readers


@pytest.mark.parametrize('audience', [0, 1, 2])
@pytest.mark.parametrize('boxes', [False, True])
def test_draw_and_encode_once_for_all_consumers_and_preserve_header(audience, boxes):
    node, frame, readers = render_node(audience, boxes)
    with patch('dinov3_cases.node.render_display_overlay', wraps=render_display_overlay) as draw, \
            patch.object(node.video_encoder, 'encode', wraps=node.video_encoder.encode) as encode:
        CaseNode.run_rendering(node)
    assert draw.call_count == int(audience > 0)
    assert encode.call_count == int(audience > 0)
    frame.release.assert_called_once()
    if readers:
        units = [reader.get(timeout=0) for reader in readers]
        assert all(unit is units[0] for unit in units)
        assert units[0].header == node.camera_frame.header
        decoded = av.CodecContext.create('h264', 'r').decode(av.Packet(units[0].data))[0]
        assert (decoded.width, decoded.height) == (960, 540)
        assert node.performance['video_encoder'] == 'libx264'
    node.get_logger().error.assert_not_called()


@pytest.mark.parametrize('audience', [0, 1])
def test_rviz_raw_image_without_requiring_encoded_viewers(audience):
    node, frame, readers = render_node(audience)
    node.annotated_image_pub.get_subscription_count.return_value = 1
    with patch.object(node.video_encoder, 'encode', wraps=node.video_encoder.encode) as encode:
        CaseNode.run_rendering(node)
    node.annotated_image_pub.publish.assert_not_called()
    node.stop.is_set.side_effect = [False, True]
    CaseNode.run_image_publications(node)
    node.annotated_image_pub.publish.assert_called_once()
    message = node.annotated_image_pub.publish.call_args.args[0]
    assert message.encoding == 'bgr8'
    assert (message.width, message.height) == (960, 540)
    assert message.header == node.camera_frame.header
    assert encode.call_count == int(audience > 0)
    frame.release.assert_called_once()
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


def test_old_generation_render_is_released_and_not_sent():
    node, frame, readers = render_node()
    node.generation = 1
    node.camera_frame = None
    CaseNode.run_rendering(node)
    assert readers[0].get(timeout=0) is None
    frame.release.assert_called_once()


def test_ros_uses_same_encoded_bytes_and_source_header():
    hub, stop, received = PacketHub(), threading.Event(), threading.Event()
    pub = Mock()
    pub.get_subscription_count.return_value = 1
    messages = []
    def publish(message):
        messages.append(message)
        received.set()
        stop.set()
    pub.publish.side_effect = publish
    node = SimpleNamespace(stop=stop, video_hub=hub, h264_pub=pub,
                           lock=threading.Lock(), generation=4, get_logger=lambda: Mock())
    thread = threading.Thread(target=CaseNode.run_video_publications, args=(node,))
    thread.start()
    try:
        # Wait for DDS reader registration rather than racing the first IDR.
        import time
        deadline = time.monotonic() + 2
        while not hub.has_readers and time.monotonic() < deadline:
            stop.wait(.01)
        assert hub.has_readers
        unit = H264Encoder('libx264').encode(np.zeros((540, 960, 3), np.uint8),
                                           Header(frame_id='camera'), 4)
        hub.publish(unit)
        assert received.wait(3)
        assert bytes(messages[0].data) == unit.data
        assert messages[0].header == unit.header
        assert messages[0].format == 'h264'
    finally:
        stop.set()
        hub.close()
        thread.join(3)
    assert not thread.is_alive()


def test_http_signaling_and_no_live_jpeg_routes(tmp_path):
    node = SimpleNamespace(root=tmp_path, bbox_visualization_mode='3d',
                           image_changed=Mock(), render_changed=Mock(), webrtc=Mock())
    node.webrtc.offer.return_value = dict(sdp='answer', type='answer', peer_id='peer')
    client = create_app(node).test_client()
    assert client.post('/api/video/offer', json={'sdp': 'offer', 'type': 'offer'}).json['sdp'] == 'answer'
    node.webrtc.offer.assert_called_once_with('offer', 'offer')
    assert client.post('/api/video/offer', json={}).status_code == 400
    node.webrtc.heartbeat.return_value = True
    assert client.post('/api/video/peers/peer/heartbeat').status_code == 204
    node.webrtc.heartbeat.return_value = False
    assert client.post('/api/video/peers/peer/heartbeat').status_code == 404
    assert client.delete('/api/video/peers/peer').status_code == 204
    node.webrtc.disconnect.assert_called_once_with('peer')
    for path in ['/view.mjpg', '/camera.mjpg', '/overlay.mjpg', '/overlay.jpg']:
        assert client.get(path).status_code == 404
    for path in ['/', '/images']:
        html = client.get(path).get_data(as_text=True)
        assert '<video ' in html and 'data-live-video' in html
        assert '/static/video.js' in html and '.mjpg' not in html


def test_two_real_webrtc_receivers_decode_without_reencoding():
    async def run():
        hub = PacketHub()
        server = WebRTCVideo(hub)
        clients, tracks = [], []
        try:
            for _ in range(2):
                client = RTCPeerConnection(RTCConfiguration(iceServers=[]))
                clients.append(client)
                track_ready = asyncio.get_running_loop().create_future()
                @client.on('track')
                def on_track(track, ready=track_ready):
                    ready.set_result(track)
                client.addTransceiver('video', direction='recvonly')
                await client.setLocalDescription(await client.createOffer())
                answer = await asyncio.to_thread(server.offer, client.localDescription.sdp, 'offer')
                assert 'VP8/90000' not in answer['sdp']
                assert 'H264/90000' in answer['sdp']
                await client.setRemoteDescription(RTCSessionDescription(answer['sdp'], answer['type']))
                tracks.append(await asyncio.wait_for(track_ready, 5))
            encoder = H264Encoder('libx264')
            stop = asyncio.Event()
            async def produce():
                index = 0
                while not stop.is_set():
                    unit = encoder.encode(np.full((540, 960, 3), index % 255, np.uint8),
                                          Header(), 0, force_keyframe=index % 10 == 0)
                    hub.publish(unit)
                    index += 1
                    await asyncio.sleep(1 / 30)
            with patch('aiortc.codecs.h264.H264Encoder.encode', side_effect=AssertionError('second encode')):
                task = asyncio.create_task(produce())
                try:
                    frames = await asyncio.wait_for(asyncio.gather(*(t.recv() for t in tracks)), 10)
                    assert all((f.width, f.height) == (960, 540) for f in frames)
                finally:
                    stop.set()
                    await task
            assert len(server.peers) == 2
        finally:
            await asyncio.gather(*(client.close() for client in clients))
            await asyncio.to_thread(server.close)
            hub.close()
        assert not server.thread.is_alive()
    asyncio.run(run())


@pytest.mark.parametrize('source', ['camera', 'upload'])
def test_stale_results_fall_back_to_camera_but_uploads_remain(source):
    node, frame, readers = render_node(source=source)
    job = list(node.pending_render)
    job[-1] = time.monotonic() - 10
    node.pending_render = tuple(job)
    with patch('dinov3_cases.node.render_display_overlay', wraps=render_display_overlay) as draw:
        CaseNode.run_rendering(node)
    assert draw.call_count == 1
    assert bool(draw.call_args.args[1]) == (source == 'upload')
    assert readers[0].get(timeout=0) is not None
    frame.release.assert_called_once()


def test_blocked_ros_video_reader_does_not_block_other_readers_and_recovers_at_idr():
    hub = PacketHub(capacity=2)
    viewer = hub.subscribe()
    stop, entered, release, recovered = (threading.Event() for _ in range(4))
    captured = []
    pub = Mock()
    pub.get_subscription_count.return_value = 1
    def publish(message):
        captured.append(bytes(message.data)[0])
        if len(captured) == 1:
            entered.set()
            assert release.wait(3)
        else:
            recovered.set()
    pub.publish.side_effect = publish
    node = SimpleNamespace(stop=stop, video_hub=hub, h264_pub=pub,
                           lock=threading.Lock(), generation=0, get_logger=lambda: Mock())
    worker = threading.Thread(target=CaseNode.run_video_publications, args=(node,))
    worker.start()
    try:
        deadline = time.monotonic() + 2
        while len(hub.readers) != 2 and time.monotonic() < deadline:
            stop.wait(.01)
        assert len(hub.readers) == 2
        first = AccessUnit(b'\x01', 1, True, Header(), 0)
        hub.publish(first)
        assert viewer.get(timeout=0) is first
        assert entered.wait(2)
        for index in range(2, 7):
            packet = AccessUnit(bytes([index]), index, False, Header(), 0)
            hub.publish(packet)
            assert viewer.get(timeout=0) is packet
        assert node.lock.acquire(timeout=1)
        node.lock.release()
        release.set()
        final = AccessUnit(b'\x07', 7, True, Header(), 0)
        hub.publish(final)
        assert recovered.wait(2)
        assert captured == [1, 7]
    finally:
        release.set()
        stop.set()
        hub.close()
        worker.join(3)
    assert not worker.is_alive()
