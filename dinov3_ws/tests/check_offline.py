"""Exercise real DINOv3 code with tiny random weights; not an accuracy test."""
import io
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import rclpy
from PIL import Image
from transformers import DINOv3ConvNextConfig, DINOv3ConvNextModel, DINOv3ViTImageProcessorFast
from dinov3_cases.node import Encoder, CaseNode, case_path, render_heatmap
from dinov3_cases.boxes import similarity_to_bboxes, detection_message
from dinov3_cases.patches import inference_dtype
from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo

# Below-threshold pixels stay unchanged; equality is included.
bgr = np.full((2, 2, 3), 120, dtype=np.uint8)
values = np.array([[.2, .7], [.9, .1]], dtype=np.float32)
filtered = render_heatmap(bgr, values, .7)
assert np.array_equal(filtered[values < .7], bgr[values < .7])
assert np.any(filtered[values >= .7] != bgr[values >= .7])
assert np.array_equal(render_heatmap(bgr, values, 1), bgr)

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    model = root / 'model'
    DINOv3ConvNextModel(DINOv3ConvNextConfig(
        depths=[1, 1, 1, 1], hidden_sizes=[8, 16, 24, 32])).save_pretrained(model)
    DINOv3ViTImageProcessorFast().save_pretrained(model)
    encoder = Encoder(str(model))
    assert encoder.dtype == inference_dtype(encoder.device)
    picture = Image.fromarray(np.random.default_rng(1).integers(0, 255, (96, 192, 3), dtype=np.uint8))
    features = encoder.encode(picture)
    assert features.shape == (18, 32)
    scores, grid, patches = encoder.similarity_map(picture, features, debug=True)
    assert scores.shape == (96, 192) and np.isfinite(scores).all()
    assert grid == (3, 6) and patches.shape == (3, 6)
    # Corrupt features must never become a partially visible heatmap or a bank.
    for invalid in (float('nan'), float('inf')):
        bad = torch.full((1, 19, 32), invalid, device=encoder.device)
        with patch.object(encoder.model, 'forward', return_value=type('Output', (), {'last_hidden_state': bad})()):
            for operation in (lambda: encoder.similarity_map(picture, features),
                              lambda: encoder.encode(picture)):
                try:
                    operation()
                    raise AssertionError('Non-finite features accepted')
                except RuntimeError as error:
                    assert 'Non-finite DINOv3 features' in str(error)
    # Random weights cannot guarantee a
    # semantic threshold. Use the full cosine range to exercise message geometry.
    result = similarity_to_bboxes(scores, -1., 4, 8)
    boxes = [(item['bbox'], item['score']) for item in result]
    assert boxes and boxes[0][0] == [0, 0, 192, 96]
    header = Header()
    header.frame_id = 'test_camera'
    header.stamp.sec = 123
    detections = detection_message(boxes, 'test_case', header)
    assert detections.header == header
    assert len(detections.detections) == len(boxes)
    for detection, (bounds, score) in zip(detections.detections, boxes):
        assert detection.header == header
        assert detection.results[0].hypothesis.class_id == 'test_case'
        assert detection.results[0].hypothesis.score == score
        assert detection.bbox.size_x == bounds[2] - bounds[0]
        assert detection.bbox.size_y == bounds[3] - bounds[1]
    assert not detection_message([], 'test_case', header).detections
    scores = encoder.heatmap(picture, features)
    assert scores.shape == (96, 192) and np.isfinite(scores).all()
    assert scores.min() >= -1.001 and scores.max() <= 1.001
    try:
        case_path(root, '../escape')
        raise AssertionError('Traversal accepted')
    except ValueError:
        pass
    rclpy.init(args=['--ros-args', '-p', 'inference_backend:=torch', '-p', f'model_id:={model}', '-p', f'cases_dir:={root / "cases"}',
                     '-p', 'debug_visualization:=true', '-p', 'port:=0', '-p', 'inference_hz:=10.0'])
    node = CaseNode()
    client = node.app.test_client()
    def png():
        data = io.BytesIO()
        picture.save(data, format='PNG')
        data.seek(0)
        return data
    try:
        # Exercise highest-score selection and ROS calibration/depth wiring.
        pose_rgb = node.bridge.cv2_to_imgmsg(np.zeros((120, 200, 3), np.uint8), encoding='rgb8')
        pose_rgb.header.frame_id = 'color_optical'
        pose_rgb.header.stamp.sec = 100
        depth_msg = node.bridge.cv2_to_imgmsg(np.ones((120, 200), np.float32), encoding='32FC1')
        depth_msg.header = pose_rgb.header
        info = CameraInfo()
        info.header = pose_rgb.header
        info.width, info.height = 200, 120
        info.k = [400., 0., 100., 0., 400., 60., 0., 0., 1.]
        info.distortion_model = 'plumb_bob'
        info.d = [0.] * 5
        node.on_depth(depth_msg)
        node.on_camera_info(info)
        mask = np.zeros((120, 200), np.uint8)
        mask[40:80, 20:180] = 1
        candidates = [dict(bbox=[0, 0, 10, 10], score=.7),
                      dict(bbox=[20, 40, 180, 80], score=.9)]
        pose, reason = node.selected_pose(candidates, mask, pose_rgb.header, list(node.depth_frames), node.camera_info)
        assert pose is not None, reason
        assert candidates[1]['selected'] and 'selected' not in candidates[0]
        assert pose['frame_id'] == 'color_optical' and pose['position'][2] == 1.
        pose, reason = node.selected_pose(candidates, mask, None, list(node.depth_frames), info)
        assert pose is None and candidates[1]['pose'] is None
        assert client.get('/').status_code == 200
        assert client.post('/api/threshold', json={'threshold':.85}).status_code == 200
        assert client.get('/api/status').json['threshold'] == .85
        for invalid in [-.1, 1.1, 'bad', None, True]:
            assert client.post('/api/threshold', json={'threshold':invalid}).status_code == 400
        msg = node.bridge.cv2_to_imgmsg(np.asarray(picture), encoding='rgb8')
        node.on_image(msg)
        deadline = time.monotonic() + 5
        while node.preview_jpeg is None and time.monotonic() < deadline:
            time.sleep(.05)
        assert node.preview_jpeg is not None, 'Preview must work without a selected case'
        assert client.get('/api/status').json['camera_connected']
        stream = client.get('/camera.mjpg', buffered=False)
        assert b'Content-Type: image/jpeg' in next(iter(stream.response))
        stream.close()
        assert client.post('/api/select', json={'case':'../escape'}).status_code == 400
        assert client.post('/api/cases', data={'case':'test_case', 'images':[(png(),'one.png'),(png(),'two.png')]}).status_code == 200
        assert len(list((root / 'cases/test_case').glob('*.png'))) == 2
        assert client.post('/api/source', data={'source':'upload','image':(png(),'target.png')}).status_code == 200
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and client.get('/overlay.jpg').status_code != 200:
            time.sleep(.1)
        assert client.get('/overlay.jpg').status_code == 200, client.get('/api/status').json
        assert client.get('/api/status').json['debug_visualization']
        for view in ('grid', 'patches', 'heatmap', 'mask', 'boxes'):
            assert client.get(f'/debug/{view}.jpg').status_code == 200
        assert client.get('/debug/invalid.jpg').status_code == 404
        assert 'detections' in client.get('/api/detections').json
        assert client.post('/api/threshold', json={'threshold':.95}).status_code == 200
        assert client.get('/debug/grid.jpg').status_code == 204
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and client.get('/overlay.jpg').status_code != 200:
            time.sleep(.1)
        assert client.get('/overlay.jpg').status_code == 200, 'Static target must refresh after threshold changes'
        assert client.post('/api/source', data={'source':'camera'}).status_code == 200
        assert client.get('/overlay.jpg').status_code == 204
        msg = node.bridge.cv2_to_imgmsg(np.asarray(picture), encoding='rgb8')
        msg.header.frame_id = 'test_camera'
        node.on_image(msg)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and client.get('/overlay.jpg').status_code != 200:
            time.sleep(.1)
        assert client.get('/overlay.jpg').status_code == 200, client.get('/api/status').json
        print('PASS: native references, single-image patch matching, global boxes, debug views, uploads, threshold updates and camera overlays')
    finally:
        node.stop.set()
        node.server.shutdown()
        node.preview_worker.join(2)
        node.worker.join(30)
        node.destroy_node()
        rclpy.shutdown()
