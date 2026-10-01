"""Exercise real DINOv3 code with tiny random weights; not an accuracy test."""
import io
import tempfile
import threading
import time
from pathlib import Path
import numpy as np
import rclpy
from PIL import Image
from transformers import DINOv3ViTConfig, DINOv3ViTModel, DINOv3ViTImageProcessorFast
from dinov3_cases.node import Encoder, CaseNode, case_path, render_heatmap
from dinov3_cases.boxes import BoxConfig, extract_boxes, detection_message
from std_msgs.msg import Header

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
    DINOv3ViTModel(DINOv3ViTConfig(hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_register_tokens=4)).save_pretrained(model)
    DINOv3ViTImageProcessorFast().save_pretrained(model)
    encoder = Encoder(str(model), 224)
    picture = Image.fromarray(np.random.default_rng(1).integers(0, 255, (96, 192, 3), dtype=np.uint8))
    features, valid, geo = encoder.encode(picture)
    assert features.shape == (196, 32)
    assert 0 < valid.sum() < 196
    patch_scores, patch_valid, geometry = encoder.similarity_map(picture, features[valid])
    assert patch_scores.shape == patch_valid.shape == (14, 14)
    assert geometry == geo
    boxes = extract_boxes(patch_scores, patch_valid, geometry, picture.size, 16, .7,
                          BoxConfig(rgb_border_fraction=0))
    assert boxes, 'Self-matching patches must produce a box'
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
    scores = encoder.heatmap(picture, features[valid])
    assert scores.shape == (96, 192) and np.isfinite(scores).all()
    assert scores.min() >= -1.001 and scores.max() <= 1.001
    try:
        case_path(root, '../escape')
        raise AssertionError('Traversal accepted')
    except ValueError:
        pass
    rclpy.init(args=['--ros-args', '-p', f'model_id:={model}', '-p', f'cases_dir:={root / "cases"}',
                     '-p', 'image_size:=224', '-p', 'port:=0', '-p', 'inference_hz:=10.0'])
    node = CaseNode()
    client = node.app.test_client()
    def png():
        data = io.BytesIO()
        picture.save(data, format='PNG')
        data.seek(0)
        return data
    try:
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
        assert client.post('/api/threshold', json={'threshold':.95}).status_code == 200
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
        print('PASS: patch extraction, letterbox geometry, upload groups, selection validation, uploaded target and camera callback overlays')
    finally:
        node.stop.set()
        node.server.shutdown()
        node.preview_worker.join(2)
        node.worker.join(30)
        node.destroy_node()
        rclpy.shutdown()
