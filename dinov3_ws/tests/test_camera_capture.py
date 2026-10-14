"""Manual capture uses the chosen prompt and crops the exact raw camera frame."""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from dinov3_cases.web import create_app


@pytest.fixture
def camera(tmp_path):
    pixels = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    calls = []
    proposals = [dict(bbox=[2, 3, 9, 10], detection_label='box', detection_score=0.9),
                 dict(bbox=[10, 0, 16, 5], detection_label='box', detection_score=0.8)]

    def validate(prompt):
        if prompt != 'Detect box.':
            raise ValueError('Invalid prompt')

    def detect(image, *args, **kwargs):
        calls.append((np.array(image), args, kwargs))
        return proposals

    node = SimpleNamespace(
        root=tmp_path, lock=threading.Lock(), image_changed=threading.Event(),
        stream_changed=threading.Condition(), camera_frame=object(),
        camera_received_at=time.monotonic(), generation=0,
        grounding_settings={'prompt': 'Detect bottle.', 'box_threshold': 0.3},
        grounding_detector=SimpleNamespace(validate_prompt=validate, detect=detect),
        bridge=SimpleNamespace(imgmsg_to_cv2=lambda frame, desired_encoding: pixels),
        param=lambda name: {'grounding_nms_iou': 0.5, 'grounding_max_detections': 20,
                            'grounding_min_bbox_width': 0, 'grounding_min_bbox_height': 0,
                            'grounding_min_bbox_area': 0}[name],
        clear_object_debug=lambda: None,
    )
    app = create_app(node)
    app.config['TESTING'] = True
    return app.test_client(), node, calls, proposals, pixels


def test_capture_creates_exact_crops_and_refreshes_gallery(camera):
    client, node, calls, _, pixels = camera
    assert b'camera-section' in client.get('/images').data
    assert client.post('/api/images/labels', json={'case': 'medicine'}).status_code == 200
    response = client.post('/api/images/capture', json={'case': 'medicine', 'prompt': 'Detect box.'})
    assert response.status_code == 200
    names = response.json['names']
    assert len(names) == 2
    np.testing.assert_array_equal(np.array(Image.open(node.root / 'medicine' / names[0])), pixels[3:10, 2:9])
    np.testing.assert_array_equal(np.array(Image.open(node.root / 'medicine' / names[1])), pixels[0:5, 10:16])
    np.testing.assert_array_equal(calls[0][0], pixels)
    assert calls[0][1] == ('Detect box.', 0.3, 0.5, 20)
    assert node.grounding_settings['prompt'] == 'Detect bottle.'
    assert node.generation == 1
    assert len(client.get('/api/images?case=medicine').json['images']) == 2


def test_no_detections_saves_nothing(camera):
    client, node, _, proposals, _ = camera
    proposals.clear()
    client.post('/api/images/labels', json={'case': 'empty'})
    response = client.post('/api/images/capture', json={'case': 'empty', 'prompt': 'Detect box.'})
    assert response.status_code == 200
    assert response.json['names'] == []
    assert list((node.root / 'empty').iterdir()) == []
    assert node.generation == 0


@pytest.mark.parametrize('data', [None, [], {}, {'case': '../escape'}, {'case': 'missing', 'prompt': 'Detect box.'},
                                       {'case': 'valid'}, {'case': 'valid', 'prompt': 5},
                                       {'case': 'valid', 'prompt': 'unknown'}])
def test_invalid_capture_rejected_without_inference(camera, data):
    client, node, calls, _, _ = camera
    (node.root / 'valid').mkdir()
    assert client.post('/api/images/capture', json=data).status_code == 400
    assert calls == []
    assert list((node.root / 'valid').iterdir()) == []


@pytest.mark.parametrize('attribute,value', [('camera_frame', None), ('camera_received_at', None),
                                            ('camera_received_at', 0), ('grounding_detector', None)])
def test_unavailable_camera_or_detector(camera, attribute, value):
    client, node, calls, _, _ = camera
    (node.root / 'valid').mkdir()
    setattr(node, attribute, value)
    response = client.post('/api/images/capture', json={'case': 'valid', 'prompt': 'Detect box.'})
    assert response.status_code == 503
    assert response.json['error']
    assert calls == []


def test_label_validation_and_standalone_mode(tmp_path):
    from dinov3_cases.image_manager import create_app as standalone
    client = standalone(tmp_path).test_client()
    for label in ['', '../escape', 'a/b']:
        assert client.post('/api/images/labels', json={'case': label}).status_code == 400
    assert client.post('/api/images/labels', json={'case': 'valid'}).status_code == 200
    assert client.post('/api/images/labels', json={'case': 'valid'}).status_code == 200
    assert client.get('/api/images').json['cases'] == ['valid']
    assert b'camera-section' not in client.get('/images').data
    assert client.post('/api/images/capture', json={'case': 'valid', 'prompt': 'Detect box.'}).status_code == 503
