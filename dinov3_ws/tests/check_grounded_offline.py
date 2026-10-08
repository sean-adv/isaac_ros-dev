"""ROS/HTTP two-stage integration with tiny DINOv3 and deterministic proposals."""
import io
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import rclpy
from transformers import DINOv3ConvNextConfig, DINOv3ConvNextModel, DINOv3ViTImageProcessorFast

from dinov3_cases.node import CaseNode


class Detector:
    def __init__(self, *args):
        pass

    def detect(self, image, prompt, *args):
        if prompt == 'nothing.':
            return []
        return [dict(bbox=[0, 0, image.width, image.height], detection_label='box',
                     detection_score=.9, area=image.width*image.height)]


def png(image):
    data = io.BytesIO()
    image.save(data, format='PNG')
    data.seek(0)
    return data


def wait_result(client):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if client.get('/overlay.jpg').status_code == 200:
            return client.get('/api/detections').json['detections']
        time.sleep(.1)
    raise AssertionError(client.get('/api/status').json)


def check_views(client, expected_views):
    manifest = client.get('/api/object_visualizations').json
    assert len(manifest['objects']) == 1
    item = manifest['objects'][0]
    assert set(item['views']) == expected_views
    assert item['bbox'] == [0, 0, 96, 64]
    for view in expected_views:
        response = client.get(f'/debug/objects/{manifest["revision"]}/0/{view}.jpg')
        assert response.status_code == 200
        image = Image.open(io.BytesIO(response.data))
        if view in ('crop', 'heatmap', 'similarity'):
            assert image.size == (96, 64)
    assert client.get(f'/debug/objects/{manifest["revision"]}/0/mask.jpg').status_code == 404
    return manifest['revision']


with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    model = root / 'model'
    DINOv3ConvNextModel(DINOv3ConvNextConfig(
        depths=[1, 1, 1, 1], hidden_sizes=[8, 16, 24, 32])).save_pretrained(model)
    DINOv3ViTImageProcessorFast().save_pretrained(model)
    picture = Image.fromarray(np.random.default_rng(9).integers(0, 255, (64, 96, 3), dtype=np.uint8))
    rclpy.init(args=['--ros-args', '-p', 'pipeline:=grounded', '-p', 'inference_backend:=torch',
                     '-p', f'model_id:={model}', '-p', f'cases_dir:={root / "cases"}',
                     '-p', 'port:=0', '-p', 'object_max_side:=64'])
    with patch('dinov3_cases.node.GroundingDetector', Detector):
        node = CaseNode()
        client = node.app.test_client()
        try:
            assert client.get('/api/status').json['pipeline'] == 'grounded'
            assert client.post('/api/source', data={'source': 'upload', 'image': (png(picture), 'target.png')}).status_code == 200
            result = wait_result(client)
            assert result[0]['reason'] == 'empty_gallery' and result[0]['score'] is None
            old_revision = check_views(client, {'crop', 'grid'})
            assert client.post('/api/cases', data={'case': 'a', 'images': (png(picture), 'one.png')}).status_code == 200
            assert client.get(f'/debug/objects/{old_revision}/0/crop.jpg').status_code == 410
            result = wait_result(client)
            assert result[0]['label'] == 'a' and result[0]['score'] > .99
            assert result[0]['bbox'] == [0, 0, 96, 64] and result[0]['pose'] is None
            check_views(client, {'crop', 'grid', 'heatmap', 'similarity', 'patches', 'reference'})
            revision = client.get('/api/object_visualizations').json['revision']
            assert client.post('/api/select', json={'case': 'a'}).status_code == 200
            assert client.get('/api/object_visualizations').json['revision'] == revision
            assert client.get('/overlay.jpg').status_code == 200
            assert client.post('/api/cases', data={'case': 'b', 'images': (png(picture), 'one.png')}).status_code == 200
            result = wait_result(client)
            assert result[0]['label'] == 'unknown' and result[0]['reason'] == 'ambiguous'
            check_views(client, {'crop', 'grid', 'heatmap', 'similarity', 'patches', 'reference'})
            original = client.get('/api/detections').json
            before = client.get('/api/object_visualizations').json
            selected_views = client.get('/api/object_visualizations?reference_case=b')
            assert selected_views.status_code == 200
            selected_manifest = selected_views.json
            assert selected_manifest['reference_cases'] == ['a', 'b']
            assert selected_manifest['reference_case'] == 'b'
            item = selected_manifest['objects'][0]
            assert item['comparison_case'] == item['comparison_reference']['case'] == 'b'
            assert item['comparison_reference_count'] == 1
            assert selected_manifest['revision'] == before['revision']
            assert client.get('/api/detections').json == original
            assert client.get(f'/debug/objects/{before["revision"]}/0/heatmap.jpg?reference_case=b').status_code == 200
            invalid = client.get('/api/object_visualizations?reference_case=missing')
            assert invalid.status_code == 400 and invalid.json['reference_cases'] == ['a', 'b']
            assert client.get(f'/debug/objects/{before["revision"]}/0/heatmap.jpg?reference_case=missing').status_code == 400
            assert client.post('/api/grounding', json={'margin': 0.}).status_code == 200
            assert wait_result(client)[0]['label'] in ('a', 'b')
            assert client.post('/api/grounding', json={'prompt': 'nothing'}).status_code == 200
            assert wait_result(client) == []
            assert client.get('/api/object_visualizations').json['objects'] == []
            for payload in [{}, [], {'extra': 2}, {'prompt': ''}, {'margin': True}, {'box_threshold': 1.1}]:
                assert client.post('/api/grounding', json=payload).status_code == 400
            assert client.post('/api/grounding', json={'prompt': 'box', 'margin': .05}).status_code == 200
            assert wait_result(client)[0]['reason'] == 'ambiguous'
            # Delete an unselected class through image manager and refresh static input.
            selected = client.get('/api/status').json['selected']
            other = 'a' if selected == 'b' else 'b'
            for path in (root / 'cases' / other).iterdir():
                path.unlink()
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                result = client.get('/api/detections').json['detections']
                if result and result[0]['label'] == selected:
                    break
                time.sleep(.1)
            assert result[0]['label'] == selected
            # Camera header survives both stages.
            captured = []
            with patch.object(node.recognition_pub, 'publish', side_effect=captured.append):
                assert client.post('/api/source', data={'source': 'camera'}).status_code == 200
                frame = node.bridge.cv2_to_imgmsg(np.asarray(picture), encoding='rgb8')
                frame.header.frame_id = 'test_camera'
                frame.header.stamp.sec = 345
                node.on_image(frame)
                wait_result(client)
            assert captured and captured[-1].header == frame.header
            assert captured[-1].detections[0].results[0].hypothesis.class_id == selected
            print('PASS: grounded ROI inference, per-object heatmaps, reference views, stale cache rejection, unknown, static refresh, camera headers')
        finally:
            node.stop.set()
            node.server.shutdown()
            node.preview_worker.join(2)
            node.worker.join(30)
            node.destroy_node()
            rclpy.shutdown()
