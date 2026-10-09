"""Smoke check the real Grounding DINO + ViT-S/16 TRT ROS/HTTP worker."""
import io
import json
from pathlib import Path
import tempfile
import time

from PIL import Image
import rclpy

from dinov3_cases.node import CaseNode
from dinov3_cases.trt_backend import ARTIFACTS


def main():
    image = Image.open(Path(__file__).resolve().parents[2] /
                       'dinov3_cases_data/stomach/1790733215386907348_2.png').convert('RGB')
    def png():
        data = io.BytesIO()
        image.save(data, format='PNG')
        data.seek(0)
        return data
    with tempfile.TemporaryDirectory() as tmp:
        rclpy.init(args=['--ros-args', '-p', f'cases_dir:={tmp}', '-p', 'port:=0',
                        '-p', 'object_max_side:=448'])
        node = CaseNode()
        client = node.app.test_client()
        def wait():
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                status = client.get('/api/status').json
                assert not status['status'].startswith(('Error:', 'Model load failed:')), status
                detections = client.get('/api/detections').json['detections']
                if status['overlay_ready'] and detections:
                    return status, detections
                time.sleep(.1)
            raise AssertionError(client.get('/api/status').json)
        try:
            assert client.post('/api/cases', data={'case': 'sample', 'images': (png(), 'ref.png')}).status_code == 200
            assert client.post('/api/source', data={'source': 'upload', 'image': (png(), 'target.png')}).status_code == 200
            status, detections = wait()
            assert status['inference_backend'] == status['grounding_backend'] == 'trt'
            assert status['grounding_model_id'] == 'IDEA-Research/grounding-dino-tiny'
            assert status['model_id'] == 'facebook/dinov3-vits16-pretrain-lvd1689m'
            assert 'vits16' in status['backend_description']
            assert any(item['label'] == 'sample' for item in detections), detections
            manifest = client.get('/api/object_visualizations').json
            assert manifest['objects']
            revision = manifest['revision']
            for view in ('crop', 'grid', 'heatmap', 'reference'):
                assert client.get(f'/debug/objects/{revision}/0/{view}.jpg').status_code == 200
            invalid = client.post('/api/grounding', json={'prompt': 'medicine bottle. medicine box. bottle. box. car. truck.'})
            assert invalid.status_code == 400, invalid.json
            assert client.get('/api/status').json['grounding']['prompt'] == status['grounding']['prompt']
            assert client.post('/api/grounding', json={'prompt': 'box. bottle.'}).status_code == 200
            changed_status, changed = wait()
            assert changed_status['grounding']['prompt'] == 'box. bottle.'
            assert any(item['label'] == 'sample' for item in changed), changed
            report = {'passed': True, 'status': status, 'detections': detections,
                      'changed_prompt_detections': changed}
            path = ARTIFACTS / 'integration_grounding_wingdzero_vits16.json'
            path.write_text(json.dumps(report, indent=2) + '\n')
            print(f'PASS: real dual-TRT worker, gallery recognition, per-object previews, prompt update. {path}')
        finally:
            node.stop.set()
            node.server.shutdown()
            node.preview_worker.join(5)
            node.worker.join(60)
            node.publisher_worker.join(5)
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
