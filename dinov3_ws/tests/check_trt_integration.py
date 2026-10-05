"""Real-GPU TRT integration smoke test; writes a numerical report, not a benchmark."""
import argparse
import gc
import io
import json
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image
import rclpy
import torch

from dinov3_cases.node import Encoder, CaseNode
from dinov3_cases.trt_backend import DEFAULT_VARIANT, MODEL_VARIANTS, model_variant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=sorted(MODEL_VARIANTS), default=DEFAULT_VARIANT)
    args = parser.parse_args()
    spec = model_variant(args.model)
    model_id, engine_path = spec['model_id'], spec['engine_path']
    workspace = Path(__file__).resolve().parents[2]
    picture = Image.open(workspace / 'dinov3_cases_data/stomach/1790733215386907348_2.png').convert('RGB')
    trt_encoder = Encoder(model_id, 'trt', engine_path)
    torch_encoder = Encoder(model_id, 'torch')
    report = {'variant': args.model, 'model_id': model_id, 'engine': engine_path, 'images': []}
    for image in (picture.crop((0, 0, 87, 53)), picture.crop((0, 0, 1280, 720)), picture):
        with torch.inference_mode():
            actual = trt_encoder.spatial_features(image)
            expected = torch_encoder.spatial_features(image)
            assert actual.shape == expected.shape
            assert torch.isfinite(actual).all()
            cosine = (actual * expected).sum(-1)
            row = {'image_wh': list(image.size), 'feature_shape': list(actual.shape),
                   'cosine_mean': cosine.mean().item(), 'cosine_min': cosine.min().item(),
                   'max_abs_error': (actual - expected).abs().max().item()}
            print(json.dumps(row), flush=True)
            # Broad smoke tolerance; detailed task accuracy still needs representative targets.
            assert row['cosine_mean'] > .99, row
            assert row['cosine_min'] > .95, row
            report['images'].append(row)
            del actual, expected, cosine
    try:
        trt_encoder.model(torch.empty((1, 3, 2561, 32), device='cuda'))
        raise AssertionError('Out-of-profile image accepted')
    except ValueError as error:
        assert 'outside' in str(error)
    del trt_encoder, torch_encoder
    gc.collect()
    torch.cuda.empty_cache()

    # Exercise the actual ROS worker, web uploads, bank rebuild and camera callback.
    with tempfile.TemporaryDirectory() as tmp:
        rclpy.init(args=['--ros-args', '-p', 'inference_backend:=trt',
                        '-p', f'model_id:={model_id}', '-p', f'engine_path:={engine_path}',
                        '-p', f'cases_dir:={tmp}',
                        '-p', 'host:=127.0.0.1', '-p', 'port:=0',
                        '-p', 'debug_visualization:=true'])
        node = CaseNode()
        client = node.app.test_client()
        image = picture.crop((0, 0, 321, 193))
        def png():
            data = io.BytesIO()
            image.save(data, format='PNG')
            data.seek(0)
            return data
        def wait_overlay():
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                status = client.get('/api/status').json
                assert not status['status'].startswith(('Error:', 'Model load failed:')), status
                if client.get('/overlay.jpg').status_code == 200:
                    return status
                time.sleep(.1)
            raise AssertionError(client.get('/api/status').json)
        try:
            assert client.post('/api/cases', data={
                'case': 'trt_smoke', 'images': [(png(), 'reference.png')]}).status_code == 200
            assert client.post('/api/source', data={
                'source': 'upload', 'image': (png(), 'target.png')}).status_code == 200
            status = wait_overlay()
            assert status['inference_backend'] == 'trt'
            assert 'TensorRT' in status['backend_description']
            assert node.bank.is_cuda and torch.isfinite(node.bank).all()
            bank_before = node.bank
            # Add another reference and ensure the worker rebuilds the bank.
            assert client.post('/api/cases', data={
                'case': 'trt_smoke', 'images': [(png(), 'reference2.png')]}).status_code == 200
            wait_overlay()
            assert node.bank is not bank_before
            assert client.post('/api/threshold', json={'threshold': .8}).status_code == 200
            wait_overlay()
            for view in ('grid', 'patches', 'heatmap', 'mask', 'boxes'):
                assert client.get(f'/debug/{view}.jpg').status_code == 200
            assert client.get('/api/detections').json['detections']
            assert client.post('/api/source', data={'source': 'camera'}).status_code == 200
            msg = node.bridge.cv2_to_imgmsg(np.asarray(image), encoding='rgb8')
            msg.header.frame_id = 'trt_test_camera'
            node.on_image(msg)
            status = wait_overlay()
            report['web_ros'] = {'passed': True, 'status': status['status'],
                                'backend_description': status['backend_description']}
        finally:
            node.stop.set()
            node.server.shutdown()
            node.preview_worker.join(5)
            node.worker.join(90)
            node.destroy_node()
            rclpy.shutdown()
    path = Path(engine_path).parent / f'integration_{args.model}.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'PASS: native TRT features, case bank rebuild, web/ROS pipeline. Report: {path}', flush=True)


if __name__ == '__main__':
    main()
