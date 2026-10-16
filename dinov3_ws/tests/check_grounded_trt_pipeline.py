"""Real dual-runtime ROS/HTTP smoke, including a deliberately stalled renderer."""
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
import rclpy

from dinov3_cases.node import CaseNode
from dinov3_cases.runtime import ARTIFACTS, WORKSPACE


def main():
    image = Image.open(WORKSPACE / 'DART/dartf/data/train2017/000000230971.jpg').convert('RGB')
    entered, release = threading.Event(), threading.Event()
    original_encode = cv2.imencode

    def stalled_encode(*args, **kwargs):
        if threading.current_thread() is node.renderer_worker and not release.is_set():
            entered.set()
            if not release.wait(30):
                raise RuntimeError('Renderer test timed out')
        return original_encode(*args, **kwargs)

    def png():
        data = io.BytesIO()
        image.save(data, format='PNG')
        data.seek(0)
        return data

    with tempfile.TemporaryDirectory() as directory:
        rclpy.init(args=['--ros-args', '-p', f'cases_dir:={directory}', '-p', 'port:=0',
                        '-p', 'pose_enabled:=false', '-p', 'grounding_prompt:=person'])
        node = CaseNode()
        client = node.app.test_client()

        def wait_for(predicate, seconds=90):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=.01)
                status = client.get('/api/status').json
                if status['status'].startswith(('Error:', 'Model load failed:')):
                    raise AssertionError(status['status'])
                if predicate(status):
                    return status
            raise AssertionError(client.get('/api/status').json)

        try:
            with patch('dinov3_cases.node.cv2.imencode', side_effect=stalled_encode):
                assert client.post('/api/cases', data={'case': 'sample', 'images': (png(), 'ref.png')}).status_code == 200
                assert client.post('/api/source', data={'source': 'upload', 'image': (png(), 'target.png')}).status_code == 200
                status = wait_for(lambda s: s['performance'].get('completed_at') is not None)
                assert entered.wait(5)
                detections = client.get('/api/detections').json['detections']
                assert detections and not status['overlay_ready']
                assert status['grounding_worker']['tensorrt'].startswith('10.')
                assert status['grounding_execution'] == 'subprocess'
                assert status['performance']['render_ms'] is None
                release.set()
                wait_for(lambda s: s['overlay_ready'])
            assert client.get('/overlay.jpg').status_code == 200
            assert client.get('/overlay.jpg?bbox=3d').status_code == 200
            assert client.post('/api/source', data={'source': 'camera'}).status_code == 200
            ros_results = []
            from vision_msgs.msg import Detection2DArray
            node.create_subscription(Detection2DArray, '/medicine_recognition/detections',
                                     lambda msg: ros_results.append(msg.header.frame_id), 10)
            rows = []
            for index in range(4):
                msg = node.bridge.cv2_to_imgmsg(np.asarray(image), encoding='rgb8')
                msg.header.frame_id = f'memory-smoke-{index}'
                msg.header.stamp = node.get_clock().now().to_msg()
                node.on_image(msg)
                expected = node.frame_seq
                status = wait_for(lambda s: s['performance'].get('frame_id') == expected)
                rows.append(status['performance'])
            wait_for(lambda _: 'memory-smoke-3' in ros_results, seconds=10)
            report = dict(passed=True, bbox_before_jpeg=True, camera_frames=len(rows),
                          ros_results=ros_results, worker=status['grounding_worker'], rows=rows)
            output = ARTIFACTS / 'sam3/memory_pipeline_validation.json'
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(report, indent=2))
        finally:
            release.set()
            node.stop.set()
            node.image_changed.set()
            node.preview_changed.set()
            node.publish_changed.set()
            node.render_changed.set()
            node.server.shutdown()
            node.close_grounding_worker()
            node.worker.join(15)
            node.preview_worker.join(3)
            node.publisher_worker.join(3)
            node.renderer_worker.join(3)
            node.destroy_node()
            rclpy.shutdown()
        assert all(slot[0] is None for slot in node.frame_pool.slots)


if __name__ == '__main__':
    main()
