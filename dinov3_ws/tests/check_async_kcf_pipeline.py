"""Real SAM3/DINOv3 smoke: results continue while a SAM3 call is stalled."""

import json
from pathlib import Path
import tempfile
import threading
import time

import numpy as np
from PIL import Image
import rclpy

from dinov3_cases.node import CaseNode
from dinov3_cases.runtime import ARTIFACTS, WORKSPACE


def main():
    image = Image.open(WORKSPACE / 'DART/dartf/data/train2017/000000230971.jpg').convert('RGB')
    entered, release = threading.Event(), threading.Event()
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory) / 'sample'
        folder.mkdir()
        image.save(folder / 'ref.png')
        rclpy.init(args=['--ros-args', '-p', f'cases_dir:={directory}', '-p', 'port:=0',
                        '-p', 'pose_enabled:=false', '-p', 'grounding_prompt:=person',
                        '-p', 'grounding_detection_interval:=3', '-p', 'opencv_num_threads:=1',
                        '-p', 'kcf_replay_max_frames:=64', '-p', 'kcf_replay_budget_ms:=2000.0'], domain_id=131)
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

        def send_frame():
            msg = node.bridge.cv2_to_imgmsg(np.asarray(image), encoding='rgb8')
            msg.header.frame_id = 'async-smoke-camera'
            msg.header.stamp = node.get_clock().now().to_msg()
            node.on_image(msg)
            return node.frame_seq

        try:
            # Keep the source running through a deliberately slow first SAM3.
            # Initialization must complete without the profiler's old input hold.
            wait_for(lambda _: hasattr(node, 'grounding_detector'))
            real_detect = node.grounding_detector.detect
            initial_entered = threading.Event()

            def slow_initial_detect(*args, **kwargs):
                initial_entered.set()
                time.sleep(.5)
                return real_detect(*args, **kwargs)

            node.grounding_detector.detect = slow_initial_detect
            bootstrap_start = time.monotonic()
            next_frame = bootstrap_start
            deadline = bootstrap_start + 30
            bootstrap_frames = 0
            while time.monotonic() < deadline:
                if time.monotonic() >= next_frame:
                    seq = send_frame()
                    bootstrap_frames += 1
                    next_frame += 1/30
                rclpy.spin_once(node, timeout_sec=.005)
                status = client.get('/api/status').json
                if status['status'].startswith(('Error:', 'Model load failed:')):
                    raise AssertionError(status['status'])
                if status['performance'].get('completed_at') is not None:
                    break
            assert initial_entered.is_set()
            assert status['performance'].get('completed_at') is not None, status
            assert bootstrap_frames > 5
            schedule = status['performance']['grounding_schedule']
            assert schedule['seed_skipped_frames'] == 0, schedule
            assert schedule['discard_reason'] is None, schedule
            assert status['performance']['grounding_schedule']['initializing'] is False
            assert client.get('/api/detections').json['detections']
            bootstrap = dict(input_frames=bootstrap_frames, input_target_fps=30,
                             elapsed_ms=(time.monotonic()-bootstrap_start)*1000,
                             performance=status['performance'])
            node.grounding_detector.detect = real_detect

            def stalled_detect(*args, **kwargs):
                entered.set()
                assert release.wait(30), 'Stalled SAM3 was never released'
                return real_detect(*args, **kwargs)

            node.grounding_detector.detect = stalled_detect
            rows = []
            # At least one next job must start while KCF keeps producing results.
            for _ in range(3):
                seq = send_frame()
                status = wait_for(lambda s: s['performance'].get('frame_id') == seq, seconds=10)
                rows.append(status['performance'])
            assert entered.wait(5)
            assert not release.is_set()
            for _ in range(2):
                seq = send_frame()
                status = wait_for(lambda s: s['performance'].get('frame_id') == seq, seconds=10)
                assert status['performance']['bbox_source'] == 'kcf'
                assert client.get('/api/detections').json['detections']
                rows.append(status['performance'])
            previous_revision = node.object_debug_revision
            release.set()
            status = wait_for(lambda s: s['performance'].get('bbox_source') == 'sam3')
            assert status['performance']['frame_id'] == seq
            assert status['performance']['grounding_ipc']['source_frame_id'] < seq
            schedule = status['performance']['grounding_schedule']
            assert schedule['seed_frame_key'][-1] == status['performance']['grounding_ipc']['source_frame_id'], schedule
            assert schedule['seed_skipped_frames'] == 0, schedule
            assert schedule['replay_frames'] > 0, schedule
            assert status['performance']['recognition_source'] == 'dino', status
            visualizations = client.get('/api/object_visualizations').json
            assert visualizations['ready'] and visualizations['objects'], visualizations
            assert visualizations['revision'] > previous_revision, visualizations
            report = dict(passed=True, bootstrap=bootstrap, kcf_results_during_stalled_sam3=True,
                          rows=rows, handoff=status['performance'])
            output = ARTIFACTS / 'sam3/async_kcf_validation.json'
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(report, indent=2))
        finally:
            release.set()
            node.stop.set()
            for event in (node.image_changed, node.preview_changed,
                          node.publish_changed, node.render_changed):
                event.set()
            with node.stream_changed:
                node.stream_changed.notify_all()
            node.server.shutdown()
            node.close_grounding_worker()
            for worker in (node.preview_worker, node.worker, node.publisher_worker,
                           node.renderer_worker):
                worker.join(10)
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
