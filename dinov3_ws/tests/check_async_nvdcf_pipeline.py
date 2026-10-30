"""Real models: geometry precedes DINO; tracking continues through either stall."""

import json
from pathlib import Path
import tempfile
import threading
import time
from unittest.mock import patch

import numpy as np
from PIL import Image
import rclpy

from dinov3_cases.node import CaseNode, TrackedObjectRecognizer
from dinov3_cases.runtime import ARTIFACTS


def main():
    image = Image.open(ARTIFACTS / 'validation/person.jpg').convert('RGB')
    entered, release = threading.Event(), threading.Event()
    dino_entered, dino_release = threading.Event(), threading.Event()
    real_recognize = TrackedObjectRecognizer.update

    def gated_recognize(self, *args, **kwargs):
        dino_entered.set()
        assert dino_release.wait(30), 'Stalled DINO was never released'
        return real_recognize(self, *args, **kwargs)

    patcher = patch.object(TrackedObjectRecognizer, 'update', gated_recognize)
    patcher.start()
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory) / 'sample'
        folder.mkdir()
        image.save(folder / 'ref.png')
        rclpy.init(args=['--ros-args', '-p', f'cases_dir:={directory}', '-p', 'port:=0',
                        '-p', 'pose_enabled:=false', '-p', 'grounding_prompt:=person',
                        '-p', 'grounding_detection_interval:=3', '-p', 'opencv_num_threads:=1'], domain_id=131)
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
            assert schedule['backend'] == 'nvdcf', schedule
            assert status['performance']['grounding_schedule']['initializing'] is False
            assert client.get('/api/detections').json['detections']
            assert status['performance']['recognition_source'] == 'pending', status
            assert dino_entered.wait(5)
            pending_initial = client.get('/api/detections').json['detections']
            assert all(p['recognition_pending'] for p in pending_initial)
            assert not dino_release.is_set()
            bootstrap = dict(input_frames=bootstrap_frames, input_target_fps=30,
                             elapsed_ms=(time.monotonic()-bootstrap_start)*1000,
                             performance=status['performance'])
            node.grounding_detector.detect = real_detect

            def stalled_detect(*args, **kwargs):
                entered.set()
                assert release.wait(30), 'Stalled SAM3 was never released'
                return real_detect(*args, **kwargs)

            node.grounding_detector.detect = stalled_detect
            dino_release.set()
            wait_for(lambda s: s['performance'].get('recognition_source') == 'dino')
            initial_ids = {p['track_id'] for p in client.get('/api/detections').json['detections']}
            assert initial_ids == {p['track_id'] for p in pending_initial}
            rows = []
            # At least one next job must start while NvDCF keeps producing results.
            for _ in range(3):
                seq = send_frame()
                status = wait_for(lambda s: s['performance'].get('frame_id') == seq, seconds=10)
                rows.append(status['performance'])
            assert entered.wait(5)
            assert not release.is_set()
            for _ in range(2):
                seq = send_frame()
                status = wait_for(lambda s: s['performance'].get('frame_id') == seq, seconds=10)
                assert status['performance']['bbox_source'] == 'nvdcf'
                assert client.get('/api/detections').json['detections']
                rows.append(status['performance'])
            previous_revision = node.object_debug_revision
            dino_entered.clear()
            dino_release.clear()
            release.set()
            status = wait_for(lambda s: s['performance'].get('bbox_source') == 'sam3')
            assert status['performance']['frame_id'] == seq
            assert status['performance']['grounding_ipc']['source_frame_id'] < seq
            schedule = status['performance']['grounding_schedule']
            assert schedule['seed_frame_key'][-1] == status['performance']['grounding_ipc']['source_frame_id'], schedule
            assert schedule['backend'] == 'nvdcf', schedule
            assert schedule['replay_frames'] > 0, schedule
            assert status['performance']['recognition_source'] == 'pending', status
            assert dino_entered.wait(5)
            geometry = status['performance']
            pending = client.get('/api/detections').json['detections']
            assert pending and all(p['recognition_pending'] for p in pending)
            assert not dino_release.is_set()
            # Advance another RGB while classification is explicitly blocked.
            seq = send_frame()
            wait_for(lambda s: s['performance'].get('frame_id') == seq)
            assert client.get('/api/detections').json['detections']
            dino_release.set()
            status = wait_for(lambda s: s['performance'].get('recognition_source') == 'dino')
            assert status['performance']['recognition_frame_id'] == geometry['grounding_ipc']['source_frame_id']
            classified = client.get('/api/detections').json['detections']
            assert {p['track_id'] for p in classified} == {p['track_id'] for p in pending}
            assert all(not p['recognition_pending'] for p in classified)
            visualizations = client.get('/api/object_visualizations').json
            assert visualizations['ready'] and visualizations['objects'], visualizations
            assert visualizations['revision'] > previous_revision, visualizations
            report = dict(passed=True, bootstrap=bootstrap, nvdcf_results_during_stalled_sam3=True,
                          geometry_before_dino=True, tracking_during_dino=True,
                          rows=rows, geometry=geometry, handoff=status['performance'])
            output = ARTIFACTS / 'sam3/async_nvdcf_validation.json'
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(dict(passed=True, bootstrap_frames=bootstrap_frames, nvdcf_results_during_stalled_sam3=True, handoff_frame=seq, report=str(output)), indent=2))
        finally:
            release.set()
            dino_release.set()
            patcher.stop()
            node.stop.set()
            for event in (node.image_changed,
                          node.publish_changed, node.render_changed):
                event.set()
            node.server.shutdown()
            node.close_grounding_worker()
            for worker in (node.video_publisher_worker, node.worker, node.publisher_worker,
                           node.renderer_worker):
                worker.join(10)
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
