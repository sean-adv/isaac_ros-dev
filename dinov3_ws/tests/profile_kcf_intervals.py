"""Compare live-camera SAM3 intervals and time real KCF calls in both threads."""
import argparse
from collections import deque
import json
from pathlib import Path
import threading
import time

import numpy as np
import rclpy
from dinov3_cases.kcf import KCFProposals
from dinov3_cases.node import CaseNode
from dinov3_cases.runtime import WORKSPACE


def stats(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return None if not values else dict(samples=len(values), median=float(np.median(values)),
                                       p95=float(np.percentile(values, 95)), mean=float(np.mean(values)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intervals', type=int, nargs='+', default=[30, 15, 10, 5, 30])
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--warmup-seconds', type=float, default=5)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument("--prompt", default=None, help="Override SAM3 prompt for a replay scene")
    parser.add_argument("--box-threshold", type=float, default=None)
    parser.add_argument("--min-objects", type=int, default=1, help="Reject benchmarks with no useful KCF workload")
    parser.add_argument("--image", type=Path, help="Replay a saved RGB scene on an isolated 30 Hz ROS input")
    parser.add_argument("--parallel-workers", type=int, default=0,
                        help="0=serial, >=1=persistent owners; output topics are isolated")
    parser.add_argument('--browser-streams', action='store_true',
                        help='Render both browser streams to measure their CPU cost')
    parser.add_argument('--no-pose', action='store_true',
                        help='Disable depth synchronization and 3D boxes for comparison')
    args = parser.parse_args()
    if min(args.intervals) < 2 or args.seconds <= 0 or args.warmup_seconds < 0 or args.parallel_workers < 0:
        parser.error('intervals must be >=2; seconds >0; warmup >=0')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    calls, results = deque(maxlen=100000), deque(maxlen=100000)
    publications, received = deque(maxlen=100000), deque(maxlen=100000)
    original = KCFProposals.update

    def timed(self, pixels, frame_key, *a, **kw):
        changed = frame_key != self.frame_key
        initialized = self.initialized
        start = time.perf_counter()
        result = original(self, pixels, frame_key, *a, **kw)
        end = time.perf_counter()
        calls.append(dict(at=time.monotonic(), ms=(end-start)*1000,
                          thread=threading.current_thread().name, objects=len(result[0]),
                          tracked=initialized and changed and result[1] == 'kcf',
                          image_size=list(pixels.shape[:2][::-1])))
        return result

    KCFProposals.update = timed

    class ProfileNode(CaseNode):
        @property
        def last_publication(self):
            return getattr(self, '_last_publication', None)

        @last_publication.setter
        def last_publication(self, value):
            publications.append(dict(value))
            self._last_publication = value

        @property
        def performance(self):
            return self._performance

        @performance.setter
        def performance(self, value):
            if 'completed_at' in value:
                results.append(dict(value, objects=len(self.detections)))
            self._performance = value

    rclpy.init(args=['--ros-args', '--params-file',
                    str(WORKSPACE / 'dinov3_ws/config/dinov3_cases.yaml'), '-p', 'port:=0',
                    '-p', 'pose_enabled:=' + str(not args.no_pose).lower(),
                    '-p', 'kcf_parallel_enabled:=' + str(args.parallel_workers > 0).lower(),
                    '-p', 'kcf_parallel_workers:=' + str(args.parallel_workers or 1),
                    '-r', '/medicine_recognition/detections:=/kcf_benchmark/detections',
                    '-r', '/medicine_recognition/detections_3d:=/kcf_benchmark/detections_3d',
                    '-p', 'tag_topic:=/kcf_benchmark/tags',
                    '-p', 'h264_topic:=/kcf_benchmark/h264',
                    '-p', 'annotated_image_topic:=/kcf_benchmark/image',
                    *(['-p', 'image_topic:=/kcf_benchmark/input'] if args.image else []),
                    *(['-p', 'grounding_prompt:=' + args.prompt] if args.prompt else []),
                    *(['-p', 'grounding_box_threshold:=' + str(args.box_threshold)]
                      if args.box_threshold is not None else [])])
    node = ProfileNode()
    from vision_msgs.msg import Detection2DArray

    def on_result(message):
        stamp = message.header.stamp
        source_ns = stamp.sec * 10**9 + stamp.nanosec
        received.append(dict(at=time.monotonic(),
                             source_to_subscriber_ms=(node.get_clock().now().nanoseconds-source_ns)/1e6,
                             objects=len(message.detections)))

    subscription = node.create_subscription(Detection2DArray, '/kcf_benchmark/detections', on_result, 10)
    if args.image:
        import cv2
        from sensor_msgs.msg import Image as ImageMsg
        from cv_bridge import CvBridge
        bgr = cv2.imread(str(args.image))
        if bgr is None:
            raise ValueError('Cannot read scene: ' + str(args.image))
        messages = [CvBridge().cv2_to_imgmsg(np.roll(bgr, n, axis=1), encoding='bgr8')
                    for n in range(4)]
        publisher = node.create_publisher(ImageMsg, '/kcf_benchmark/input', 1)
        frame_counter = 0

        def publish_input():
            nonlocal frame_counter
            message = messages[frame_counter % len(messages)]
            message.header.stamp = node.get_clock().now().to_msg()
            message.header.frame_id = 'kcf_benchmark_camera'
            publisher.publish(message)
            frame_counter += 1

        timer = node.create_timer(1/30, publish_input)

    if args.browser_streams:
        from sensor_msgs.msg import CompressedImage
        node.create_subscription(CompressedImage, node.param("h264_topic"), lambda _: None, 30)
    reports = []

    def spin_until(deadline):
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.01)
            if node.status.startswith(('Error:', 'Model load failed:')):
                raise RuntimeError(node.status)

    try:
        deadline = time.monotonic() + 180
        while not results and time.monotonic() < deadline:
            spin_until(time.monotonic() + .1)
        if not results:
            raise RuntimeError('No camera results: ' + node.status)
        client = node.app.test_client()
        for interval in args.intervals:
            previous = node.performance.get('completed_at')
            response = client.post('/api/grounding', json={'detection_interval': interval})
            if response.status_code != 200:
                raise RuntimeError(response.get_data(as_text=True))
            bootstrap_deadline = time.monotonic() + 30
            while node.performance.get('completed_at') == previous and time.monotonic() < bootstrap_deadline:
                spin_until(time.monotonic() + .05)
            if node.performance.get('completed_at') == previous:
                raise RuntimeError('No initial detection for interval ' + str(interval))
            spin_until(time.monotonic() + args.warmup_seconds)
            start = time.monotonic()
            start_camera_seq = node.camera_seq
            cpu_start = time.process_time()
            spin_until(start + args.seconds)
            end = time.monotonic()
            cpu_elapsed = time.process_time()-cpu_start
            pub_rows = [p for p in list(publications) if start <= p["published_at"] <= end]
            sub_rows = [p for p in list(received) if start <= p["at"] <= end]
            rows = [r for r in list(results) if start <= r['completed_at'] <= end
                    and r['detection_interval'] == interval]
            timings = [c for c in list(calls) if start <= c['at'] <= end]
            main_calls = [c for c in timings if c['tracked'] and c['thread'] != 'sam3-kcf-replay']
            if len(rows) < 2:
                raise RuntimeError('No steady-state results for interval ' + str(interval))
            if np.median([r['objects'] for r in rows]) < args.min_objects:
                raise RuntimeError('Insufficient objects for a KCF comparison')
            unique_frames = len({r['frame_id'] for r in rows})
            sam_rows = [r for r in rows if r.get('grounding_ipc')]
            source_ids = {r['grounding_ipc'].get('source_frame_id') for r in sam_rows}
            summary = dict(architecture='persistent_workers' if args.parallel_workers else 'serial',
                           workers=args.parallel_workers, process_cpu_percent=100*cpu_elapsed/(end-start),
                           ros_publications=len(pub_rows), ros_received=len(sub_rows),
                           callback_to_publish_ms=stats([r['callback_to_publish_ms'] for r in pub_rows]),
                           result_to_publish_ms=stats([r['result_to_publish_ms'] for r in pub_rows]),
                           source_to_subscriber_ms=stats([r['source_to_subscriber_ms'] for r in sub_rows]),
                           interval=interval, samples=len(rows), unique_frames=unique_frames,
                           result_fps=len(rows)/(end-start), processed_frame_fps=unique_frames/(end-start),
                           camera_callback_fps=(node.camera_seq-start_camera_seq)/(end-start),
                           image_reliability="reliable",
                           objects=stats([r['objects'] for r in rows]),
                           kcf_update_ms=stats([c['ms'] for c in main_calls]),
                           replay_update_ms=stats([c['ms'] for c in timings if c['tracked'] and c['thread']=='sam3-kcf-replay']),
                           kcf_detect_stage_ms=stats([r['detect_ms'] for r in rows if r['bbox_source']=='kcf']),
                           processing_ms=stats([r['processing_ms'] for r in rows]),
                           callback_to_result_ms=stats([r['callback_to_result_ms'] for r in rows]),
                           sam3_unique_handoffs=len(source_ids),
                           sam3_handoff_hz=len(source_ids)/(end-start),
                           sam3_worker_ms=stats([r['grounding_ipc'].get('worker_detect_ms') for r in sam_rows]))
            for key in ('image_prepare_ms', 'shared_copy_ms', 'depth_wait_ms',
                        'proposal_update_ms', 'recognize_ms', 'bbox3d_ms', 'render_ms'):
                summary[key] = stats([r.get(key) for r in rows])
            reports.append(dict(summary=summary, rows=rows, kcf_calls=timings,
                                ros_publications=pub_rows, ros_received=sub_rows))
            args.output.write_text(json.dumps(dict(options=vars(args) | {'output':str(args.output), 'image':str(args.image) if args.image else None},
                                                   status=node.status, reports=reports), indent=2)+'\n')
            print(json.dumps(summary), flush=True)
    finally:
        node.stop.set()
        for event in (node.image_changed, node.publish_changed, node.render_changed):
            event.set()
        node.server.shutdown()
        node.worker.join(15)
        node.close_grounding_worker()
        for worker in (node.video_publisher_worker, node.publisher_worker, node.renderer_worker):
            worker.join(3)
        node.destroy_node()
        rclpy.shutdown()
        KCFProposals.update = original


if __name__ == '__main__':
    main()
