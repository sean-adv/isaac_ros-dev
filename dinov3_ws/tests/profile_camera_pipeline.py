"""Profile the real camera pipeline without changing production inference code."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import rclpy
from dinov3_cases import node as pipeline
from dinov3_cases.runtime import GROUNDING_ENGINE, DINO_ENGINE, GROUNDING_PYTHON, PROJECT
from dinov3_cases.trt_backend import TRTBackbone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--no-pose', action='store_true')
    parser.add_argument('--max-detections', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    extras = {}
    def timed(cls, method, key):
        original = getattr(cls, method)
        def wrapper(self, *a, **kw):
            start = time.monotonic()
            result = original(self, *a, **kw)
            extras[key] = extras.get(key, 0) + (time.monotonic() - start) * 1000
            return result
        setattr(cls, method, wrapper)
    timed(pipeline.Encoder, 'encode', 'roi_encode_ms')
    timed(TRTBackbone, '__call__', 'dinov3_trt_ms')
    class ProfileNode(pipeline.CaseNode):
        @property
        def performance(self):
            return self._performance
        @performance.setter
        def performance(self, value):
            if 'completed_at' in value:
                value = dict(value, **extras, objects=len(self.detections),
                             bbox3d_statuses=[x.get('bbox3d_status') for x in self.detections])
            self._performance = value
        def on_image(self, msg):
            self.arrivals.append(time.monotonic())
            super().on_image(msg)
        def depth_inputs_for(self, frame):
            extras.clear()
            if isinstance(frame, pipeline.ImageMsg):
                stamp = frame.header.stamp
                extras['stamp_to_depth_stage_ms'] = (self.get_clock().now().nanoseconds / 1e9 - stamp.sec - stamp.nanosec / 1e9) * 1000
                extras['image_size'] = [frame.width, frame.height]
                if not self.saved_frame:
                    from PIL import Image
                    Image.fromarray(self.bridge.imgmsg_to_cv2(frame, desired_encoding='rgb8')).save(args.output.parent / 'camera_sample.png')
                    self.saved_frame = True
            start = time.monotonic()
            result = super().depth_inputs_for(frame)
            extras['depth_wait_ms'] = (time.monotonic() - start) * 1000
            return result
    ros_args = ['--ros-args', '--params-file', str(PROJECT / 'config/dinov3_cases.yaml')]
    for key, value in dict(port=0, engine_path=DINO_ENGINE, grounding_engine_path=GROUNDING_ENGINE,
                           grounding_worker_python=GROUNDING_PYTHON,
                           pose_enabled=str(not args.no_pose).lower(), grounding_max_detections=args.max_detections).items():
        ros_args += ['-p', f'{key}:={value}']
    ProfileNode.arrivals = []
    ProfileNode.saved_frame = False
    rclpy.init(args=ros_args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    node = ProfileNode()
    rows, output_rows = [], []
    def on_result(msg):
        stamp = msg.header.stamp
        output_rows.append(dict(at=time.monotonic(), stamp_to_ros_result_ms=(node.get_clock().now().nanoseconds / 1e9 - stamp.sec - stamp.nanosec / 1e9) * 1000))
    node.create_subscription(pipeline.Detection2DArray, '/medicine_recognition/detections', on_result, 10)
    seen, unique, started = None, 0, None
    deadline = time.monotonic() + 180
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.01)
            with node.lock:
                perf, status = dict(node.performance), node.status
            if status.startswith(('Error:', 'Model load failed:')):
                raise RuntimeError(status)
            completed = perf.get('completed_at')
            if completed is not None and completed != seen:
                seen, unique = completed, unique + 1
                if unique > args.warmup:
                    if started is None:
                        started = time.monotonic()
                        print('Warmup complete; measuring camera pipeline.', flush=True)
                    rows.append(perf)
            if started and time.monotonic() - started >= args.seconds:
                break
        if len(rows) < 2:
            raise RuntimeError(f'No camera results: {status}')
        def stats(values):
            return dict(mean=float(np.mean(values)), median=float(np.median(values)), p95=float(np.percentile(values,95)), min=float(np.min(values)), max=float(np.max(values)))
        summary = {}
        for key in rows[0]:
            if key.endswith('_ms') or key == 'objects':
                values = [r[key] for r in rows if isinstance(r.get(key), (int,float))]
                if values: summary[key] = stats(values)
        for key in rows[0].get('grounding_ipc', {}):
            if key.endswith('_ms'):
                summary['ipc.'+key] = stats([r['grounding_ipc'][key] for r in rows])
        summary['result_fps'] = (len(rows)-1)/(rows[-1]['completed_at']-rows[0]['completed_at'])
        arrivals = [t for t in node.arrivals if rows[0]['completed_at'] <= t <= rows[-1]['completed_at']]
        summary['camera_callback_fps'] = (len(arrivals)-1)/(arrivals[-1]-arrivals[0]) if len(arrivals)>1 else None
        outputs = [r for r in output_rows if rows[0]['completed_at'] <= r['at'] <= rows[-1]['completed_at']]
        if outputs: summary['stamp_to_ros_result_ms'] = stats([r['stamp_to_ros_result_ms'] for r in outputs])
        report = dict(options=vars(args) | {'output':str(args.output)}, samples=len(rows), status=status, summary=summary, rows=rows, ros_outputs=outputs)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        node.stop.set()
        node.image_changed.set()
        node.publish_changed.set()
        node.render_changed.set()
        node.server.shutdown()
        node.worker.join(15)
        node.close_grounding_worker()
        node.video_publisher_worker.join(3)
        node.publisher_worker.join(3)
        node.renderer_worker.join(3)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
