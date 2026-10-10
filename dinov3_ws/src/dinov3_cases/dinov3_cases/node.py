"""ROS input, inference scheduling and result publication for both TRT modes."""

import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image as ImageMsg, CameraInfo
from std_msgs.msg import Header
from werkzeug.serving import make_server

from .patches import feature_stride
from .encoder import Encoder
from .runtime import (
    WORKSPACE,
    DINO_ENGINE,
    GROUNDING_ENGINE,
    LEGACY_PYTHON,
    INPUT_SIZE,
)
from vision_msgs.msg import Detection2DArray, Detection3DArray
from .depth_pose import matched_inputs, attach_surface_poses, poses_message
from .web import create_app
from .grounded import (
    DEFAULT_PROMPT,
    GroundingDetector,
    ObjectGallery,
    validate_settings,
    recognize_objects,
    recognition_message,
    ObjectVisualizationSnapshot,
)


class CaseNode(Node):
    def __init__(self):
        super().__init__("dinov3_cases")
        defaults = dict(
            cases_dir=str(WORKSPACE / "dinov3_cases_data"),
            grounding_execution="inprocess",
            grounding_worker_python=str(LEGACY_PYTHON),
            grounding_ipc_timeout=60.0,
            grounding_engine_path=str(GROUNDING_ENGINE),
            grounding_prompt=DEFAULT_PROMPT,
            grounding_box_threshold=0.3,
            grounding_text_threshold=0.25,
            grounding_nms_iou=0.5,
            grounding_max_detections=20,
            recognition_threshold=0.75,
            recognition_margin=0.05,
            image_topic="/camera/color/image_raw",
            pose_enabled=True,
            depth_topic="/camera/depth/image_raw",
            depth_camera_info_topic="/camera/depth/camera_info",
            pose_sync_tolerance=0.05,
            pose_min_depth=0.1,
            pose_max_depth=5.0,
            pose_depth_band=0.08,
            pose_min_points=80,
            pose_axis_ratio=1.15,
            engine_path=str(DINO_ENGINE),
            port=8090,
            host="127.0.0.1",
            inference_hz=0.0,
            preview_max_width=960,
            jpeg_quality=80,
        )
        for k, v in defaults.items():
            self.declare_parameter(k, v)
        self.param = lambda k: self.get_parameter(k).value
        self.grounding_settings = dict(
            prompt=self.param("grounding_prompt"),
            box_threshold=self.param("grounding_box_threshold"),
            text_threshold=self.param("grounding_text_threshold"),
            recognition_threshold=self.param("recognition_threshold"),
            margin=self.param("recognition_margin"),
        )
        self.grounding_settings["prompt"] = validate_settings(**self.grounding_settings)
        if not 0 <= self.param("grounding_nms_iou") <= 1:
            raise ValueError("grounding_nms_iou must be between 0 and 1.")
        if not 1 <= self.param("grounding_max_detections") <= 100:
            raise ValueError("grounding_max_detections must be between 1 and 100.")
        if self.param("grounding_execution") not in ("inprocess", "subprocess"):
            raise ValueError("grounding_execution must be inprocess or subprocess.")
        if (
            not np.isfinite(self.param("grounding_ipc_timeout"))
            or self.param("grounding_ipc_timeout") <= 0
        ):
            raise ValueError("grounding_ipc_timeout must be positive.")
        if not np.isfinite(self.param("inference_hz")) or self.param("inference_hz") < 0:
            raise ValueError("inference_hz must be finite and nonnegative; 0 is unlimited.")
        if not 1 <= self.param("jpeg_quality") <= 100 or self.param("preview_max_width") < 1:
            raise ValueError("jpeg_quality must be 1–100 and preview_max_width must be positive.")
        for name in ("pose_sync_tolerance", "pose_min_depth", "pose_max_depth", "pose_depth_band"):
            if not np.isfinite(self.param(name)) or self.param(name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.param("pose_min_depth") >= self.param("pose_max_depth"):
            raise ValueError("pose_min_depth must be less than pose_max_depth.")
        if self.param("pose_min_points") < 3:
            raise ValueError("pose_min_points must be at least 3.")
        if not np.isfinite(self.param("pose_axis_ratio")) or self.param("pose_axis_ratio") <= 1:
            raise ValueError("pose_axis_ratio must be finite and greater than 1.")
        self.root = Path(self.param("cases_dir")).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.image_changed = threading.Event()
        self.preview_changed = threading.Event()
        self.stream_changed = threading.Condition(self.lock)
        self.depth_changed = threading.Condition(self.lock)
        self.depth_frames = deque(maxlen=30)
        self.depth_infos = deque(maxlen=8)
        self.performance = {}
        self.pending_publication = None
        self.publish_changed = threading.Event()
        self.stop = threading.Event()
        self.selected = ""
        self.generation = 0
        self.source = "camera"
        self.frame = None
        self.frame_seq = 0
        self.jpeg = None
        self.clear_object_debug()
        self.detections = []
        self.preview_jpeg = None
        self.camera_frame = None
        self.camera_seq = 0
        self.camera_received_at = None
        self.preview_error = None
        self.overlay_at = None
        self.status = "Loading DINOv3 model…"
        self.backend_description = None
        self.reference_stride = None
        self.pub = self.create_publisher(ImageMsg, "/dinov3/heatmap/overlay", 1)
        self.recognition_pub = self.create_publisher(
            Detection2DArray, "/medicine_recognition/detections", 10
        )
        self.pose_pub = self.create_publisher(
            Detection3DArray, "/medicine_recognition/poses", 10
        )
        self.recognition_debug_pub = self.create_publisher(
            ImageMsg, "/medicine_recognition/debug_image", 1
        )
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.BEST_EFFORT
        )
        self.create_subscription(ImageMsg, self.param("image_topic"), self.on_image, image_qos)
        if self.param("pose_enabled"):
            self.create_subscription(ImageMsg, self.param("depth_topic"), self.on_depth, image_qos)
            self.create_subscription(
                CameraInfo, self.param("depth_camera_info_topic"), self.on_depth_info, image_qos
            )
        self.app = create_app(self)
        self.server = make_server(self.param("host"), self.param("port"), self.app, threaded=True)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.preview_worker = threading.Thread(target=self.run_preview, daemon=True)
        self.preview_worker.start()
        self.publisher_worker = threading.Thread(target=self.run_publications, daemon=True)
        self.publisher_worker.start()
        self.worker = threading.Thread(target=self.run_grounded_worker, daemon=True)
        self.worker.start()
        self.get_logger().info(f"Browser: http://{self.param('host')}:{self.param('port')}")

    def on_image(self, msg):
        with self.lock:
            self.camera_frame = msg
            self.camera_seq += 1
            self.camera_received_at = time.monotonic()
            self.preview_changed.set()
            if self.source != "camera":
                return
            self.frame = msg
            self.frame_seq += 1
            self.image_changed.set()

    def on_depth(self, msg):
        with self.depth_changed:
            self.depth_frames.append(msg)
            self.depth_changed.notify_all()

    def on_depth_info(self, msg):
        with self.depth_changed:
            self.depth_infos.append(msg)
            self.depth_changed.notify_all()

    def depth_inputs_for(self, frame):
        """Capture depth before slow inference, so later frames cannot replace it."""
        if not self.param("pose_enabled") or not isinstance(frame, ImageMsg):
            return None
        deadline = time.monotonic() + self.param("pose_sync_tolerance")
        with self.depth_changed:
            while not self.stop.is_set():
                inputs = matched_inputs(
                    frame.header, frame.width, frame.height, self.depth_frames,
                    self.depth_infos, self.param("pose_sync_tolerance")
                )
                if inputs is not None:
                    return inputs
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.depth_changed.wait(remaining)
        return None

    def estimate_poses(self, detections, inputs):
        if inputs is None:
            for item in detections:
                item["pose_status"] = (
                    "unknown" if item["label"] == "unknown"
                    else "missing_synchronized_registered_depth"
                )
            return
        depth_msg, info = inputs
        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            if depth_msg.encoding == "16UC1":
                depth = depth.astype(np.float64) * 0.001
            elif depth_msg.encoding == "32FC1":
                depth = depth.astype(np.float64)
            else:
                raise ValueError(f"unsupported_depth_encoding:{depth_msg.encoding}")
            attach_surface_poses(
                detections, depth, info,
                min_depth=self.param("pose_min_depth"), max_depth=self.param("pose_max_depth"),
                depth_band=self.param("pose_depth_band"), min_points=self.param("pose_min_points"),
                axis_ratio=self.param("pose_axis_ratio"),
            )
        except (ValueError, CvBridgeError) as error:
            for item in detections:
                item["pose_status"] = "unknown" if item["label"] == "unknown" else str(error)

    def worker_ticks(self):
        """Wake on input; rate limit start-to-start, never sleep after processing."""
        deadline = 0.0
        while not self.stop.is_set():
            self.image_changed.wait(0.1)  # Also recheck uploads, settings and gallery changes.
            self.image_changed.clear()
            if self.stop.wait(max(0.0, deadline - time.monotonic())):
                return
            hz = float(self.param("inference_hz"))
            deadline = time.monotonic() + (1 / hz if hz > 0 else 0)
            yield

    def run_publications(self):
        """DDS output cannot stall inference; pending output is a single latest slot."""
        while not self.stop.is_set():
            self.publish_changed.wait(0.1)
            self.publish_changed.clear()
            with self.lock:
                pending, self.pending_publication = self.pending_publication, None
                generation = self.generation
            if pending is None:
                continue
            overlay, detections, header, gen = pending
            if gen != generation:
                continue
            try:
                if header is None:
                    header = Header(stamp=self.get_clock().now().to_msg())
                if self.recognition_pub.get_subscription_count():
                    self.recognition_pub.publish(recognition_message(detections, header))
                if self.pose_pub.get_subscription_count():
                    self.pose_pub.publish(poses_message(detections, header))
                image_publishers = [
                    pub
                    for pub in (self.pub, self.recognition_debug_pub)
                    if pub.get_subscription_count()
                ]
                if image_publishers:
                    out = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
                    out.header = header
                    for publisher in image_publishers:
                        publisher.publish(out)
            except Exception as error:
                self.get_logger().error(f"ROS result publication failed: {error}")

    def run_preview(self):
        processed = -1
        while not self.stop.is_set():
            self.preview_changed.wait(0.1)
            self.preview_changed.clear()
            with self.lock:
                frame, seq = self.camera_frame, self.camera_seq
            if frame is None or seq == processed:
                continue
            try:
                bgr = self.bridge.imgmsg_to_cv2(frame, desired_encoding="bgr8")
                width = max(1, int(self.param("preview_max_width")))
                if bgr.shape[1] > width:
                    bgr = cv2.resize(bgr, (width, round(bgr.shape[0] * width / bgr.shape[1])))
                ok, data = cv2.imencode(
                    ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.param("jpeg_quality")]
                )
                if not ok:
                    raise ValueError("Could not encode camera frame")
                with self.lock:
                    self.preview_jpeg = data.tobytes()
                    self.preview_error = None
                    self.stream_changed.notify_all()
            except Exception as e:
                with self.lock:
                    self.preview_error = str(e)
            processed = seq

    def close_grounding_worker(self):
        detector = getattr(self, "grounding_detector", None)
        close = getattr(detector, "close", None)
        if close is not None:
            close()

    def destroy_node(self):
        self.stop.set()
        self.close_grounding_worker()
        return super().destroy_node()

    def clear_object_debug(self):
        self.object_debug_revision = getattr(self, "object_debug_revision", 0) + 1
        self.object_debug_snapshot = None
        self.object_debug_history = deque(maxlen=3)

    def run_grounded_worker(self):
        try:
            encoder = Encoder(self.param("engine_path"))
            stride = feature_stride(encoder.model)
            with self.lock:
                self.reference_stride = stride
                self.backend_description = encoder.description
                self.status = "Loading Grounding DINO model…"
            detector_class = GroundingDetector
            detector_options = {}
            if self.param("grounding_execution") == "subprocess":
                from .grounding_ipc import SharedMemoryGroundingDetector

                detector_class = SharedMemoryGroundingDetector
                detector_options = dict(
                    python=self.param("grounding_worker_python"),
                    timeout=self.param("grounding_ipc_timeout"),
                )
            detector = detector_class(
                self.param("grounding_engine_path"), encoder.device, **detector_options
            )
            self.grounding_detector = detector
            if self.stop.is_set():
                self.close_grounding_worker()
                return
            with self.lock:
                detector.validate_prompt(self.grounding_settings["prompt"])
                config = getattr(getattr(detector, "model", None), "config", None)
                self.grounding_text_token_limit = getattr(config, "max_text_len", None)
        except Exception as error:
            self.close_grounding_worker()
            with self.lock:
                self.status = f"Model load failed: {type(error).__name__}: {error}"
            self.get_logger().error(self.status)
            return
        gallery = ObjectGallery()
        processed = None
        next_gallery_scan = 0.0
        gallery_revision = 0
        max_side = INPUT_SIZE
        for _ in self.worker_ticks():
            with self.lock:
                gen, frame, seq = self.generation, self.frame, self.frame_seq
                settings = dict(self.grounding_settings)
                received_at = self.camera_received_at if self.source == "camera" else None
            try:
                # Include filesystem changes even with an uploaded, otherwise static target.
                if (
                    time.monotonic() >= next_gallery_scan
                    or processed is None
                    or processed[0] != gen
                ):
                    if gallery.refresh(self.root, encoder, max_side, stride):
                        gallery_revision += 1
                    next_gallery_scan = time.monotonic() + 1.0
                with self.lock:
                    if gen != self.generation:
                        continue
                    gen, frame, seq = self.generation, self.frame, self.frame_seq
                    settings = dict(self.grounding_settings)
                    received_at = self.camera_received_at if self.source == "camera" else None
                if frame is None:
                    with self.lock:
                        if gen == self.generation:
                            self.status = f"Ready: {len(set(gallery.labels))} classes. Waiting for target image…"
                    continue
                if processed == (gen, seq, gallery_revision):
                    continue
                start = time.monotonic()
                header = None
                if isinstance(frame, ImageMsg):
                    image = Image.fromarray(
                        self.bridge.imgmsg_to_cv2(frame, desired_encoding="rgb8")
                    )
                    header = frame.header
                else:
                    image = frame
                depth_inputs = self.depth_inputs_for(frame)
                proposals = detector.detect(
                    image,
                    settings["prompt"],
                    settings["box_threshold"],
                    settings["text_threshold"],
                    self.param("grounding_nms_iou"),
                    self.param("grounding_max_detections"),
                )
                detected_at = time.monotonic()
                diagnostics = []
                detections = recognize_objects(
                    image,
                    proposals,
                    encoder,
                    gallery,
                    max_side,
                    stride,
                    settings["recognition_threshold"],
                    settings["margin"],
                    diagnostics=diagnostics,
                    defer_diagnostics=True,
                )
                recognized_at = time.monotonic()
                if self.param("pose_enabled") and header is not None:
                    self.estimate_poses(detections, depth_inputs)
                else:
                    for item in detections:
                        item["pose_status"] = "disabled" if not self.param("pose_enabled") else "no_camera_depth"
                posed_at = time.monotonic()
                object_snapshot = ObjectVisualizationSnapshot(
                    detections, diagnostics, gallery, max_side, stride
                )
                overlay = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                for index, item in enumerate(detections):
                    x0, y0, x1, y1 = item["bbox"]
                    color = (0, 190, 255) if item["label"] == "unknown" else (60, 220, 80)
                    cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), color, 2)
                    score = "--" if item["score"] is None else f"{item['score']:.2f}"
                    label = f"#{index+1} {item['label']} | {item['detection_label']} | det {item['detection_score']:.2f} sim {score}"
                    font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, font, font_scale, thickness
                    )
                    origin = (
                        max(4, min(x0, overlay.shape[1] - text_width - 4)),
                        max(text_height + 6, y0 - 8),
                    )
                    cv2.putText(overlay, label, origin, font, font_scale, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(
                        overlay, label, origin, font, font_scale, color, thickness, cv2.LINE_AA
                    )
                ok, jpeg = cv2.imencode(
                    ".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, self.param("jpeg_quality")]
                )
                if not ok:
                    raise RuntimeError("JPEG encoding failed.")
                with self.lock:
                    if gen != self.generation:
                        continue
                    self.jpeg = jpeg.tobytes()
                    self.object_debug_revision += 1
                    self.object_debug_snapshot = object_snapshot
                    self.detections = detections
                    self.overlay_at = time.monotonic()
                    self.stream_changed.notify_all()
                    previous_completed = self.performance.get("completed_at")
                    self.performance = dict(
                        completed_at=self.overlay_at,
                        result_fps=(
                            None
                            if previous_completed is None
                            else 1 / max(1e-9, self.overlay_at - previous_completed)
                        ),
                        grounding_ipc=dict(getattr(detector, "last_metrics", {})),
                        detect_ms=(detected_at - start) * 1000,
                        recognize_ms=(recognized_at - detected_at) * 1000,
                        pose_ms=(posed_at - recognized_at) * 1000,
                        pose_count=sum("pose" in item for item in detections),
                        render_ms=(self.overlay_at - posed_at) * 1000,
                        processing_ms=(self.overlay_at - start) * 1000,
                        callback_to_result_ms=(
                            None if received_at is None else (self.overlay_at - received_at) * 1000
                        ),
                    )
                    self.status = (
                        f"Grounding DINO → {encoder.description} | {len(detections)} objects | "
                        f"{len(set(gallery.labels))} classes / {len(gallery.labels)} references | "
                        f"{time.monotonic()-start:.2f}s inference"
                    )
                    self.pending_publication = (overlay, detections, header, gen)
                    self.publish_changed.set()
                processed = (gen, seq, gallery_revision)
            except Exception as error:
                with self.lock:
                    if gen == self.generation:
                        self.jpeg = None
                        self.clear_object_debug()
                        self.detections = []
                        self.overlay_at = None
                        self.status = f"Error: {type(error).__name__}: {error}"
                self.get_logger().error(str(error))
                # Retry failed files/model calls at a bounded rate, including static uploads.
                self.stop.wait(1.0)


def main(args=None):
    rclpy.init(args=args)
    node = CaseNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop.set()
        node.image_changed.set()
        node.preview_changed.set()
        node.publish_changed.set()
        with node.stream_changed:
            node.stream_changed.notify_all()
        node.server.shutdown()
        node.close_grounding_worker()
        node.preview_worker.join(timeout=2)
        node.worker.join(timeout=10)
        node.publisher_worker.join(timeout=2)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
