"""ROS input, inference scheduling and result publication with isolated TRT10/TRT11 runtimes."""

import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image as ImageMsg, CameraInfo
from std_msgs.msg import Header
from werkzeug.serving import make_server

from .timing import ResultRate
from .patches import feature_stride
from .encoder import Encoder
from .grounding_ipc import SharedMemoryGroundingDetector
from .frame_pool import SharedFramePool
from .tracking import BBoxSmoother
from .kcf import KCFProposals, kcf_factory
from .async_kcf import AsyncKCFProposals, validate_replay_limits
from .runtime import (
    WORKSPACE,
    DINO_ENGINE,
    GROUNDING_ENGINE,
    GROUNDING_PYTHON,
    INPUT_SIZE,
)
from vision_msgs.msg import Detection2DArray, Detection3DArray
from isaac_ros_apriltag_interfaces.msg import AprilTagDetectionArray
from .apriltag import apriltag_message
from .depth_sync import matched_inputs
from .display_stream import has_display_boxes, display_overlay_requested, display_uses_overlay
from .depth_bbox import ProjectionRayCache, attach_bboxes3d, bboxes3d_message
from .web import create_app
from .visualization import render_bbox_overlay, render_display_overlay
from .object_visualization import ObjectVisualizationSnapshot
from .grounded import (
    DEFAULT_PROMPT,
    ObjectGallery,
    validate_settings,
    validate_bbox_limits,
    TrackedObjectRecognizer,
    recognition_message,
)


def encode_display_jpeg(pixels, quality):
    """Encode browser images at one fixed size, with at most one resize."""
    if pixels.shape[:2] != (540, 960):
        pixels = cv2.resize(pixels, (960, 540), interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return data.tobytes()


class CaseNode(Node):
    def __init__(self):
        super().__init__("dinov3_cases")
        defaults = dict(
            cases_dir=str(WORKSPACE / "dinov3_cases_data"),
            grounding_worker_python=str(GROUNDING_PYTHON),
            grounding_ipc_timeout=60.0,
            grounding_engine_path=str(GROUNDING_ENGINE),
            grounding_prompt=DEFAULT_PROMPT,
            grounding_box_threshold=0.3,
            grounding_nms_iou=1.0,
            grounding_max_detections=20,
            grounding_detection_interval=1,
            bbox_smoothing_frames=3,
            kcf_roi_padding=0.5,
            opencv_num_threads=0,
            kcf_replay_max_frames=8,
            kcf_replay_budget_ms=150.0,
            grounding_min_bbox_width=0,
            grounding_min_bbox_height=0,
            grounding_min_bbox_area=0,
            recognition_threshold=0.75,
            recognition_margin=0.05,
            image_topic="/camera/color/image_raw",
            tag_topic="/tag_detections",
            pose_enabled=True,
            depth_topic="/camera/depth/image_raw",
            depth_camera_info_topic="/camera/depth/camera_info",
            pose_sync_tolerance=0.05,
            pose_min_depth=0.1,
            pose_max_depth=5.0,
            bbox3d_depth_band=0.08,
            bbox3d_min_points=80,
            bbox3d_quantile=0.02,
            engine_path=str(DINO_ENGINE),
            port=8090,
            host="127.0.0.1",
            inference_hz=0.0,
            jpeg_quality=70,
            bbox_visualization_mode="3d",
        )
        for k, v in defaults.items():
            self.declare_parameter(k, v)
        self.param = lambda k: self.get_parameter(k).value
        self.bbox_visualization_mode = self.param("bbox_visualization_mode")
        if self.bbox_visualization_mode not in ("2d", "3d"):
            raise ValueError("bbox_visualization_mode must be 2d or 3d.")
        opencv_threads = self.param("opencv_num_threads")
        if isinstance(opencv_threads, bool) or not isinstance(opencv_threads, int) or opencv_threads < 0:
            raise ValueError("opencv_num_threads must be an integer >= 0 (0 keeps the OpenCV default).")
        # Set the global OpenCV pool before starting inference/render workers.
        if opencv_threads > 0:
            cv2.setNumThreads(opencv_threads)
        validate_replay_limits(self.param("kcf_replay_max_frames"),
                               self.param("kcf_replay_budget_ms"))
        KCFProposals(roi_padding=self.param("kcf_roi_padding"))
        self.bbox_smoother = BBoxSmoother(self.param("bbox_smoothing_frames"))
        self.grounding_settings = dict(
            prompt=self.param("grounding_prompt"),
            box_threshold=self.param("grounding_box_threshold"),
            recognition_threshold=self.param("recognition_threshold"),
            margin=self.param("recognition_margin"),
            detection_interval=self.param("grounding_detection_interval"),
        )
        self.grounding_settings["prompt"] = validate_settings(**self.grounding_settings)
        if self.grounding_settings["detection_interval"] > 1:
            kcf_factory()
        if not 0 <= self.param("grounding_nms_iou") <= 1:
            raise ValueError("grounding_nms_iou must be between 0 and 1.")
        if not 1 <= self.param("grounding_max_detections") <= 100:
            raise ValueError("grounding_max_detections must be between 1 and 100.")
        validate_bbox_limits(
            self.param("grounding_min_bbox_width"),
            self.param("grounding_min_bbox_height"),
            self.param("grounding_min_bbox_area"),
        )
        if (
            not np.isfinite(self.param("grounding_ipc_timeout"))
            or self.param("grounding_ipc_timeout") <= 0
        ):
            raise ValueError("grounding_ipc_timeout must be positive.")
        if not np.isfinite(self.param("inference_hz")) or self.param("inference_hz") < 0:
            raise ValueError("inference_hz must be finite and nonnegative; 0 is unlimited.")
        if not 1 <= self.param("jpeg_quality") <= 100:
            raise ValueError("jpeg_quality must be 1–100.")
        for name in ("pose_sync_tolerance", "pose_min_depth", "pose_max_depth", "bbox3d_depth_band"):
            if not np.isfinite(self.param(name)) or self.param(name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not 0 <= self.param("bbox3d_quantile") < 0.5:
            raise ValueError("bbox3d_quantile must be in [0, 0.5).")
        if (not isinstance(self.param("bbox3d_min_points"), int)
                or self.param("bbox3d_min_points") < 5):
            raise ValueError("bbox3d_min_points must be an integer >= 5.")
        if self.param("pose_min_depth") >= self.param("pose_max_depth"):
            raise ValueError("pose_min_depth must be less than pose_max_depth.")
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
        self.depth_ray_cache = ProjectionRayCache()
        self.performance = {}
        self.frame_pool = SharedFramePool(slots=3)
        self.pending_render = None
        self.render_changed = threading.Event()
        self.stream_viewers = {}
        self.stream_requested_at = {}
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
        self.camera_rate = ResultRate(window_seconds=3.0)
        self.camera_fps = None
        self.preview_error = None
        self.overlay_at = None
        self.overlay_has_boxes = False
        self.status = "Loading DINOv3 model…"
        self.backend_description = None
        self.reference_stride = None
        self.pub = self.create_publisher(ImageMsg, "/dinov3/heatmap/overlay", 1)
        self.recognition_pub = self.create_publisher(
            Detection2DArray, "/medicine_recognition/detections", 10
        )
        self.bbox3d_pub = self.create_publisher(
            Detection3DArray, "/medicine_recognition/detections_3d", 10
        )
        self.tag_pub = self.create_publisher(
            AprilTagDetectionArray, self.param("tag_topic"), 10
        )
        self.recognition_debug_pub = self.create_publisher(
            ImageMsg, "/medicine_recognition/debug_image", 1
        )
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.RELIABLE
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
        self.renderer_worker = threading.Thread(target=self.run_rendering, daemon=True)
        self.renderer_worker.start()
        self.worker = threading.Thread(target=self.run_grounded_worker, daemon=True)
        self.worker.start()
        self.get_logger().info(f"Browser: http://{self.param('host')}:{self.param('port')}")

    def on_image(self, msg):
        with self.lock:
            self.camera_frame = msg
            self.camera_seq += 1
            self.camera_received_at = time.monotonic()
            self.camera_fps = self.camera_rate.update(self.camera_received_at, msg.header.frame_id)
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

    def estimate_bboxes3d(self, detections, inputs):
        if inputs is None:
            for item in detections:
                item["bbox3d_status"] = "missing_synchronized_registered_depth"
            return
        depth_msg, info = inputs
        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            if depth_msg.encoding == "16UC1":
                depth_scale = 0.001
            elif depth_msg.encoding == "32FC1":
                depth_scale = 1.0
            else:
                raise ValueError(f"unsupported_depth_encoding:{depth_msg.encoding}")
            attach_bboxes3d(
                detections, depth, info,
                depth_scale=depth_scale, ray_cache=self.depth_ray_cache,
                min_depth=self.param("pose_min_depth"), max_depth=self.param("pose_max_depth"),
                depth_band=self.param("bbox3d_depth_band"),
                min_points=self.param("bbox3d_min_points"),
                quantile=self.param("bbox3d_quantile"),
            )
        except (ValueError, CvBridgeError) as error:
            for item in detections:
                item["bbox3d_status"] = str(error)

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
            detections, header, gen = pending
            if gen != generation:
                continue
            try:
                if header is None:
                    header = Header(stamp=self.get_clock().now().to_msg())
                if self.recognition_pub.get_subscription_count():
                    self.recognition_pub.publish(recognition_message(detections, header))
                if self.bbox3d_pub.get_subscription_count():
                    self.bbox3d_pub.publish(bboxes3d_message(detections, header))
                if self.tag_pub.get_subscription_count():
                    self.tag_pub.publish(apriltag_message(detections, header))
            except Exception as error:
                self.get_logger().error(f"ROS result publication failed: {error}")

    def wants_stream(self, attribute):
        with self.lock:
            direct_reader = (self.stream_viewers.get(attribute, 0) > 0
                             or time.monotonic() - self.stream_requested_at.get(
                                 attribute, -float("inf")) < 2)
            if direct_reader:
                return True
            if self.stream_viewers.get("display", 0) > 0:
                if attribute == "jpeg":
                    return display_overlay_requested(self)
                return self.source == "camera" and not display_uses_overlay(self)
            # Keep the existing initial snapshot for standalone image consumers.
            return getattr(self, attribute) is None

    def acquire_input_frame(self, pixels, *, frame_id, timestamp):
        """Reclaim a queued old overlay before letting rendering exhaust input slots."""
        frame = self.frame_pool.acquire(pixels, frame_id=frame_id, timestamp=timestamp)
        if frame is not None:
            return frame
        with self.lock:
            pending, self.pending_render = self.pending_render, None
        if pending is None:
            return None
        pending[0].release()
        return self.frame_pool.acquire(pixels, frame_id=frame_id, timestamp=timestamp)

    def run_rendering(self):
        """One active and one latest pending job; never block bbox publication."""
        while not self.stop.is_set():
            self.render_changed.wait(0.1)
            self.render_changed.clear()
            with self.lock:
                job, self.pending_render = self.pending_render, None
            if job is None:
                continue
            frame, detections, header, gen, seq, info, completed_at = job
            try:
                with self.lock:
                    if gen != self.generation:
                        continue
                want_jpeg = self.wants_stream("jpeg")
                publishers = [p for p in (self.pub, self.recognition_debug_pub)
                              if p.get_subscription_count()]
                if not (want_jpeg or publishers):
                    continue
                start = time.monotonic()
                jpeg = None
                if want_jpeg:
                    display = render_display_overlay(
                        frame.pixels, detections, info, self.bbox_visualization_mode
                    )
                    jpeg = encode_display_jpeg(display, self.param("jpeg_quality"))
                with self.lock:
                    if gen != self.generation:
                        continue
                    if want_jpeg:
                        self.jpeg = jpeg
                        self.overlay_has_boxes = has_display_boxes(
                            detections, self.bbox_visualization_mode)
                    self.overlay_at = time.monotonic()
                    if self.performance.get("completed_at") == completed_at:
                        self.performance["render_ms"] = (self.overlay_at - start) * 1000
                        self.performance["result_to_overlay_ms"] = (self.overlay_at - completed_at) * 1000
                        self.performance["overlay_frame_id"] = seq
                    self.stream_changed.notify_all()
                if publishers:
                    # Full-resolution debug output is rendered only for ROS readers.
                    bgr = cv2.cvtColor(frame.pixels, cv2.COLOR_RGB2BGR)
                    overlay = render_bbox_overlay(bgr, detections, info, self.bbox_visualization_mode)
                    out = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
                    out.header = header if header is not None else Header(stamp=self.get_clock().now().to_msg())
                    for publisher in publishers:
                        publisher.publish(out)
            except Exception as error:
                self.get_logger().error(f"Result rendering failed: {error}")
            finally:
                frame.release()

    def run_preview(self):
        processed = -1
        while not self.stop.is_set():
            self.preview_changed.wait(0.1)
            self.preview_changed.clear()
            with self.lock:
                frame, seq = self.camera_frame, self.camera_seq
            if frame is None or seq == processed or not self.wants_stream("preview_jpeg"):
                continue
            try:
                bgr = self.bridge.imgmsg_to_cv2(frame, desired_encoding="bgr8")
                jpeg = encode_display_jpeg(bgr, self.param("jpeg_quality"))
                with self.lock:
                    self.preview_jpeg = jpeg
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
        proposals = getattr(self, "proposal_tracker", None)
        if proposals is not None:
            proposals.close()

    def destroy_node(self):
        self.stop.set()
        self.close_grounding_worker()
        self.render_changed.set()
        self.renderer_worker.join(timeout=3)
        with self.lock:
            if self.pending_render is not None:
                self.pending_render[0].release()
                self.pending_render = None
        self.frame_pool.close()
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
                self.status = "Loading SAM3 W8A8 model…"
            detector = SharedMemoryGroundingDetector(
                self.param("grounding_engine_path"), encoder.device,
                python=self.param("grounding_worker_python"),
                timeout=self.param("grounding_ipc_timeout"),
            )
            self.grounding_detector = detector
            if self.stop.is_set():
                self.close_grounding_worker()
                return
            with self.lock:
                detector.validate_prompt(self.grounding_settings["prompt"])
                self.status = "Models ready: Preparing first detection and recognition…"
        except Exception as error:
            self.close_grounding_worker()
            with self.lock:
                self.status = f"Model load failed: {type(error).__name__}: {error}"
            self.get_logger().error(self.status)
            return
        gallery = ObjectGallery()
        processed = None
        next_upload_refresh = 0.0
        proposal_tracker = AsyncKCFProposals(
            wake=self.image_changed.set,
            roi_padding=self.param("kcf_roi_padding"),
            max_replay_frames=self.param("kcf_replay_max_frames"),
            replay_budget_ms=self.param("kcf_replay_budget_ms"),
        )
        self.proposal_tracker = proposal_tracker
        last_recognition = None
        had_detections = False
        result_rate = ResultRate(window_seconds=3.0)
        max_side = INPUT_SIZE
        for _ in self.worker_ticks():
            shared_frame = None
            with self.lock:
                gen = self.generation
            try:
                # Static uploads have no new frames to trigger periodic model jobs.
                if self.source == "upload" and time.monotonic() >= next_upload_refresh:
                    with proposal_tracker.lock:
                        proposal_tracker.force_detection = True
                    next_upload_refresh = time.monotonic() + 1.0
                with self.lock:
                    if gen != self.generation:
                        continue
                    gen, frame, seq = self.generation, self.frame, self.frame_seq
                    settings = dict(self.grounding_settings)
                    received_at = self.camera_received_at if self.source == "camera" else None
                if frame is None:
                    with self.lock:
                        if gen == self.generation:
                            self.status = "Ready: Waiting for target image…"
                    continue
                if (processed == (gen, seq)
                        and not proposal_tracker.has_result()):
                    continue
                start = time.monotonic()
                header = frame.header if isinstance(frame, ImageMsg) else None
                image_size = (frame.width, frame.height) if header is not None else frame.size
                context = (gen, image_size, header.frame_id if header is not None else None)
                if isinstance(frame, ImageMsg):
                    pixels = self.bridge.imgmsg_to_cv2(frame, desired_encoding="rgb8")
                else:
                    pixels = np.asarray(frame.convert("RGB"))
                shared_frame = self.acquire_input_frame(pixels, frame_id=seq, timestamp=received_at)
                if shared_frame is None:
                    continue
                image_ready_at = time.monotonic()
                # Preserve synchronized depth before first-result inference waits.
                depth_inputs = self.depth_inputs_for(frame)
                depth_ready_at = time.monotonic()
                depth_wait_ms = (depth_ready_at - image_ready_at) * 1000
                # Capture settings by value: the job may outlive this loop iteration.
                request = dict(
                    prompt=settings["prompt"], box_threshold=settings["box_threshold"],
                    nms_iou=self.param("grounding_nms_iou"),
                    limit=self.param("grounding_max_detections"),
                    min_width=self.param("grounding_min_bbox_width"),
                    min_height=self.param("grounding_min_bbox_height"),
                    min_area=self.param("grounding_min_bbox_area"),
                )

                def detect(frame, request=request):
                    proposals = detector.detect(frame, **request)
                    return proposals, dict(getattr(detector, "last_metrics", {}))

                def recognize(seed_pixels, proposals, seed_key, settings=settings):
                    # This callback runs in the same background job as SAM3.
                    # All gallery GPU work and diagnostics stay off the KCF loop.
                    started = time.monotonic()
                    gallery.refresh(self.root, encoder, max_side, stride)
                    recognizer = TrackedObjectRecognizer()
                    detections, diagnostics, _ = recognizer.update(
                        seed_pixels, proposals, encoder, gallery, max_side, stride,
                        settings["recognition_threshold"], settings["margin"],
                        token=seed_key, gallery_revision=0,
                    )
                    snapshot = ObjectVisualizationSnapshot(
                        recognizer.diagnostic_detections, diagnostics, gallery, max_side, stride
                    )
                    return detections, dict(
                        frame_id=seed_key[1] if proposals else None,
                        snapshot=snapshot,
                        class_count=len(set(gallery.labels)),
                        reference_count=len(gallery.labels),
                        recognize_ms=(time.monotonic() - started) * 1000,
                    )

                result = proposal_tracker.update(
                    shared_frame.pixels, (gen, seq), context,
                    settings["detection_interval"], detect, frame=shared_frame, recognize=recognize,
                    wait_for_initial_result=True,
                )
                if result is None:
                    processed = (gen, seq)
                    continue  # Keep admitting new RGB frames without publishing empty results.
                proposals, proposal_source = result
                detected_at = time.monotonic()
                proposals = self.bbox_smoother.update(proposals, (gen, seq), context)
                # Classification fields follow individual proposals through KCF,
                # including partial loss. Never zip survivors with an old label list.
                detections = proposals
                recognition = proposal_tracker.last_recognition
                recognition_changed = (recognition is not last_recognition
                                       or (had_detections and not detections))
                had_detections = bool(detections)
                last_recognition = recognition
                recognition_frame_id = recognition["frame_id"] if detections else None
                recognition_source = ("none" if not detections else
                                      "dino" if recognition_changed else "cached")
                recognized_at = time.monotonic()
                with self.lock:
                    if gen != self.generation:
                        continue
                    for item in detections:
                        item["bbox_visible"] = True
                if self.param("pose_enabled") and header is not None:
                    self.estimate_bboxes3d(detections, depth_inputs)
                else:
                    for item in detections:
                        item["bbox3d_status"] = "disabled" if not self.param("pose_enabled") else "no_camera_depth"
                bbox3d_at = time.monotonic()
                # Diagnostic crops/features stay on their recognition frame;
                # never relabel an old crop as the current KCF image.
                object_snapshot = (recognition["snapshot"]
                                   if recognition_changed and detections else None)
                with self.lock:
                    if gen != self.generation:
                        continue
                    if recognition_changed:
                        self.object_debug_revision += 1
                        self.object_debug_snapshot = object_snapshot
                    self.detections = detections
                    completed_at = time.monotonic()
                    previous_completed = self.performance.get("completed_at")
                    self.performance = dict(
                        completed_at=completed_at,
                        result_fps=result_rate.update(completed_at, context),
                        result_fps_window_seconds=result_rate.window_seconds,
                        result_fps_instant=(
                            None
                            if previous_completed is None
                            else 1 / max(1e-9, completed_at - previous_completed)
                        ),
                        grounding_ipc=(dict(proposal_tracker.last_metrics)
                                       if proposal_source == "sam3" else {}),
                        grounding_schedule=proposal_tracker.scheduler_metrics(),
                        bbox_source=proposal_source,
                        recognition_source=recognition_source,
                        recognition_frame_id=recognition_frame_id,
                        detection_interval=settings["detection_interval"],
                        image_prepare_ms=(image_ready_at - start) * 1000,
                        shared_copy_ms=shared_frame.copy_ms,
                        depth_wait_ms=depth_wait_ms,
                        proposal_update_ms=(detected_at - depth_ready_at) * 1000,
                        detect_ms=(detected_at - start) * 1000,
                        recognize_ms=(recognition["recognize_ms"] if recognition_changed else 0.0),
                        bbox3d_ms=(bbox3d_at - recognized_at) * 1000,
                        bbox3d_count=sum("bbox3d" in item for item in detections),
                        render_ms=None,
                        frame_id=seq,
                        processing_ms=(completed_at - start) * 1000,
                        callback_to_result_ms=(
                            None if received_at is None else (completed_at - received_at) * 1000
                        ),
                    )
                    self.status = (
                        f"{('SAM3 W8A8' if proposal_source == 'sam3' else 'KCF')} → "
                        f"{encoder.description}{'' if recognition_changed else ' (cached)'} | {len(detections)} objects | "
                        f"{recognition['class_count']} classes / {recognition['reference_count']} references | "
                        f"{time.monotonic()-start:.2f}s inference"
                    )
                    self.pending_publication = (detections, header, gen)
                    if self.pending_render is not None:
                        self.pending_render[0].release()
                    self.pending_render = (shared_frame.retain(), detections, header, gen,
                                           seq, depth_inputs[1] if depth_inputs else None, completed_at)
                    self.render_changed.set()
                    self.publish_changed.set()
                processed = (gen, seq)
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
            finally:
                if shared_frame is not None:
                    shared_frame.release()


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
        node.render_changed.set()
        with node.stream_changed:
            node.stream_changed.notify_all()
        node.server.shutdown()
        node.close_grounding_worker()
        node.preview_worker.join(timeout=2)
        node.worker.join(timeout=10)
        node.publisher_worker.join(timeout=2)
        node.renderer_worker.join(timeout=3)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
