"""ROS input, inference scheduling and result publication with official Torch-TensorRT models."""

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
from sensor_msgs.msg import Image as ImageMsg, CameraInfo, CompressedImage
from std_msgs.msg import Header
from werkzeug.serving import make_server

from .timing import ResultRate
from .patches import feature_stride
from .encoder import Encoder
from .grounding_ipc import SharedMemoryGroundingDetector
from .frame_pool import SharedFramePool
from .tracking import BBoxSmoother
from .nvdcf import check_nvdcf, DEFAULT_CONFIG
from .async_nvdcf import AsyncNvDCFProposals, validate_missed_frames
from .async_bbox3d import AsyncBBox3D, validate_bbox3d_schedule
from .runtime import (
    CASES_DIR,
    DINO_ENGINE,
    GROUNDING_ENGINE,
    GROUNDING_PYTHON,
    INPUT_SIZE,
)
from vision_msgs.msg import Detection2DArray, Detection3DArray
from isaac_ros_apriltag_interfaces.msg import AprilTagDetectionArray
from .apriltag import apriltag_message
from .depth_sync import matched_inputs
from .h264_stream import H264Encoder, PacketHub, WebRTCVideo
from .cloud_filter import CloudFilter
from .depth_bbox import ProjectionRayCache, attach_bboxes3d, bboxes3d_message
from .web import create_app
from .visualization import render_display_overlay
from .object_visualization import ObjectVisualizationSnapshot
from .grounded import (
    DEFAULT_PROMPT,
    ObjectGallery,
    validate_settings,
    validate_bbox_limits,
    validate_containment_threshold,
    TrackedObjectRecognizer,
    recognition_message,
)


class CaseNode(Node):
    def __init__(self):
        super().__init__("dinov3_cases")
        defaults = dict(
            cases_dir=str(CASES_DIR),
            grounding_worker_python=str(GROUNDING_PYTHON),
            grounding_ipc_timeout=60.0,
            grounding_engine_path=str(GROUNDING_ENGINE),
            grounding_prompt=DEFAULT_PROMPT,
            grounding_box_threshold=0.3,
            grounding_containment_threshold=0.95,
            grounding_max_detections=20,
            grounding_detection_interval=1,
            bbox_smoothing_frames=3,
            opencv_num_threads=0,
            nvdcf_config=str(DEFAULT_CONFIG),
            nvdcf_width=640,
            nvdcf_height=384,
            nvdcf_max_missed_detections=3,
            nvdcf_max_missed_frames=30,
            grounding_min_bbox_width=0,
            grounding_min_bbox_height=0,
            grounding_min_bbox_area=0,
            recognition_threshold=0.75,
            recognition_margin=0.05,
            image_topic="/camera/color/image_raw",
            h264_topic="/medicine_recognition/image_annotated/h264",
            annotated_image_topic="/medicine_recognition/image_annotated",
            tag_topic="/tag_detections",
            pose_enabled=True,
            depth_topic="/camera/depth/image_raw",
            depth_camera_info_topic="/camera/depth/camera_info",
            pose_sync_tolerance=0.05,
            pose_min_depth=0.1,
            pose_max_depth=5.0,
            bbox3d_interval=5,
            bbox3d_max_age=1.5,
            bbox3d_depth_band=0.08,
            bbox3d_min_points=80,
            bbox3d_quantile=0.02,
            bbox3d_min_cloud_coverage=0.15,
            bbox3d_min_coverage_ratio=0.35,
            bbox3d_max_depth_jump=0.10,
            bbox3d_depth_confirmations=3,
            engine_path=str(DINO_ENGINE),
            port=8090,
            host="127.0.0.1",
            inference_hz=0.0,
            video_fps=30,
            h264_bitrate=2_000_000,
            h264_encoder="auto",
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
        self.bbox_smoother = BBoxSmoother(self.param("bbox_smoothing_frames"))
        self.grounding_settings = dict(
            prompt=self.param("grounding_prompt"),
            box_threshold=self.param("grounding_box_threshold"),
            recognition_threshold=self.param("recognition_threshold"),
            margin=self.param("recognition_margin"),
            detection_interval=self.param("grounding_detection_interval"),
        )
        self.grounding_settings["prompt"] = validate_settings(**self.grounding_settings)
        validate_missed_frames(self.param("nvdcf_max_missed_frames"))
        if self.grounding_settings["detection_interval"] > 1:
            check_nvdcf(self.param("nvdcf_config"))
        validate_containment_threshold(self.param("grounding_containment_threshold"))
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
        self.video_encoder = H264Encoder(
            self.param("h264_encoder"), self.param("video_fps"), self.param("h264_bitrate")
        )
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
        validate_bbox3d_schedule(self.param("bbox3d_interval"), self.param("bbox3d_max_age"))
        self.cloud_filter = CloudFilter(
            min_coverage=self.param("bbox3d_min_cloud_coverage"),
            min_coverage_ratio=self.param("bbox3d_min_coverage_ratio"),
            max_depth_jump=self.param("bbox3d_max_depth_jump"),
            confirmations=self.param("bbox3d_depth_confirmations"),
        )
        self.root = Path(self.param("cases_dir")).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.image_changed = threading.Event()
        self.depth_changed = threading.Condition(self.lock)
        self.depth_frames = deque(maxlen=30)
        self.depth_infos = deque(maxlen=8)
        self.depth_ray_cache = ProjectionRayCache()
        self.performance = {}
        self.sam3_event_stream = str(time.time_ns())
        self.sam3_event_sequence = 0
        self.sam3_events = deque(maxlen=101)
        self.frame_pool = SharedFramePool(slots=3)
        self.pending_render = None
        self.pending_annotated_image = None
        self.annotated_image_changed = threading.Event()
        self.render_changed = threading.Event()
        self.pending_publication = None
        self.publish_changed = threading.Event()
        self.stop = threading.Event()
        self.selected = ""
        self.generation = 0
        self.source = "camera"
        self.frame = None
        self.frame_seq = 0
        self.clear_object_debug()
        self.detections = []
        self.camera_frame = None
        self.camera_seq = 0
        self.camera_received_at = None
        self.camera_rate = ResultRate(window_seconds=3.0)
        self.camera_fps = None
        self.video_at = None
        self.video_error = None
        self.status = "Loading DINOv3 model…"
        self.backend_description = None
        self.reference_stride = None
        self.recognition_pub = self.create_publisher(
            Detection2DArray, "/medicine_recognition/detections", 10
        )
        self.bbox3d_pub = self.create_publisher(
            Detection3DArray, "/medicine_recognition/detections_3d", 10
        )
        self.tag_pub = self.create_publisher(
            AprilTagDetectionArray, self.param("tag_topic"), 10
        )
        self.h264_pub = self.create_publisher(
            CompressedImage, self.param("h264_topic"), 30
        )
        self.annotated_image_pub = self.create_publisher(
            ImageMsg, self.param("annotated_image_topic"),
            QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                       reliability=ReliabilityPolicy.RELIABLE),
        )
        self.video_hub = PacketHub(wake=self.render_changed.set)
        self.webrtc = WebRTCVideo(self.video_hub)
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
        self.video_publisher_worker = threading.Thread(target=self.run_video_publications, daemon=True)
        self.video_publisher_worker.start()
        self.image_publisher_worker = threading.Thread(target=self.run_image_publications, daemon=True)
        self.image_publisher_worker.start()
        self.publisher_worker = threading.Thread(target=self.run_publications, daemon=True)
        self.publisher_worker.start()
        self.renderer_worker = threading.Thread(target=self.run_rendering, daemon=True)
        self.renderer_worker.start()
        self.bbox3d_worker = AsyncBBox3D(
            self.estimate_bboxes3d, interval=self.param("bbox3d_interval"),
            max_age=self.param("bbox3d_max_age"), wake=self.image_changed.set,
            on_error=lambda error: self.get_logger().error(f"Background 3D boxes failed: {error}"),
        )
        self.worker = threading.Thread(target=self.run_grounded_worker, daemon=True)
        self.worker.start()
        self.get_logger().info(f"Browser: http://{self.param('host')}:{self.param('port')}")

    def on_image(self, msg):
        with self.lock:
            self.camera_frame = msg
            self.camera_seq += 1
            self.camera_received_at = time.monotonic()
            self.camera_fps = self.camera_rate.update(self.camera_received_at, msg.header.frame_id)
            self.render_changed.set()
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
        """Retain a matching depth snapshot without waiting in the RGB loop."""
        if not self.param("pose_enabled") or not isinstance(frame, ImageMsg):
            return None
        with self.depth_changed:
            return matched_inputs(
                frame.header, frame.width, frame.height, self.depth_frames,
                self.depth_infos, self.param("pose_sync_tolerance")
            )

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
                cloud_filter=self.cloud_filter,
                cloud_frame=(depth_msg.header.frame_id, depth_msg.header.stamp.sec,
                             depth_msg.header.stamp.nanosec),
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
                timing = getattr(self, "pending_publication_timing", None)
                self.pending_publication_timing = None
                generation = self.generation
            if pending is None:
                continue
            detections, header, gen = pending
            boxes = getattr(self, "bbox3d_worker", None)
            if boxes is not None:
                detections, _ = boxes.latest(detections)
            if gen != generation:
                continue
            try:
                publish_started = time.monotonic()
                topic_count = 0
                if header is None:
                    header = Header(stamp=self.get_clock().now().to_msg())
                if self.recognition_pub.get_subscription_count():
                    topic_count += 1
                    self.recognition_pub.publish(recognition_message(detections, header))
                if self.bbox3d_pub.get_subscription_count():
                    topic_count += 1
                    self.bbox3d_pub.publish(bboxes3d_message(detections, header))
                if self.tag_pub.get_subscription_count():
                    topic_count += 1
                    self.tag_pub.publish(apriltag_message(detections, header))
                published_at = time.monotonic()
                if timing is not None and topic_count:
                    received_at = timing["received_at"]
                    metrics = dict(
                        frame_id=timing["frame_id"], generation=gen,
                        published_at=published_at, topic_count=topic_count,
                        result_to_publish_ms=(published_at-timing["completed_at"])*1000,
                        dds_submit_ms=(published_at-publish_started)*1000,
                        callback_to_publish_ms=(None if received_at is None else
                                                (published_at-received_at)*1000),
                    )
                    with self.lock:
                        if gen == self.generation:
                            self.last_publication = metrics
                            if (self.performance.get("frame_id") == timing["frame_id"]
                                    and self.performance.get("completed_at") == timing["completed_at"]):
                                self.performance["ros_publication"] = metrics
            except Exception as error:
                self.get_logger().error(f"ROS result publication failed: {error}")

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
        """Select one view, draw once at 960x540, encode once for every consumer."""
        cached = None
        last_key, last_generation = None, None
        last_started, last_keyframe = 0.0, 0.0
        try:
            while not self.stop.is_set():
                self.render_changed.wait(0.2)
                self.render_changed.clear()
                # Pace frame starts: drawing/encoding already consumes the budget.
                # Overruns take the newest pending frame without a catch-up burst.
                delay = last_started + 1 / self.param("video_fps") - time.monotonic()
                if delay > 0 and self.stop.wait(delay):
                    break
                with self.lock:
                    job, self.pending_render = self.pending_render, None
                    generation, source = self.generation, self.source
                    camera, camera_seq = self.camera_frame, self.camera_seq
                    camera_received_at = getattr(self, "camera_received_at", None)
                try:
                    encoded_readers = self.video_hub.has_readers
                    image_readers = self.annotated_image_pub.get_subscription_count() > 0
                    if not encoded_readers and not image_readers:
                        self.video_encoder.reset()
                        cached, last_key = None, None
                        continue
                    if generation != last_generation:
                        self.video_encoder.reset()
                        self.video_hub.reset()
                        cached, last_key = None, None
                        last_generation = generation
                    start = time.monotonic()
                    if job is not None:
                        frame, detections, header, gen, seq, info, completed_at = job
                        if gen == generation:
                            boxes = getattr(self, "bbox3d_worker", None)
                            if boxes is not None:
                                detections, measured_info = boxes.latest(detections)
                                info = measured_info or info
                            visible = any(item.get("bbox_visible", True) and
                                          (self.bbox_visualization_mode == "2d" or
                                           ("bbox3d" in item and info is not None))
                                          for item in detections)
                            if source == "upload" or (visible and time.monotonic() - completed_at < 3.0):
                                pixels = render_display_overlay(
                                    frame.pixels, detections, info, self.bbox_visualization_mode)
                                cached = (pixels, header, ("result", gen, seq, completed_at),
                                          completed_at)
                            else:
                                cached = None
                    if cached is not None and (source == "upload" or
                                               time.monotonic() - cached[3] < 3.0):
                        pixels, header, key, completed_at = cached
                    elif source == "camera" and camera is not None:
                        key, completed_at = ("camera", generation, camera_seq), None
                        pixels, header = None, camera.header
                    else:
                        continue
                    requested_keyframe = self.video_hub.keyframe_requested.is_set()
                    force_keyframe = (requested_keyframe or
                                      time.monotonic() - last_keyframe >= 1.0)
                    if key == last_key:
                        if not force_keyframe:
                            continue
                        # A periodic IDR can use the next live RGB instead of
                        # re-encoding cached pixels between two result arrivals.
                        # Explicit viewer requests and paused inputs still resend.
                        if (source == "camera" and not requested_keyframe
                                and camera_received_at is not None
                                and time.monotonic() - camera_received_at < 0.2):
                            continue
                    if pixels is None:
                        rgb = self.bridge.imgmsg_to_cv2(camera, desired_encoding="rgb8")
                        pixels = render_display_overlay(rgb, [], mode=self.bbox_visualization_mode)
                    if header is None:
                        header = Header(stamp=self.get_clock().now().to_msg())
                    if image_readers:
                        with self.lock:
                            if generation != self.generation:
                                continue
                            # Replace queued frames instead of waiting for DDS in the renderer.
                            self.pending_annotated_image = (pixels, header, generation)
                            self.annotated_image_changed.set()
                    if not encoded_readers:
                        self.video_encoder.reset()
                        last_started, last_key = start, key
                        continue
                    # Clear before encode so requests arriving during encode survive.
                    self.video_hub.keyframe_requested.clear()
                    encode_start = time.monotonic()
                    unit = self.video_encoder.encode(pixels, header, generation, force_keyframe)
                    finished = time.monotonic()
                    with self.lock:
                        if generation != self.generation:
                            self.video_encoder.reset()
                            continue
                        self.video_hub.publish(unit)
                        self.video_at, self.video_error = finished, None
                        self.performance.update(
                            video_encode_ms=(finished - encode_start) * 1000,
                            video_render_ms=(encode_start - start) * 1000,
                            video_encoder=self.video_encoder.codec,
                            video_packet_bytes=len(unit.data),
                        )
                        if completed_at is not None and self.performance.get("completed_at") == completed_at:
                            self.performance["render_ms"] = (finished - start) * 1000
                            self.performance["result_to_overlay_ms"] = (finished - completed_at) * 1000
                    last_started, last_key = start, key
                    if unit.keyframe:
                        last_keyframe = finished
                except Exception as error:
                    self.video_encoder.reset()
                    self.video_hub.reset()
                    with self.lock:
                        self.video_error = str(error)
                    self.get_logger().error(f"H.264 rendering failed: {error}")
                    self.stop.wait(0.5)
                finally:
                    if job is not None:
                        job[0].release()
        finally:
            self.video_encoder.reset()

    def run_image_publications(self):
        """Publish only the newest raw overlay without blocking rendering or tracking."""
        while not self.stop.is_set():
            self.annotated_image_changed.wait(0.2)
            self.annotated_image_changed.clear()
            with self.lock:
                job, self.pending_annotated_image = self.pending_annotated_image, None
                generation = self.generation
            if job is None:
                continue
            pixels, header, gen = job
            if gen != generation or not self.annotated_image_pub.get_subscription_count():
                continue
            try:
                message = self.bridge.cv2_to_imgmsg(pixels, encoding="bgr8")
                message.header = header
                with self.lock:
                    if gen != self.generation:
                        continue
                # DDS may wait for a slow reader; never hold the shared state lock here.
                self.annotated_image_pub.publish(message)
            except Exception as error:
                self.get_logger().error(f"Raw image publication failed: {error}")

    def run_video_publications(self):
        """DDS cannot block the encoder or the WebRTC viewers."""
        reader = None
        try:
            while not self.stop.is_set():
                if not self.h264_pub.get_subscription_count():
                    if reader is not None:
                        reader.close()
                        reader = None
                    self.stop.wait(0.1)
                    continue
                if reader is None:
                    reader = self.video_hub.subscribe()
                unit = reader.get(timeout=0.2)
                if unit is None:
                    continue
                with self.lock:
                    if unit.generation != self.generation:
                        continue
                message = CompressedImage(header=unit.header, format="h264", data=unit.data)
                try:
                    self.h264_pub.publish(message)
                except Exception as error:
                    reader.reset()
                    self.get_logger().error(f"H.264 publication failed: {error}")
        finally:
            if reader is not None:
                reader.close()

    def close_grounding_worker(self):
        detector = getattr(self, "grounding_detector", None)
        close = getattr(detector, "close", None)
        if close is not None:
            close()
        proposals = getattr(self, "proposal_tracker", None)
        if proposals is not None:
            proposals.close()
        boxes = getattr(self, "bbox3d_worker", None)
        if boxes is not None:
            boxes.close()

    def destroy_node(self):
        self.stop.set()
        self.close_grounding_worker()
        self.video_hub.close()
        self.webrtc.close()
        self.render_changed.set()
        self.annotated_image_changed.set()
        self.image_publisher_worker.join(timeout=3)
        self.renderer_worker.join(timeout=3)
        self.video_publisher_worker.join(timeout=3)
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
                self.status = "Loading official SAM3 model…"
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
        completed_frame = None
        last_rgb_completed_at = None
        last_rgb_context = None
        next_upload_refresh = 0.0
        proposal_tracker = AsyncNvDCFProposals(
            wake=self.image_changed.set,
            config=self.param("nvdcf_config"),
            width=self.param("nvdcf_width"),
            height=self.param("nvdcf_height"),
            containment_threshold=self.param("grounding_containment_threshold"),
            max_missed_detections=self.param("nvdcf_max_missed_detections"),
            max_missed_frames=self.param("nvdcf_max_missed_frames"),
            defer_recognition=True,
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
                if processed == (gen, seq):
                    # Fold background completions into the next live RGB. Do not
                    # recopy, retrack, publish or encode an already processed frame.
                    # Flush after 200 ms without input so paused cameras still get
                    # their first 3D boxes and completed labels; uploads flush now.
                    if (self.source == "camera" and received_at is not None
                            and time.monotonic() - received_at < 0.2):
                        continue
                    if (not proposal_tracker.has_result()
                            and not (self.param("pose_enabled") and isinstance(frame, ImageMsg)
                                     and self.bbox3d_worker.has_result())):
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
                    limit=self.param("grounding_max_detections"),
                    min_width=self.param("grounding_min_bbox_width"),
                    min_height=self.param("grounding_min_bbox_height"),
                    min_area=self.param("grounding_min_bbox_area"),
                    # Containment runs once on the final merged NvDCF tracks.
                    containment_threshold=0.0,
                )

                def detect(frame, request=request):
                    started_at = time.monotonic()
                    proposals = detector.detect(frame, **request)
                    with self.lock:
                        self.sam3_event_sequence += 1
                        self.sam3_events.append(dict(
                            sequence=self.sam3_event_sequence, started_at=started_at,
                        ))
                    return proposals, dict(getattr(detector, "last_metrics", {}))

                def recognize(seed_pixels, proposals, seed_key, settings=settings):
                    # Geometry is published before this background callback finishes.
                    # All gallery GPU work and diagnostics stay off the NvDCF loop.
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
                # Classification fields follow individual proposals through NvDCF,
                # including partial loss. Never zip survivors with an old label list.
                detections = proposals
                recognition = proposal_tracker.last_recognition
                recognition_info = recognition or dict(
                    frame_id=None, snapshot=None, class_count=len(set(gallery.labels)),
                    reference_count=len(gallery.labels), recognize_ms=0.0)
                recognition_changed = (recognition is not last_recognition
                                       or (had_detections and not detections))
                had_detections = bool(detections)
                last_recognition = recognition
                recognition_frame_id = recognition_info["frame_id"] if detections else None
                recognition_source = ("none" if not detections else
                                      "dino" if recognition_changed else "cached")
                if any(item.get("recognition_pending") for item in detections):
                    recognition_source = "pending"
                recognized_at = time.monotonic()
                with self.lock:
                    if gen != self.generation:
                        continue
                    for item in detections:
                        item["bbox_visible"] = True
                bbox3d_schedule = None
                bbox3d_info = depth_inputs[1] if depth_inputs else None
                if self.param("pose_enabled") and header is not None:
                    if depth_inputs is None:
                        depth_inputs = self.depth_inputs_for(frame)
                    detections, measured_info, bbox3d_schedule = self.bbox3d_worker.update(
                        detections, depth_inputs, header=header,
                        frame_key=(gen, seq), context=context, captured_at=received_at,
                    )
                    bbox3d_info = measured_info or bbox3d_info
                else:
                    for item in detections:
                        item["bbox3d_status"] = "disabled" if not self.param("pose_enabled") else "no_camera_depth"
                bbox3d_at = time.monotonic()
                # Diagnostic crops/features stay on their recognition frame;
                # never relabel an old crop as the current NvDCF image.
                object_snapshot = (recognition_info["snapshot"]
                                   if recognition_changed and detections else None)
                with self.lock:
                    if gen != self.generation:
                        continue
                    if recognition_changed:
                        self.object_debug_revision += 1
                        self.object_debug_snapshot = object_snapshot
                    self.detections = detections
                    completed_at = time.monotonic()
                    new_rgb_result = completed_frame != (gen, seq)
                    result_fps = self.performance.get("result_fps")
                    result_fps_instant = None
                    if new_rgb_result:
                        if last_rgb_context == context and last_rgb_completed_at is not None:
                            result_fps_instant = 1 / max(1e-9, completed_at - last_rgb_completed_at)
                        result_fps = result_rate.update(completed_at, context)
                        completed_frame = (gen, seq)
                        last_rgb_completed_at, last_rgb_context = completed_at, context
                    self.publication_drops = getattr(self, "publication_drops", 0) + int(
                        self.pending_publication is not None)
                    self.performance = dict(
                        completed_at=completed_at,
                        result_fps=result_fps,
                        result_fps_window_seconds=result_rate.window_seconds,
                        result_fps_instant=result_fps_instant,
                        new_rgb_result=new_rgb_result,
                        grounding_ipc=(dict(proposal_tracker.last_metrics)
                                       if proposal_source == "sam3" else {}),
                        grounding_schedule=proposal_tracker.scheduler_metrics(),
                        ros_publication=getattr(self, "last_publication", None),
                        bbox_source=proposal_source,
                        recognition_source=recognition_source,
                        recognition_frame_id=recognition_frame_id,
                        detection_interval=settings["detection_interval"],
                        image_prepare_ms=(image_ready_at - start) * 1000,
                        shared_copy_ms=shared_frame.copy_ms,
                        depth_wait_ms=depth_wait_ms,
                        proposal_update_ms=(detected_at - depth_ready_at) * 1000,
                        detect_ms=(detected_at - start) * 1000,
                        recognize_ms=(recognition_info["recognize_ms"] if recognition_changed else 0.0),
                        bbox3d_ms=(bbox3d_at - recognized_at) * 1000,
                        bbox3d_count=sum("bbox3d" in item for item in detections),
                        bbox3d_schedule=bbox3d_schedule,
                        render_ms=None,
                        frame_id=seq,
                        input_skipped_frames=(max(0, seq-processed[1]-1)
                                              if processed and processed[0] == gen else 0),
                        publication_drops=self.publication_drops,
                        processing_ms=(completed_at - start) * 1000,
                        callback_to_result_ms=(
                            None if received_at is None else (completed_at - received_at) * 1000
                        ),
                    )
                    self.status = (
                        f"{('SAM3 Torch-TensorRT' if proposal_source == 'sam3' else 'NvDCF')} → "
                        f"{encoder.description}{'' if recognition_changed else ' (cached)'} | {len(detections)} objects | "
                        f"{recognition_info['class_count']} classes / {recognition_info['reference_count']} references | "
                        f"{time.monotonic()-start:.2f}s inference"
                    )
                    self.pending_publication = (detections, header, gen)
                    self.pending_publication_timing = dict(
                        frame_id=seq, received_at=received_at, completed_at=completed_at)
                    if self.pending_render is not None:
                        self.pending_render[0].release()
                    self.pending_render = (shared_frame.retain(), detections, header, gen,
                                           seq, bbox3d_info, completed_at)
                    self.render_changed.set()
                    self.publish_changed.set()
                processed = (gen, seq)
            except Exception as error:
                with self.lock:
                    if gen == self.generation:
                        self.clear_object_debug()
                        self.detections = []
                        self.video_at = None
                        self.video_hub.reset()
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
        node.publish_changed.set()
        node.render_changed.set()
        node.server.shutdown()
        node.close_grounding_worker()
        node.worker.join(timeout=10)
        node.publisher_worker.join(timeout=2)
        node.video_hub.close()
        node.webrtc.close()
        node.renderer_worker.join(timeout=3)
        node.video_publisher_worker.join(timeout=3)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
