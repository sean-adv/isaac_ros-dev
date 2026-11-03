"""ROS input, inference scheduling and result publication with official Torch-TensorRT models."""

import json
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image as ImageMsg, CameraInfo
from std_msgs.msg import Header, String
from werkzeug.serving import make_server

from dinov3_cases.timing import ResultRate, ModelTimings
from dinov3_cases.inference.patches import feature_stride
from dinov3_cases.inference.encoder import Encoder
from dinov3_cases.inference.grounding_ipc import SharedMemoryGroundingDetector
from dinov3_cases.frame_pool import SharedFramePool
from dinov3_cases.tracking.identities import BBoxSmoother
from dinov3_cases.config import default_parameters, validate_parameters
from dinov3_cases.tracking.async_nvdcf import AsyncNvDCFProposals
from dinov3_cases.geometry.async_bbox3d import AsyncBBox3D
from dinov3_cases.runtime import INPUT_SIZE
from vision_msgs.msg import Detection2DArray, Detection3DArray
from isaac_ros_apriltag_interfaces.msg import AprilTagDetectionArray
from dinov3_cases.ros.messages import apriltag_message
from dinov3_cases.geometry.depth_sync import matched_inputs
from dinov3_cases.inference.sam3_roi import DEFAULT_ROI, validate_roi, roi_bounds, filter_roi
from dinov3_cases.api.image_stream import ImageStream
from dinov3_cases.geometry.cloud_filter import CloudFilter
from dinov3_cases.geometry.depth_bbox import ProjectionRayCache, attach_bboxes3d
from dinov3_cases.ros.messages import bboxes3d_message
from dinov3_cases.api.web import create_app
from dinov3_cases.api.visualization import render_display_overlay
from dinov3_cases.recognition.object_visualization import ObjectVisualizationSnapshot
from dinov3_cases.recognition.grounded import ObjectGallery, recognize_objects
from dinov3_cases.ros.messages import recognition_message


class CaseNode(Node):
    def __init__(self):
        super().__init__("dinov3_cases")
        defaults = default_parameters()
        for k, v in defaults.items():
            self.declare_parameter(k, v)
        self.param = lambda k: self.get_parameter(k).value
        self.sam3_roi = validate_roi(*(self.param("sam3_roi_" + key)
                                     for key in ("width", "height", "center_x", "center_y")))
        self.grounding_settings = validate_parameters(self.param)
        self.bbox_visualization_mode = self.param("bbox_visualization_mode")
        # Set the global OpenCV pool before starting inference/render workers.
        opencv_threads = self.param("opencv_num_threads")
        if opencv_threads > 0:
            cv2.setNumThreads(opencv_threads)
        self.bbox_smoother = BBoxSmoother(self.param("bbox_smoothing_frames"))
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
        self.depth_frames = deque(maxlen=30)
        self.depth_infos = deque(maxlen=8)
        self.depth_ray_cache = ProjectionRayCache()
        self.performance = {}
        self.model_timings = ModelTimings()
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
        self.model_timings_pub = self.create_publisher(
            String, "/medicine_recognition/model_timings",
            QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                       reliability=ReliabilityPolicy.RELIABLE),
        )
        self.model_timings_timer = self.create_timer(0.2, self.publish_model_timings)
        self.bbox3d_pub = self.create_publisher(
            Detection3DArray, "/medicine_recognition/detections_3d", 10
        )
        self.tag_pub = self.create_publisher(
            AprilTagDetectionArray, self.param("tag_topic"), 10
        )
        self.annotated_image_pub = self.create_publisher(
            ImageMsg, self.param("annotated_image_topic"),
            QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                       reliability=ReliabilityPolicy.RELIABLE),
        )
        self.image_stream = ImageStream(
            self.bridge, wake=self.render_changed.set,
            on_error=lambda error: self.get_logger().error(f"JPEG stream failed: {error}"),
        )
        # The browser bridge consumes the same ROS Image topic as RViz.
        self.annotated_image_subscription = self.create_subscription(
            ImageMsg, self.param("annotated_image_topic"), self.image_stream.update,
            QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                       reliability=ReliabilityPolicy.RELIABLE),
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
        self.image_publisher_worker = threading.Thread(target=self.run_image_publications, daemon=True)
        self.image_publisher_worker.start()
        self.publisher_worker = threading.Thread(target=self.run_publications, daemon=True)
        self.publisher_worker.start()
        self.renderer_worker = threading.Thread(target=self.run_rendering, daemon=True)
        self.renderer_worker.start()
        plane_backend = self.param("bbox3d_plane_backend")
        self.plane_fitter = None
        if self.param("pose_enabled") and plane_backend == "cuda":
            from dinov3_cases.gpu_geometry import CudaPlaneFitter
            self.plane_fitter = CudaPlaneFitter()
        self.bbox3d_worker = AsyncBBox3D(
            self.estimate_bboxes3d, interval=self.param("bbox3d_interval"),
            max_age=self.param("bbox3d_max_age"), wake=self.image_changed.set,
            on_error=lambda error: self.get_logger().error(f"Background 3D boxes failed: {error}"),
        )
        self.worker = threading.Thread(target=self.run_grounded_worker, daemon=True)
        self.worker.start()
        self.get_logger().info(f"Browser: http://{self.param('host')}:{self.param('port')}")

    def publish_model_timings(self):
        payload = self.model_timings.milliseconds()
        self.model_timings_pub.publish(String(data=json.dumps(payload, allow_nan=False)))

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
        with self.lock:
            self.depth_frames.append(msg)

    def on_depth_info(self, msg):
        with self.lock:
            self.depth_infos.append(msg)

    def depth_inputs_for(self, frame):
        """Retain a matching depth snapshot without waiting in the RGB loop."""
        if not self.param("pose_enabled") or not isinstance(frame, ImageMsg):
            return None
        with self.lock:
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
                plane_fitter=getattr(self, "plane_fitter", None),
                cloud_filter=self.cloud_filter,
                cloud_frame=(depth_msg.header.frame_id, depth_msg.header.stamp.sec,
                             depth_msg.header.stamp.nanosec),
            )
        except (ValueError, CvBridgeError) as error:
            for item in detections:
                item["bbox3d_status"] = str(error)

    def worker_ticks(self):
        """Wake on input and periodically recheck non-image changes."""
        while not self.stop.is_set():
            self.image_changed.wait(0.1)  # Also recheck uploads, settings and gallery changes.
            self.image_changed.clear()
            if self.stop.is_set():
                return
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
        """Draw once for the shared ROS Image output; keep the latest pending frame."""
        cached = None
        last_key, last_generation = None, None
        last_started = 0.0
        while not self.stop.is_set():
            self.render_changed.wait(0.2)
            self.render_changed.clear()
            delay = last_started + 1 / self.param("video_fps") - time.monotonic()
            if delay > 0 and self.stop.wait(delay):
                break
            with self.lock:
                job, self.pending_render = self.pending_render, None
                generation, source = self.generation, self.source
                camera, camera_seq = self.camera_frame, self.camera_seq
            try:
                # Exclude our own HTTP bridge subscription when no browser is watching.
                if (self.annotated_image_pub.get_subscription_count() <= 1
                        and not self.image_stream.has_readers):
                    cached, last_key = None, None
                    continue
                if generation != last_generation:
                    cached, last_key = None, None
                    last_generation = generation
                start = time.monotonic()
                render = render_display_overlay
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
                            pixels = render(
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
                # No codec refresh is needed; static frames stay in the HTTP JPEG cache.
                if key == last_key and not self.image_stream.needs_frame:
                    continue
                if pixels is None:
                    rgb = self.bridge.imgmsg_to_cv2(camera, desired_encoding="rgb8")
                    pixels = render(rgb, [], mode=self.bbox_visualization_mode)
                if header is None:
                    header = Header(stamp=self.get_clock().now().to_msg())
                finished = time.monotonic()
                with self.lock:
                    if generation != self.generation:
                        continue
                    self.pending_annotated_image = (pixels, header, generation)
                    self.annotated_image_changed.set()
                    self.video_at, self.video_error = finished, None
                    self.performance["video_render_ms"] = (finished - start) * 1000
                    if completed_at is not None and self.performance.get("completed_at") == completed_at:
                        self.performance["render_ms"] = (finished - start) * 1000
                        self.performance["result_to_overlay_ms"] = (finished - completed_at) * 1000
                last_started, last_key = start, key
            except Exception as error:
                with self.lock:
                    self.video_error = str(error)
                self.get_logger().error(f"Image rendering failed: {error}")
                self.stop.wait(0.5)
            finally:
                if job is not None:
                    job[0].release()

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
        self.image_stream.close()
        self.render_changed.set()
        self.annotated_image_changed.set()
        self.image_publisher_worker.join(timeout=3)
        self.renderer_worker.join(timeout=3)
        with self.lock:
            if self.pending_render is not None:
                self.pending_render[0].release()
                self.pending_render = None
        self.frame_pool.close()
        fitter = getattr(self, "plane_fitter", None)
        if fitter is not None:
            fitter.close()
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
            model_timings=self.model_timings,
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
                    region = getattr(self, "sam3_roi", DEFAULT_ROI)
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
                bounds = (roi_bounds(*image_size, region)
                          if header is not None and region[:2] != (0, 0) else None)
                context = (gen, image_size, header.frame_id if header is not None else None)
                timing_revision = self.model_timings.set_context((context, tuple(settings.items())))
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

                def detect(frame, request=request, frame_key=(gen, seq), revision=timing_revision,
                           bounds=bounds):
                    started_at = time.monotonic()
                    with self.model_timings.measure("sam3", frame_key, revision) as timing:
                        proposals = detector.detect(frame, **request)
                        timing["duration_ms"] = detector.last_metrics["worker_detect_ms"]
                    with self.lock:
                        self.sam3_event_sequence += 1
                        self.sam3_events.append(dict(
                            sequence=self.sam3_event_sequence, started_at=started_at,
                        ))
                    return filter_roi(proposals, bounds), dict(getattr(detector, "last_metrics", {}))

                def recognize_impl(seed_pixels, proposals, seed_key, settings=settings):
                    # Geometry is published before this background callback finishes.
                    # All gallery GPU work and diagnostics stay off the NvDCF loop.
                    started = time.monotonic()
                    gallery.refresh(self.root, encoder, max_side, stride)
                    diagnostics = []
                    detections = recognize_objects(
                        Image.fromarray(seed_pixels), proposals, encoder, gallery, max_side, stride,
                        settings["recognition_threshold"], diagnostics=diagnostics,
                        defer_diagnostics=True,
                    )
                    snapshot = ObjectVisualizationSnapshot(
                        [dict(item, bbox=list(item["bbox"])) for item in detections],
                        diagnostics, gallery, max_side, stride,
                    )
                    return detections, dict(
                        frame_id=seed_key[1] if proposals else None,
                        snapshot=snapshot,
                        class_count=len(set(gallery.labels)),
                        reference_count=len(gallery.labels),
                        recognize_ms=(time.monotonic() - started) * 1000,
                    )

                def recognize(seed_pixels, proposals, seed_key, revision=timing_revision,
                              implementation=recognize_impl):
                    with self.model_timings.measure("dinov3", seed_key, revision):
                        return implementation(seed_pixels, proposals, seed_key)

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
                # A track may leave the region between SAM3 refreshes.
                proposals = filter_roi(proposals, bounds)
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
                        self.image_stream.reset()
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
        node.image_stream.close()
        node.renderer_worker.join(timeout=3)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
