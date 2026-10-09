"""Local ROS 2 / browser interface for grouped DINOv3 reference matching."""
import re
from io import BytesIO
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge, CvBridgeError
from flask import Flask, jsonify, request, Response
from PIL import Image, ImageOps
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image as ImageMsg
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from transformers import AutoModel, AutoImageProcessor
from werkzeug.serving import make_server

from .boxes import BoxConfig, similarity_to_bboxes, merge_boxes_by_iou, detection_message, draw_boxes
from .patches import (run_dino_image, compute_similarity_maps, similarity_pixels,
                      debug_visualizations, inference_dtype, feature_stride, prepare_reference,
                      pad_target_image, validate_target_padding, reference_grid_image)
from .trt_backend import DEFAULT_ENGINE, MODEL_ID as MODEL
from vision_msgs.msg import Detection2DArray
from .pose import closest_depth, estimate_pose, draw_axes
from .image_manager import register_image_manager, image_path
from .grounded import (DEFAULT_PROMPT, GroundingDetector, ObjectGallery, validate_settings,
                       recognize_objects, recognition_message, prepare_object, ObjectVisualizationSnapshot)

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


def case_path(root, name):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', name):
        raise ValueError('Case name must contain 1–80 letters, numbers, underscores or hyphens.')
    path = root / name
    if path.resolve().parent != root.resolve() or path.is_symlink():
        raise ValueError('Invalid case directory.')
    return path


def decode_image(stream):
    with Image.open(stream) as image:
        if image.width * image.height > 25_000_000:
            raise ValueError('Image exceeds 25 megapixels.')
        return ImageOps.exif_transpose(image).convert('RGB')


def render_heatmap(bgr, scores, threshold):
    color = cv2.applyColorMap((np.clip(scores, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    blended = cv2.addWeighted(bgr, .55, color, .45, 0)
    return np.where((scores >= threshold)[..., None], blended, bgr)


class Encoder:
    def __init__(self, model_id, backend='torch', engine_path='',
                 target_padding=0, target_padding_mode='reflect', square_input=False):
        if backend not in ('torch', 'trt'):
            raise ValueError('inference_backend must be torch or trt.')
        from .trt_backend import MODEL_VARIANTS
        cache_dir = next((spec.get('cache_dir') for spec in MODEL_VARIANTS.values()
                          if spec['model_id'] == model_id), None)
        self.square_input = square_input
        self.backend = backend
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if backend == 'trt':
            from .trt_backend import TRTBackbone
            self.model = TRTBackbone(engine_path, model_id)
            self.device = str(self.model.device)
            self.dtype = torch.float32  # The engine exposes FP32 I/O regardless of compute precision.
            self.description = self.model.description
        else:
            self.dtype = inference_dtype(self.device)
            self.model = AutoModel.from_pretrained(model_id, dtype=self.dtype, cache_dir=cache_dir).to(self.device).eval()
            self.description = f'PyTorch {str(self.dtype).removeprefix("torch.")}'
        validate_target_padding(target_padding, target_padding_mode, feature_stride(self.model))
        self.target_padding = target_padding
        self.target_padding_mode = target_padding_mode
        self.processor = AutoImageProcessor.from_pretrained(
            model_id, cache_dir=cache_dir, local_files_only=backend == 'trt')

    def spatial_features(self, image):
        pixels = torch.from_numpy(np.array(image, dtype=np.uint8)).permute(2, 0, 1).to(self.device)
        return run_dino_image(pixels, self.model, self.processor, self.dtype)

    @torch.inference_mode()
    def encode(self, image):
        spatial = self.spatial_features(image)
        return spatial[0].reshape(-1, spatial.shape[-1])

    @torch.inference_mode()
    def similarity_map(self, image, bank, debug=False):
        stride = feature_stride(self.model)
        padding = self.target_padding
        # Validate the TRT profile before allocating a padded image. Never resize
        # or silently disable padding to accommodate an out-of-profile target.
        if self.backend == 'trt':
            from .trt_backend import validate_shape
            try:
                validate_shape((1, 3, image.height + 2 * padding, image.width + 2 * padding),
                               self.model.minimum, self.model.maximum)
            except ValueError as error:
                raise ValueError(
                    f'Target {image.size} with target_padding={padding} is outside the TRT profile; '
                    'reduce target_padding (0 disables it), rebuild the engine profile, '
                    'or select inference_backend:=torch.') from error
        padded = pad_target_image(image, padding, self.target_padding_mode, stride)
        features = self.spatial_features(padded)
        offset = padding // stride
        gh, gw = image.height // stride, image.width // stride
        # Crop BEFORE matching/interpolation: padding tokens never enter output,
        # and incomplete bottom/right strips retain the existing -1 behavior.
        features = features[:, offset:offset + gh, offset:offset + gw]
        patch_scores = compute_similarity_maps(features, bank)[0]
        scores = similarity_pixels(patch_scores, (image.height, image.width), stride)
        raw = scores.cpu().numpy()
        if not np.isfinite(raw).all():
            raise RuntimeError(f'Non-finite DINOv3 similarity with {self.dtype}; heatmap and detections rejected.')
        return raw, tuple(patch_scores.shape), patch_scores.cpu().numpy() if debug else None

    def heatmap(self, image, bank):
        return self.similarity_map(image, bank)[0]


class CaseNode(Node):
    def __init__(self):
        super().__init__('dinov3_cases')
        defaults = dict(cases_dir='/workspaces/isaac_ros-dev/dinov3_cases_data',
                        pipeline='grounded', grounding_model_id='IDEA-Research/grounding-dino-tiny',
                        grounding_cache_dir='/workspaces/isaac_ros-dev/dinov3_ws/artifacts/huggingface',
                        grounding_backend='trt', grounding_execution='inprocess',
                        grounding_worker_python='/workspaces/isaac_ros-dev/dinov3_venv/bin/python',
                        grounding_ipc_timeout=60.0,
                        grounding_engine_path='/workspaces/isaac_ros-dev/dinov3_ws/artifacts/tensorrt/grounding_dino_wingdzero_fp16.engine',
                        grounding_prompt=DEFAULT_PROMPT, grounding_box_threshold=0.3,
                        grounding_text_threshold=0.25, grounding_nms_iou=0.5,
                        grounding_max_detections=20, recognition_threshold=0.75,
                        recognition_margin=0.05, object_max_side=224, object_square_input=False,
                        image_topic='/camera/color/image_raw', model_id=MODEL,
                        inference_backend='trt', engine_path=DEFAULT_ENGINE,
                        depth_topic='/camera/depth/image_raw',
                        camera_info_topic='/camera/color/camera_info',
                        depth_sync_tolerance=0.05, pose_depth_band=0.04, axis_length=0.05,
                        port=8090, host='127.0.0.1', inference_hz=0.0,
                        preview_max_width=960, jpeg_quality=80,
                        max_reference_patches=8192, cosine_threshold=0.65,
                        min_area=2048, bbox_padding=1, debug_visualization=False,
                        **{k: v for k, v in vars(BoxConfig()).items()
                           if k not in ('min_component_patches', 'padding_patches')})
        for k, v in defaults.items():
            self.declare_parameter(k, v)
        self.declare_parameter('reference_max_side', 0, ParameterDescriptor(
            read_only=True,
            description='Reference long-edge limit in pixels; 0 keeps original size. Restart to change.'))
        self.declare_parameter('target_padding', 64, ParameterDescriptor(
            read_only=True, description='Target context padding per side in pixels; '
            'must be a feature-stride multiple. 0 disables it. Restart to change.'))
        self.declare_parameter('target_padding_mode', 'reflect', ParameterDescriptor(
            read_only=True, description='Target padding: reflect or replicate. Restart to change.'))
        self.param = lambda k: self.get_parameter(k).value
        if self.param('pipeline') not in ('grounded', 'heatmap'):
            raise ValueError('pipeline must be grounded or heatmap.')
        self.grounding_settings = dict(
            prompt=self.param('grounding_prompt'), box_threshold=self.param('grounding_box_threshold'),
            text_threshold=self.param('grounding_text_threshold'),
            recognition_threshold=self.param('recognition_threshold'), margin=self.param('recognition_margin'))
        self.grounding_settings['prompt'] = validate_settings(**self.grounding_settings)
        if not 0 <= self.param('grounding_nms_iou') <= 1:
            raise ValueError('grounding_nms_iou must be between 0 and 1.')
        if not 1 <= self.param('grounding_max_detections') <= 100:
            raise ValueError('grounding_max_detections must be between 1 and 100.')
        if self.param('grounding_execution') not in ('inprocess', 'subprocess'):
            raise ValueError('grounding_execution must be inprocess or subprocess.')
        if self.param('grounding_ipc_timeout') <= 0:
            raise ValueError('grounding_ipc_timeout must be positive.')
        if not 32 <= self.param('object_max_side') <= 2048:
            raise ValueError('object_max_side must be between 32 and 2048.')
        validate_target_padding(self.param('target_padding'), self.param('target_padding_mode'))
        if self.param('reference_max_side') < 0:
            raise ValueError('reference_max_side must be nonnegative.')
        if not np.isfinite(self.param('inference_hz')) or self.param('inference_hz') < 0:
            raise ValueError('inference_hz must be finite and nonnegative; 0 is unlimited.')
        if not 1 <= self.param('jpeg_quality') <= 100 or self.param('preview_max_width') < 1:
            raise ValueError('jpeg_quality must be 1–100 and preview_max_width must be positive.')
        self.root = Path(self.param('cases_dir')).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.bridge = CvBridge()
        for key in ('depth_sync_tolerance', 'pose_depth_band', 'axis_length'):
            if not np.isfinite(self.param(key)) or self.param(key) <= 0:
                raise ValueError(f'{key} must be finite and positive.')
        self.lock = threading.Lock()
        self.image_changed = threading.Event()
        self.preview_changed = threading.Event()
        self.stream_changed = threading.Condition(self.lock)
        self.performance = {}
        self.pending_publication = None
        self.publish_changed = threading.Event()
        self.depth_frames = deque(maxlen=90)
        self.camera_info = None
        self.stop = threading.Event()
        self.threshold = float(self.param('cosine_threshold'))
        if not 0 <= self.threshold <= 1:
            raise ValueError('cosine_threshold must be between 0 and 1.')
        self.box_config = BoxConfig(**{key: self.param(key) for key in vars(BoxConfig())
                                       if key not in ('min_component_patches', 'padding_patches')})
        # Validate pixel post-processing parameters before starting worker threads.
        similarity_to_bboxes(np.zeros((1, 1), np.float32), self.threshold,
                             self.param('min_area'), self.param('bbox_padding'), self.box_config.closing_kernel)
        self.selected = '' 
        self.generation = 0
        self.source = 'camera'
        self.frame = None
        self.frame_seq = 0
        self.jpeg = None
        self.debug_jpegs = {}
        self.clear_object_debug()
        self.detections = []
        self.preview_jpeg = None
        self.camera_frame = None
        self.camera_seq = 0
        self.camera_received_at = None
        self.preview_error = None
        self.overlay_at = None
        self.status = 'Loading DINOv3 model…'
        self.backend_description = None
        self.reference_stride = None
        self.bank = None
        self.bank_key = None
        self.pub = self.create_publisher(ImageMsg, '/dinov3/heatmap/overlay', 1)
        self.raw_pub = self.create_publisher(ImageMsg, '/dinov3/heatmap/similarity', 1)
        self.recognition_pub = self.create_publisher(Detection2DArray, '/medicine_recognition/detections', 10)
        self.recognition_debug_pub = self.create_publisher(ImageMsg, '/medicine_recognition/debug_image', 1)
        self.pose_pub = self.create_publisher(PoseStamped, '/medicine_recognition/pose', 10)
        image_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                               reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(ImageMsg, self.param('image_topic'), self.on_image, image_qos)
        if self.param('pipeline') == 'heatmap':
            self.create_subscription(ImageMsg, self.param('depth_topic'), self.on_depth, qos_profile_sensor_data)
            self.create_subscription(CameraInfo, self.param('camera_info_topic'), self.on_camera_info, qos_profile_sensor_data)
        self.app = self.make_app()
        self.server = make_server(self.param('host'), self.param('port'), self.app, threaded=True)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.preview_worker = threading.Thread(target=self.run_preview, daemon=True)
        self.preview_worker.start()
        self.publisher_worker = threading.Thread(target=self.run_publications, daemon=True)
        self.publisher_worker.start()
        self.worker = threading.Thread(target=self.run_worker, daemon=True)
        self.worker.start()
        self.get_logger().info(f"Browser: http://{self.param('host')}:{self.param('port')}")

    def on_depth(self, msg):
        with self.lock:
            self.depth_frames.append(msg)

    def on_camera_info(self, msg):
        with self.lock:
            self.camera_info = msg

    def selected_pose(self, detections, mask, header, depth_frames, info):
        if not detections:
            return None, 'No detection'
        selected = max(detections, key=lambda item: item['score'])
        selected['selected'] = True
        selected['pose'] = None
        try:
            if header is None:
                raise ValueError('Uploaded RGB image has no paired depth')
            depth = closest_depth(depth_frames, header, self.param('depth_sync_tolerance'))
            if info is None or info.header.frame_id != header.frame_id:
                raise ValueError('Waiting for RGB camera calibration')
            if (info.height, info.width) != mask.shape:
                raise ValueError('Camera calibration dimensions do not match RGB')
            if info.distortion_model not in ('', 'plumb_bob', 'rational_polynomial'):
                raise ValueError('Unsupported camera distortion model')
            pose = estimate_pose(self.bridge.imgmsg_to_cv2(depth, desired_encoding='passthrough'),
                                 depth.encoding, info.k, info.d, selected['bbox'], mask,
                                 self.param('pose_depth_band'))
            pose['frame_id'] = header.frame_id
            pose['stamp'] = dict(sec=header.stamp.sec, nanosec=header.stamp.nanosec)
            selected['pose'] = pose
            return pose, 'Depth surface pose'
        except (ValueError, cv2.error, CvBridgeError) as error:
            selected['pose_error'] = str(error)
            return None, str(error)

    def on_image(self, msg):
        with self.lock:
            self.camera_frame = msg
            self.camera_seq += 1
            self.camera_received_at = time.monotonic()
            self.preview_changed.set()
            if self.source != 'camera':
                return
            self.frame = msg
            self.frame_seq += 1
            self.image_changed.set()

    def worker_ticks(self):
        """Wake on input; rate limit start-to-start, never sleep after processing."""
        deadline = 0.0
        while not self.stop.is_set():
            self.image_changed.wait(.1)  # Also recheck uploads, settings and gallery changes.
            self.image_changed.clear()
            if self.stop.wait(max(0., deadline - time.monotonic())):
                return
            hz = float(self.param("inference_hz"))
            deadline = time.monotonic() + (1 / hz if hz > 0 else 0)
            yield

    def run_publications(self):
        """DDS output cannot stall inference; pending output is a single latest slot."""
        while not self.stop.is_set():
            self.publish_changed.wait(.1)
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
                image_publishers = [pub for pub in (self.pub, self.recognition_debug_pub)
                                    if pub.get_subscription_count()]
                if image_publishers:
                    out = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
                    out.header = header
                    for publisher in image_publishers:
                        publisher.publish(out)
            except Exception as error:
                self.get_logger().error(f'ROS result publication failed: {error}')

    def run_preview(self):
        processed = -1
        while not self.stop.is_set():
            self.preview_changed.wait(.1)
            self.preview_changed.clear()
            with self.lock:
                frame, seq = self.camera_frame, self.camera_seq
            if frame is None or seq == processed:
                continue
            try:
                bgr = self.bridge.imgmsg_to_cv2(frame, desired_encoding='bgr8')
                width = max(1, int(self.param('preview_max_width')))
                if bgr.shape[1] > width:
                    bgr = cv2.resize(bgr, (width, round(bgr.shape[0] * width / bgr.shape[1])))
                ok, data = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, self.param('jpeg_quality')])
                if not ok:
                    raise ValueError('Could not encode camera frame')
                with self.lock:
                    self.preview_jpeg = data.tobytes()
                    self.preview_error = None
                    self.stream_changed.notify_all()
            except Exception as e:
                with self.lock:
                    self.preview_error = str(e)
            processed = seq

    def make_app(self):
        app = Flask(__name__)
        app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024

        @app.after_request
        def wake_after_update(response):
            if request.method != 'GET':
                self.image_changed.set()
                with self.stream_changed:
                    self.stream_changed.notify_all()
            return response

        def images_changed(group):
            with self.lock:
                if self.param('pipeline') == 'grounded' or group == self.selected:
                    self.generation += 1
                    self.jpeg = None
                    self.debug_jpegs = {}
                    self.clear_object_debug()
                    self.detections = []
                    self.overlay_at = None
                    self.status = 'Building reference features…'

        register_image_manager(app, self.root, images_changed)

        @app.errorhandler(ValueError)
        def bad_request(error):
            return jsonify(error=str(error)), 400

        @app.get('/')
        def index():
            return PAGE

        @app.get('/api/status')
        def status():
            cases = sorted(p.name for p in self.root.iterdir() if p.is_dir() and not p.is_symlink()
                           and re.fullmatch(r'[A-Za-z0-9_-]{1,80}', p.name))
            with self.lock:
                age = None if self.camera_received_at is None else time.monotonic() - self.camera_received_at
                return jsonify(cases=cases, selected=self.selected, source=self.source, status=self.status, threshold=self.threshold,
                               performance=self.performance, overlay_ready=self.jpeg is not None,
                               pipeline=self.param('pipeline'), grounding=self.grounding_settings,
                               grounding_model_id=self.param('grounding_model_id'),
                               grounding_backend=self.param('grounding_backend'),
                               grounding_execution=self.param('grounding_execution'),
                               grounding_worker=getattr(getattr(self, 'grounding_detector', None), 'worker_info', None),
                               grounding_engine_path=self.param('grounding_engine_path'),
                               grounding_text_token_limit=getattr(self, 'grounding_text_token_limit', None),
                               object_max_side=self.param('object_max_side'),
                               object_square_input=self.param('object_square_input'),
                               reference_revision=self.generation, reference_stride=self.reference_stride,
                               inference_backend=self.param('inference_backend'),
                               model_id=self.param('model_id'),
                               target_padding=self.param('target_padding'),
                               target_padding_mode=self.param('target_padding_mode'),
                               backend_description=self.backend_description,
                               camera_topic=self.param('image_topic'), camera_age=age,
                               camera_connected=age is not None and age < 3,
                               preview_error=self.preview_error, debug_visualization=self.param('debug_visualization'),
                               overlay_age=None if self.overlay_at is None else time.monotonic() - self.overlay_at)

        @app.get('/api/references')
        def references():
            name = request.args.get('case', '')
            folder = case_path(self.root, name)
            if not folder.is_dir():
                raise ValueError('Case does not exist.')
            with self.lock:
                stride = self.reference_stride
            if stride is None:
                return jsonify(error='模型載入中，稍後顯示 patch 網格。'), 503
            items = []
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() not in EXTENSIONS or not path.is_file() or path.is_symlink():
                    continue
                item = dict(name=path.name, version=f'{path.stat().st_mtime_ns}-{path.stat().st_size}')
                try:
                    original = decode_image(path)
                    reference = (prepare_object(original, self.param('object_max_side'), stride,
                                                square=self.param('object_square_input'))
                                 if self.param('pipeline') == 'grounded' else
                                 prepare_reference(original, self.param('reference_max_side'), stride))
                    item.update(original_size=list(original.size), size=list(reference.size),
                                rows=reference.height // stride, columns=reference.width // stride)
                except (ValueError, OSError, Image.DecompressionBombError) as error:
                    item['error'] = str(error)
                items.append(item)
            return jsonify(case=name, patch_size=stride, images=items)

        @app.get('/api/references/grid/<group>/<filename>')
        def reference_grid(group, filename):
            path = image_path(self.root, group, filename)
            with self.lock:
                stride = self.reference_stride
            if stride is None:
                return jsonify(error='模型載入中。'), 503
            reference = (prepare_object(decode_image(path), self.param('object_max_side'), stride,
                                        square=self.param('object_square_input'))
                         if self.param('pipeline') == 'grounded' else
                         prepare_reference(decode_image(path), self.param('reference_max_side'), stride))
            data = BytesIO()
            reference_grid_image(reference, stride).save(data, format='PNG')
            return Response(data.getvalue(), mimetype='image/png', headers={'Cache-Control': 'no-store'})

        @app.post('/api/grounding')
        def grounding_settings():
            payload = request.get_json(silent=True)
            if not isinstance(payload, dict) or not payload or set(payload) - set(self.grounding_settings):
                raise ValueError('Expected prompt and/or grounding/recognition thresholds.')
            with self.lock:
                settings = {**self.grounding_settings, **payload}
                settings['prompt'] = validate_settings(**settings)
                detector = getattr(self, 'grounding_detector', None)
                if detector is not None:
                    detector.validate_prompt(settings['prompt'])
                self.grounding_settings = settings
                self.generation += 1
                self.jpeg = None
                self.debug_jpegs = {}
                self.clear_object_debug()
                self.detections = []
                self.overlay_at = None
            return jsonify(ok=True, grounding=settings)

        @app.post('/api/threshold')
        def threshold():
            payload = request.get_json(silent=True)
            value = payload.get('threshold') if isinstance(payload, dict) else None
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError('Threshold must be a number between 0 and 1.')
            with self.lock:
                self.threshold = float(value)
                self.generation += 1
                self.jpeg = None
                self.debug_jpegs = {}
                self.clear_object_debug()
                self.detections = []
                self.overlay_at = None
            return jsonify(ok=True, threshold=self.threshold)

        @app.post('/api/cases')
        def upload():
            name = request.form.get('case', '')
            folder = case_path(self.root, name)
            files = request.files.getlist('images')
            if not files:
                raise ValueError('Choose at least one picture.')
            # Validate all pictures before modifying the case.
            try:
                images = [decode_image(f.stream) for f in files]
            except (OSError, Image.DecompressionBombError) as e:
                raise ValueError(f'Unsupported picture: {e}') from e
            folder.mkdir(exist_ok=True)
            for i, image in enumerate(images):
                dest = folder / f'{time.time_ns()}_{i}.png'
                temp = dest.with_suffix('.tmp')
                image.save(temp, format='PNG')
                temp.replace(dest)
            with self.lock:
                self.selected = name
                self.generation += 1
                self.jpeg = None
                self.debug_jpegs = {}
                self.clear_object_debug()
                self.detections = []
                self.status = 'Building reference features…'
            return jsonify(ok=True, count=len(images))

        @app.post('/api/select')
        def select():
            name = (request.get_json(silent=True) or {}).get('case', '')
            folder = case_path(self.root, name)
            if not folder.is_dir():
                raise ValueError('Case does not exist.')
            with self.lock:
                self.selected = name
                # Gallery selection is a preview action in the all-class pipeline.
                if self.param('pipeline') == 'heatmap':
                    self.generation += 1
                    self.jpeg = None
                    self.debug_jpegs = {}
                    self.clear_object_debug()
                    self.detections = []
                    self.status = 'Building reference features…'
            return jsonify(ok=True)

        @app.post('/api/source')
        def source():
            mode = request.form.get('source', 'camera')
            if mode not in ('camera', 'upload'):
                raise ValueError('Invalid source.')
            image = None
            if mode == 'upload':
                if 'image' not in request.files:
                    raise ValueError('Choose a target picture.')
                try:
                    image = decode_image(request.files['image'].stream)
                except (OSError, Image.DecompressionBombError) as e:
                    raise ValueError(f'Unsupported picture: {e}') from e
            with self.lock:
                self.source = mode
                self.frame = image
                self.frame_seq += 1
                self.generation += 1
                self.jpeg = None
                self.debug_jpegs = {}
                self.clear_object_debug()
                self.detections = []
            return jsonify(ok=True)

        def stream(attribute):
            def frames():
                previous = None
                while not self.stop.is_set():
                    with self.stream_changed:
                        self.stream_changed.wait_for(
                            lambda: self.stop.is_set() or getattr(self, attribute) is not previous,
                            timeout=.5)
                        data = getattr(self, attribute)
                    if data is None or data is previous:
                        previous = data
                        continue
                    previous = data
                    # No per-client frame queue: slow clients pick the newest result next.
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                           + str(len(data)).encode() + b'\r\n\r\n' + data + b'\r\n')
            return Response(frames(), mimetype='multipart/x-mixed-replace; boundary=frame',
                            headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})

        @app.get('/camera.mjpg')
        def camera_stream():
            return stream('preview_jpeg')

        @app.get('/overlay.mjpg')
        def overlay_stream():
            return stream('jpeg')

        @app.get('/api/detections')
        def detections():
            with self.lock:
                return jsonify(detections=self.detections)

        @app.get('/api/object_visualizations')
        def object_visualizations():
            reference_case = request.args.get('reference_case', '')
            with self.lock:
                revision, snapshot = self.object_debug_revision, self.object_debug_snapshot
                if snapshot is not None and not any(v == revision for v, _ in self.object_debug_history):
                    self.object_debug_history.append((revision, snapshot))
            if snapshot is None:
                return jsonify(revision=revision, objects=[], reference_cases=[], reference_case='', ready=False)
            try:
                items, _ = snapshot.render(reference_case)
            except ValueError as error:
                return jsonify(error=str(error), reference_cases=snapshot.reference_cases), 400
            return jsonify(revision=revision, objects=items, reference_cases=snapshot.reference_cases,
                           reference_case=reference_case, ready=True)

        @app.get('/debug/objects/<int:revision>/<int:index>/<view>.jpg')
        def object_debug_view(revision, index, view):
            with self.lock:
                snapshot = next((snapshot for version, snapshot in self.object_debug_history
                                 if version == revision), None)
                if snapshot is None:
                    return Response(status=410)
            _, images = snapshot.render(request.args.get('reference_case', ''))
            data = images.get((index, view))
            return Response(data or b'', status=200 if data else 404, mimetype='image/jpeg',
                            headers={'Cache-Control': 'no-store'})

        @app.get('/debug/<view>.jpg')
        def debug_view(view):
            if view not in ('grid', 'patches', 'heatmap', 'mask', 'boxes'):
                return Response(status=404)
            with self.lock:
                data = self.debug_jpegs.get(view)
            return Response(data or b'', status=200 if data else 204,
                            mimetype='image/jpeg', headers={'Cache-Control': 'no-store'})

        @app.get('/overlay.jpg')
        def overlay():
            with self.lock:
                data = self.jpeg
            return Response(data or b'', status=200 if data else 204,
                            mimetype='image/jpeg', headers={'Cache-Control': 'no-store'})
        return app

    def run_worker(self):
        if self.param('pipeline') == 'grounded':
            self.run_grounded_worker()
            return
        try:
            encoder = Encoder(self.param('model_id'), self.param('inference_backend'),
                              self.param('engine_path'), self.param('target_padding'),
                              self.param('target_padding_mode'))
        except Exception as e:
            with self.lock:
                self.status = f'Model load failed: {type(e).__name__}: {e}'
            self.get_logger().error(self.status)
            return
        with self.lock:
            self.backend_description = encoder.description
            self.reference_stride = feature_stride(encoder.model)
            self.status = f'{encoder.description} ready on {encoder.device}. Upload or select a case.'
        self.get_logger().info(self.status)
        processed = None
        for _ in self.worker_ticks():
            with self.lock:
                name, gen, frame, seq = self.selected, self.generation, self.frame, self.frame_seq
                threshold = self.threshold
                depth_frames, info = list(self.depth_frames), self.camera_info
            if not name or frame is None:
                with self.lock:
                    if gen == self.generation:
                        self.status = ('Select or upload a reference case.' if not name else
                                       f'Waiting for target image on {self.param("image_topic")}…')
                continue
            if processed == (gen, seq):
                continue
            try:
                folder = case_path(self.root, name)
                paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS
                               and p.is_file() and not p.is_symlink())
                reference_max_side = self.param('reference_max_side')
                key = (name, reference_max_side,
                       tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in paths))
                if key != self.bank_key:
                    if not paths:
                        raise ValueError('Selected case has no supported pictures.')
                    limit = self.param('max_reference_patches')
                    if limit < len(paths):
                        raise ValueError(f'Too many pictures; maximum is {limit}.')
                    per_image = max(1, limit // len(paths))
                    banks = []
                    for path in paths:
                        reference = prepare_reference(decode_image(path), reference_max_side,
                                                     feature_stride(encoder.model))
                        features = encoder.encode(reference)
                        indices = torch.linspace(0, len(features) - 1, min(per_image, len(features)),
                                                 device=features.device).long()
                        banks.append(features[indices])
                    self.bank = torch.cat(banks)
                    self.bank_key = key
                header = None
                if isinstance(frame, ImageMsg):
                    image = Image.fromarray(self.bridge.imgmsg_to_cv2(frame, desired_encoding='rgb8'))
                    header = frame.header
                else:
                    image = frame
                start = time.monotonic()
                scores, patch_grid, patch_scores = encoder.similarity_map(
                    image, self.bank, self.param('debug_visualization'))
                bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                detections, mask = similarity_to_bboxes(
                    scores, threshold, self.param('min_area'), self.param('bbox_padding'),
                    self.box_config.closing_kernel, bgr=bgr, cfg=self.box_config, return_mask=True)
                detections = merge_boxes_by_iou(detections, self.box_config.merge_iou_threshold)
                pose, pose_status = self.selected_pose(detections, mask, header, depth_frames, info)
                boxes = [(item['bbox'], item['score']) for item in detections]
                recognition = detection_message(boxes, name, header)
                debug_jpegs = {}
                if patch_scores is not None:
                    views = debug_visualizations(bgr, patch_scores, scores, mask,
                                                 detections, feature_stride(encoder.model))
                    for view, pixels in views.items():
                        encoded, jpeg = cv2.imencode('.jpg', pixels)
                        if not encoded:
                            raise RuntimeError(f'Could not encode debug view: {view}')
                        debug_jpegs[view] = jpeg.tobytes()
                overlay = render_heatmap(bgr, scores, threshold)
                draw_boxes(overlay, boxes)
                if pose is not None:
                    draw_axes(overlay, pose, info.k, info.d, self.param('axis_length'))
                    x, y, z = pose['position']
                    pose_status = f'Surface XYZ: {x:.3f}, {y:.3f}, {z:.3f} m'
                cv2.putText(overlay, pose_status, (12, 52), cv2.FONT_HERSHEY_SIMPLEX,
                            .5, (255, 255, 255), 1)
                cv2.putText(overlay, f'{name} | cosine >= {threshold:.2f} | max {scores.max():.2f}',
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 2)
                ok, data = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, self.param('jpeg_quality')])
                if not ok:
                    raise RuntimeError('JPEG encoding failed.')
                with self.lock:
                    if gen != self.generation:
                        continue
                    self.jpeg = data.tobytes()
                    self.debug_jpegs = debug_jpegs
                    self.detections = detections
                    self.overlay_at = time.monotonic()
                    self.stream_changed.notify_all()
                    self.status = f'{name}: {len(paths)} references | {patch_grid[0]}×{patch_grid[1]} patches / batch 1 | {encoder.device} {encoder.description} | {time.monotonic()-start:.2f}s inference'
                out = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
                raw = self.bridge.cv2_to_imgmsg(scores.astype(np.float32), encoding='32FC1')
                if header is not None:
                    out.header = header
                    raw.header = header
                else:
                    out.header.stamp = self.get_clock().now().to_msg()
                    raw.header = out.header
                self.pub.publish(out)
                self.raw_pub.publish(raw)
                recognition.header = out.header
                for detection in recognition.detections:
                    detection.header = out.header
                self.recognition_pub.publish(recognition)
                if pose is not None:
                    pose_msg = PoseStamped()
                    pose_msg.header = out.header
                    pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = pose['position']
                    q = pose_msg.pose.orientation
                    q.x, q.y, q.z, q.w = pose['quaternion_xyzw']
                    self.pose_pub.publish(pose_msg)
                self.recognition_debug_pub.publish(out)
                processed = (gen, seq)
            except Exception as e:
                with self.lock:
                    if gen == self.generation:
                        self.jpeg = None
                        self.debug_jpegs = {}
                        self.clear_object_debug()
                        self.detections = []
                        self.status = f'Error: {type(e).__name__}: {e}'
                self.get_logger().error(str(e))
                processed = (gen, seq)

    def close_grounding_worker(self):
        detector = getattr(self, 'grounding_detector', None)
        close = getattr(detector, 'close', None)
        if close is not None:
            close()

    def destroy_node(self):
        self.stop.set()
        self.close_grounding_worker()
        return super().destroy_node()

    def clear_object_debug(self):
        self.object_debug_revision = getattr(self, 'object_debug_revision', 0) + 1
        self.object_debug_items = []
        self.object_debug_jpegs = {}
        self.object_debug_snapshot = None
        self.object_debug_history = deque(maxlen=3)

    def run_grounded_worker(self):
        try:
            encoder = Encoder(self.param('model_id'), self.param('inference_backend'),
                              self.param('engine_path'), 0, self.param('target_padding_mode'),
                              square_input=self.param('object_square_input'))
            stride = feature_stride(encoder.model)
            with self.lock:
                self.reference_stride = stride
                self.backend_description = encoder.description
                self.status = 'Loading Grounding DINO model…'
            detector_class = GroundingDetector
            detector_options = {}
            if self.param('grounding_execution') == 'subprocess':
                from .grounding_ipc import SharedMemoryGroundingDetector
                detector_class = SharedMemoryGroundingDetector
                detector_options = dict(python=self.param('grounding_worker_python'),
                                        timeout=self.param('grounding_ipc_timeout'))
            detector = detector_class(self.param('grounding_model_id'), encoder.device,
                                         self.param('grounding_cache_dir') or None,
                                         self.param('grounding_backend'), self.param('grounding_engine_path'),
                                         **detector_options)
            self.grounding_detector = detector
            with self.lock:
                detector.validate_prompt(self.grounding_settings['prompt'])
                self.grounding_detector = detector
                config = getattr(getattr(detector, 'model', None), 'config', None)
                self.grounding_text_token_limit = getattr(config, 'max_text_len', None)
        except Exception as error:
            self.close_grounding_worker()
            with self.lock:
                self.status = f'Model load failed: {type(error).__name__}: {error}'
            self.get_logger().error(self.status)
            return
        gallery = ObjectGallery()
        processed = None
        next_gallery_scan = 0.
        gallery_revision = 0
        max_side = self.param('object_max_side')
        for _ in self.worker_ticks():
            with self.lock:
                gen, frame, seq = self.generation, self.frame, self.frame_seq
                settings = dict(self.grounding_settings)
                received_at = self.camera_received_at if self.source == 'camera' else None
            try:
                # Include filesystem changes even with an uploaded, otherwise static target.
                if time.monotonic() >= next_gallery_scan or processed is None or processed[0] != gen:
                    if gallery.refresh(self.root, encoder, max_side, stride):
                        gallery_revision += 1
                    next_gallery_scan = time.monotonic() + 1.
                with self.lock:
                    if gen != self.generation:
                        continue
                    gen, frame, seq = self.generation, self.frame, self.frame_seq
                    settings = dict(self.grounding_settings)
                    received_at = self.camera_received_at if self.source == 'camera' else None
                if frame is None:
                    with self.lock:
                        if gen == self.generation:
                            self.status = f'Ready: {len(set(gallery.labels))} classes. Waiting for target image…'
                    continue
                if processed == (gen, seq, gallery_revision):
                    continue
                start = time.monotonic()
                header = None
                if isinstance(frame, ImageMsg):
                    image = Image.fromarray(self.bridge.imgmsg_to_cv2(frame, desired_encoding='rgb8'))
                    header = frame.header
                else:
                    image = frame
                proposals = detector.detect(image, settings['prompt'], settings['box_threshold'],
                                            settings['text_threshold'], self.param('grounding_nms_iou'),
                                            self.param('grounding_max_detections'))
                detected_at = time.monotonic()
                diagnostics = []
                detections = recognize_objects(image, proposals, encoder, gallery, max_side, stride,
                                               settings['recognition_threshold'], settings['margin'],
                                               diagnostics=diagnostics, defer_diagnostics=True)
                recognized_at = time.monotonic()
                object_snapshot = ObjectVisualizationSnapshot(detections, diagnostics, gallery, max_side, stride)
                overlay = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                for index, item in enumerate(detections):
                    x0, y0, x1, y1 = item['bbox']
                    color = (0, 190, 255) if item['label'] == 'unknown' else (60, 220, 80)
                    cv2.rectangle(overlay, (x0, y0), (x1-1, y1-1), color, 2)
                    score = '--' if item['score'] is None else f"{item['score']:.2f}"
                    label = f"#{index+1} {item['label']} | {item['detection_label']} | det {item['detection_score']:.2f} sim {score}"
                    font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, .8, 2
                    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                    origin = (max(4, min(x0, overlay.shape[1]-text_width-4)),
                              max(text_height+6, y0-8))
                    cv2.putText(overlay, label, origin, font, font_scale, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(overlay, label, origin, font, font_scale, color, thickness, cv2.LINE_AA)
                ok, jpeg = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, self.param('jpeg_quality')])
                if not ok:
                    raise RuntimeError('JPEG encoding failed.')
                with self.lock:
                    if gen != self.generation:
                        continue
                    self.jpeg = jpeg.tobytes()
                    self.debug_jpegs = {'boxes': self.jpeg}
                    self.object_debug_revision += 1
                    self.object_debug_snapshot = object_snapshot
                    self.detections = detections
                    self.overlay_at = time.monotonic()
                    self.stream_changed.notify_all()
                    previous_completed = self.performance.get('completed_at')
                    self.performance = dict(
                        completed_at=self.overlay_at,
                        result_fps=None if previous_completed is None else 1 / max(1e-9, self.overlay_at-previous_completed),
                        grounding_ipc=dict(getattr(detector, 'last_metrics', {})),
                        detect_ms=(detected_at-start)*1000,
                        recognize_ms=(recognized_at-detected_at)*1000,
                        render_ms=(self.overlay_at-recognized_at)*1000,
                        processing_ms=(self.overlay_at-start)*1000,
                        callback_to_result_ms=None if received_at is None else (self.overlay_at-received_at)*1000)
                    self.status = (f'Grounding DINO → {encoder.description} | {len(detections)} objects | '
                                   f'{len(set(gallery.labels))} classes / {len(gallery.labels)} references | '
                                   f'{time.monotonic()-start:.2f}s inference')
                    self.pending_publication = (overlay, detections, header, gen)
                    self.publish_changed.set()
                processed = (gen, seq, gallery_revision)
            except Exception as error:
                with self.lock:
                    if gen == self.generation:
                        self.jpeg = None
                        self.debug_jpegs = {}
                        self.clear_object_debug()
                        self.detections = []
                        self.overlay_at = None
                        self.status = f'Error: {type(error).__name__}: {error}'
                self.get_logger().error(str(error))
                # Retry failed files/model calls at a bounded rate, including static uploads.
                self.stop.wait(1.)


PAGE = '''<!doctype html><html><head><meta charset="utf-8"><title>DINOv3 Cases</title>
<style>body{font:16px system-ui;background:#101821;color:#e7eef8;max-width:1150px;margin:32px auto;padding:0 20px}section{background:#1b2836;padding:20px;border-radius:12px;margin:16px 0}button,input,select{padding:10px;margin:6px;border-radius:6px}button{background:#8ddcc4;cursor:pointer}img{max-width:100%;border-radius:10px}small{color:#b0becd}.reference-gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(280px,100%),1fr));gap:16px}.reference-card{margin:0;padding:12px;background:#101821;border-radius:8px;overflow-wrap:anywhere}.reference-card img{display:block;width:100%;image-rendering:pixelated}.reference-card figcaption{margin-top:10px}</style></head><body>
<h1>Grounding DINO → DINOv3 物體辨識</h1><p><a href="/images" style="color:#8ddcc4">圖片管理／HSV 去背／標籤分類</a></p><p id="pipeline-help">先透過文字找出物體，再與所有商品參考圖比對。每個 case 代表一個商品身分。</p>
<section id="grounding-section"><h2>偵測與辨識設定</h2><form id="grounding-form">
<label>文字提示 <input id="prompt" name="prompt" maxlength="1000" required style="width:min(600px,85%)"></label><br>
<label>偵測門檻 <input name="box_threshold" type="number" min="0" max="1" step="0.01" required></label>
<label>文字門檻 <input name="text_threshold" type="number" min="0" max="1" step="0.01" required></label><br>
<label>辨識相似度門檻 <input name="recognition_threshold" type="number" min="0" max="1" step="0.01" required></label>
<label>前兩名最小差距 <input name="margin" type="number" min="0" max="2" step="0.01" required></label>
<button>套用設定</button></form><small>英文類別以句點分隔，例如 medicine bottle. medicine box. bottle. box.。門檻或差距不足時顯示 unknown。</small></section>
<section><h2>1. Reference groups</h2><form id="upload"><input name="case" placeholder="Case name, e.g. medicine_01" required pattern="[A-Za-z0-9_-]{1,80}"><input name="images" type="file" accept="image/*" multiple required><button>Add pictures to case</button></form>
<select id="cases"><option value="">Select case</option></select><button id="select">檢視此圖庫</button><p><small>每個 case 放同一商品的多角度近距離照片。兩階段模式會比對所有圖庫；這裡的選擇僅切換參考圖預覽。</small></p></section>
<section><h2>範例照片 · Patch 分割示意圖</h2><p id="reference-status" role="status">請選擇 case 以查看每張範例照片。</p><p><small>黃色線為模型輸入的 patch 網格；右側與底部不足一格時，自動複製邊緣像素補齊。</small></p><div id="reference-gallery" class="reference-gallery"></div></section>
<section><h2>2. Target image</h2><button id="camera">Use ROS camera</button><form id="target"><input name="image" type="file" accept="image/*" required><button>Use uploaded target</button></form></section>
<section><h2>Live camera</h2><p id="camera-status">Waiting for camera…</p><img id="preview" src="/camera.mjpg" alt="Live Orbbec camera"></section>
<section><h2>辨識結果</h2><div id="legacy-threshold"><label for="threshold">Cosine threshold: <output id="threshold-value">0.70</output></label><input id="threshold" type="range" min="0" max="1" step="0.01" value="0.70" style="width: min(450px, 90%)"></div><p id="status">Connecting…</p><p id="error" role="alert"></p><img id="overlay" src="/overlay.mjpg" alt="物體框選與辨識結果"><p><small>det 為 Grounding DINO 偵測分數；sim 為 DINOv3 圖庫相似度，並非商品辨識機率。綠框為已辨識商品，黃框為 unknown。</small></p><div id="recognition-list"></div></section>
<section id="debug-section" hidden><h2>Patch diagnostics</h2><select id="debug-view"><option value="grid">Original + patch grid</option><option value="patches">Patch similarity</option><option value="heatmap">Full-image similarity</option><option value="mask">Mask after threshold, closing and hole filling</option><option value="boxes">Final bounding boxes</option></select><p><img id="debug-image" alt="Patch diagnostic"></p></section>
<section id="object-section" hidden><h2>逐物體 DINOv3 視覺化</h2><label><input id="object-enabled" type="checkbox" checked>開啟診斷圖片（會增加運算負載）</label>
<label>比對 reference 標籤 <select id="object-reference"><option value="">自動最佳參考圖</option></select></label>
<label>顯示內容 <select id="object-view"><option value="heatmap">ROI 相似度熱圖疊加</option><option value="crop">Grounding bbox 裁切原圖</option><option value="grid">模型輸入與 patch 網格</option><option value="patches">逐 patch 相似度</option><option value="similarity">純相似度熱圖</option><option value="reference">最佳參考圖</option></select></label>
<div id="object-scale" style="max-width:320px"><div style="height:12px;background:linear-gradient(to right,#30123b,#466be3,#1ae4b6,#a4fc3b,#f9ba38,#e3440a,#7a0403);border-radius:4px"></div><small style="display:flex;justify-content:space-between"><span>0</span><span>0.5</span><span>1</span></small></div>
<p id="object-help"></p><p><small>自動模式使用各物體的最佳參考圖；指定標籤時，每個 Grounding object 都與該標籤所有圖片的 patch 比對。最佳參考圖視圖顯示該標籤內對各物體最相似的圖片。局部 patch 相似度與整體商品辨識分數不同。網格內藍框為縮放後 ROI 範圍，外側為補邊。</small></p>
<p id="object-status" role="status">等待物體偵測結果…</p><div id="object-gallery" class="reference-gallery"></div></section>
<script>
const el=id=>document.getElementById(id);
let thresholdDirty=false;
let referenceKey='',referenceVersion=0;
let groundingLoaded=false;
el('grounding-form').onsubmit=e=>{e.preventDefault();const values=Object.fromEntries(new FormData(e.target));for(const key of Object.keys(values)){if(key!=='prompt')values[key]=Number(values[key]);}send('/api/grounding',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(values)});};
function updateGrounding(d){const active=d.pipeline==='grounded';el('grounding-section').hidden=!active;el('legacy-threshold').hidden=active;el('debug-section').hidden=active||!d.debug_visualization;el('object-section').hidden=!active;if(!active){el('pipeline-help').textContent='全圖 DINOv3 熱圖模式：選擇一個 case 進行 patch 比對。';el('select').textContent='Use selected case';}if(!groundingLoaded&&d.grounding){for(const [key,value] of Object.entries(d.grounding))el('grounding-form').elements.namedItem(key).value=value;groundingLoaded=true;}}
async function refreshDetections(){const r=await fetch('/api/detections');if(!r.ok)return;const data=await r.json();const list=el('recognition-list');list.replaceChildren();for(const item of data.detections){if(!item.label)continue;const row=document.createElement('p');row.textContent=item.label+' · '+item.detection_label+' · det '+item.detection_score.toFixed(2)+' · sim '+(item.score===null?'—':item.score.toFixed(3))+' · '+item.reason+(item.label==='unknown'&&item.candidate?' · 最接近 '+item.candidate:'')+(item.margin===null?'':' · 差距 '+item.margin.toFixed(3));list.append(row);}}
async function refreshReferences(group,revision,stride){
 const key=JSON.stringify([group,revision,stride]);if(key===referenceKey)return;referenceKey=key;const version=++referenceVersion;
 el('reference-gallery').replaceChildren();el('reference-status').textContent=group?'載入 '+group+' 的範例照片…':'請選擇 case 以查看每張範例照片。';if(!group)return;
 try{const r=await fetch('/api/references?case='+encodeURIComponent(group));const d=await r.json();if(version!==referenceVersion)return;if(!r.ok)throw Error(d.error||r.statusText);
 el('reference-status').textContent=d.case+' · '+d.images.length+' 張範例照片 · 每格 '+d.patch_size+' × '+d.patch_size+' 像素';
 for(const item of d.images){const card=document.createElement('figure');card.className='reference-card';const caption=document.createElement('figcaption');caption.textContent=item.name;
 if(item.error){const error=document.createElement('p');error.textContent=item.error;card.append(error);}else{const image=document.createElement('img');image.loading='lazy';image.alt=item.name+' 的 patch 分割示意圖';image.src='/api/references/grid/'+encodeURIComponent(group)+'/'+encodeURIComponent(item.name)+'?v='+encodeURIComponent(item.version);image.onerror=()=>{if(version===referenceVersion)caption.textContent+=' · 示意圖載入失敗';};card.append(image);const details=document.createElement('p');details.textContent='原圖 '+item.original_size.join(' × ')+' → 模型輸入 '+item.size.join(' × ')+' · '+item.rows+' 列 × '+item.columns+' 欄（'+item.rows*item.columns+' patches）';card.append(details);}card.append(caption);el('reference-gallery').append(card);}
 }catch(e){if(version===referenceVersion){el('reference-status').textContent=e.message;referenceKey='';}}
}
el('cases').onchange=()=>{referenceKey='';refresh().catch(()=>{});};
el('threshold').oninput=()=>{thresholdDirty=true;el('threshold-value').textContent=Number(el('threshold').value).toFixed(2);};
el('threshold').onchange=async()=>{await send('/api/threshold',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({threshold:Number(el('threshold').value)})});thresholdDirty=false;};
async function send(url, options){el('error').textContent='';try{const r=await fetch(url,options);const d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);await refresh();}catch(e){el('error').textContent=e.message;}}
el('upload').onsubmit=e=>{e.preventDefault();send('/api/cases',{method:'POST',body:new FormData(e.target)});};
el('select').onclick=()=>send('/api/select',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({case:el('cases').value})});
el('target').onsubmit=e=>{e.preventDefault();let f=new FormData(e.target);f.set('source','upload');send('/api/source',{method:'POST',body:f});};
el('camera').onclick=()=>{let f=new FormData();f.set('source','camera');send('/api/source',{method:'POST',body:f});};
async function refresh(){const r=await fetch('/api/status');const d=await r.json();updateGrounding(d);if(!thresholdDirty){el('threshold').value=d.threshold;el('threshold-value').textContent=Number(d.threshold).toFixed(2);}el('status').textContent=d.status+' | source: '+d.source+(d.overlay_age===null?'':' | result age: '+d.overlay_age.toFixed(1)+'s');el('camera-status').textContent=d.preview_error|| (d.camera_connected?'Live · '+d.camera_topic:'No recent camera frames · '+d.camera_topic+' · start the Orbbec driver');el('preview').style.opacity=d.camera_connected?'1':'0.3';const s=el('cases');if(JSON.stringify(d.cases)!==s.dataset.list){const value=s.value||d.selected;s.replaceChildren(new Option('Select case',''),...d.cases.map(c=>new Option(c,c)));s.value=value;s.dataset.list=JSON.stringify(d.cases);}refreshReferences(s.value||d.selected,d.reference_revision,d.reference_stride);el('overlay').style.display=d.overlay_ready?'':'none';if(d.performance&&d.performance.result_fps)el('status').textContent+=' | '+d.performance.result_fps.toFixed(1)+' FPS | callback→result '+(d.performance.callback_to_result_ms===null?'—':d.performance.callback_to_result_ms.toFixed(0)+' ms');}

let debugBusy=false;
async function refreshDebug(){if(debugBusy||el('debug-section').hidden)return;debugBusy=true;try{const r=await fetch('/debug/'+el('debug-view').value+'.jpg');const im=el('debug-image');if(r.status===200){const old=im.src;im.src=URL.createObjectURL(await r.blob());if(old.startsWith('blob:'))URL.revokeObjectURL(old);}else{const old=im.src;im.removeAttribute('src');if(old.startsWith('blob:'))URL.revokeObjectURL(old);}}finally{debugBusy=false;}}
el('debug-view').onchange=()=>refreshDebug().catch(()=>{});
setInterval(()=>refreshDebug().catch(()=>{}),1000);
const objectHelp={heatmap:'ROI 原圖疊加局部 patch 相似度。固定色階：0（深藍）→ 0.5（綠）→ 1（紅）；負值顯示深藍。',crop:'Grounding DINO 在原圖座標中找到的物體裁切，保持原長寬比。',grid:'實際 DINOv3 模型輸入：黃色為 patch 邊界，藍框外為右側／底部補邊。',patches:'每個 ROI patch 與最佳參考圖所有 patch 的最高 cosine similarity；顯示完整模型網格（含補邊）。',similarity:'移除補邊並映射回 ROI 的純相似度圖。固定色階 0–1；負值顯示深藍。',reference:'整體商品特徵比對分數最高的那張參考圖，使用與辨識相同的縮放與補邊。'};
let objectBusy=false,objectKey='',objectUrls=[];
function clearObjectUrls(){for(const url of objectUrls)URL.revokeObjectURL(url);objectUrls=[];}
function updateObjectReferences(cases){const select=el('object-reference');const list=JSON.stringify(cases);if(select.dataset.list===list)return;const selected=select.value;select.replaceChildren(new Option('自動最佳參考圖',''),...cases.map(name=>new Option(name,name)));select.value=cases.includes(selected)?selected:'';select.dataset.list=list;}
async function refreshObjects(){
 if(objectBusy||el('object-section').hidden)return;
 if(!el('object-enabled').checked){el('object-status').textContent='診斷圖片已關閉；勾選「開啟診斷圖片」以顯示物體熱圖。';return;}objectBusy=true;
 const view=el('object-view').value;el('object-help').textContent=objectHelp[view];
 const referenceCase=el('object-reference').value;const comparisonQuery=referenceCase?'?reference_case='+encodeURIComponent(referenceCase):'';
 if(referenceCase)el('object-help').textContent='比對標籤 '+referenceCase+' 的全部參考圖片。'+objectHelp[view].replace('最佳參考圖所有 patch','指定標籤全部參考圖的 patch');
 el('object-scale').hidden=!['heatmap','patches','similarity'].includes(view);
 try{
  const response=await fetch('/api/object_visualizations'+comparisonQuery);const data=await response.json();if(data.ready||!response.ok&&data.reference_cases)updateObjectReferences(data.reference_cases);if(!response.ok)throw Error(data.error||'無法讀取物體視覺化');
  const key=JSON.stringify([data.revision,view,referenceCase]);if(key===objectKey)return;
  if(!data.objects.length){clearObjectUrls();el('object-gallery').replaceChildren();el('object-status').textContent='目前沒有物體結果；請提供目標影像並檢查文字提示與偵測門檻。';objectKey=key;return;}
  const images=await Promise.all(data.objects.map(async item=>{
   if(!item.views.includes(view))return null;
   const image=await fetch('/debug/objects/'+data.revision+'/'+item.index+'/'+view+'.jpg'+comparisonQuery);
   if(image.status===410)throw Error('結果已更新，正在取得最新視覺化…');if(!image.ok)throw Error('物體 '+(item.index+1)+' 視覺化讀取失敗');return image.blob();
  }));
  if(!el('object-enabled').checked||view!==el('object-view').value||referenceCase!==el('object-reference').value)return;
  clearObjectUrls();const gallery=el('object-gallery');gallery.replaceChildren();
  data.objects.forEach((item,index)=>{
   const card=document.createElement('figure');card.className='reference-card';
   const title=document.createElement('figcaption');title.textContent='#'+(item.index+1)+' · '+item.label+' · '+item.detection_label;card.append(title);
   const detail=document.createElement('p');detail.textContent='bbox ['+item.bbox.join(', ')+'] · ROI '+item.roi_size.join(' × ')+' → 輸入 '+item.input_size.join(' × ')+' · '+item.patch_grid.join(' × ')+' patches';card.append(detail);
   const score=document.createElement('p');score.textContent='det '+item.detection_score.toFixed(3)+' · 商品 sim '+(item.score===null?'—':item.score.toFixed(3))+' · '+item.reason;card.append(score);
   const comparison=document.createElement('p');comparison.textContent=item.comparison_reference?'比對 '+item.comparison_case+' · 標籤 sim '+item.comparison_score.toFixed(3)+' · '+item.comparison_reference_count+' 張參考圖 · 最佳 '+item.comparison_reference.name:'沒有可用參考圖';card.append(comparison);
   if(images[index]){const image=document.createElement('img');image.alt=title.textContent+' '+objectHelp[view];const url=URL.createObjectURL(images[index]);objectUrls.push(url);image.src=url;card.append(image);}else{const empty=document.createElement('p');empty.textContent='沒有參考圖庫，無法計算相似度；可切換裁切原圖或 patch 網格。';card.append(empty);}
   if(item.heatmap_available){const range=document.createElement('small');range.textContent='ROI 局部 cosine 範圍 '+item.similarity_min.toFixed(3)+'–'+item.similarity_max.toFixed(3);card.append(range);}gallery.append(card);
  });
  el('object-status').textContent=data.objects.length+' 個物體 · '+(referenceCase?'比對 '+referenceCase:'自動最佳參考圖')+' · 同一幀結果 #'+data.revision;objectKey=key;
 }catch(error){el('object-status').textContent=error.message;}finally{objectBusy=false;}
}
el('object-view').onchange=()=>{objectKey='';refreshObjects();};
el('object-reference').onchange=()=>{objectKey='';clearObjectUrls();el('object-gallery').replaceChildren();el('object-status').textContent='正在更新標籤熱圖…';refreshObjects();};
el('object-enabled').onchange=()=>{objectKey='';if(!el('object-enabled').checked){clearObjectUrls();el('object-gallery').replaceChildren();}refreshObjects();};
setInterval(refreshObjects,1000);
async function poll(){try{await refresh();await refreshDetections();}catch(e){el('status').textContent='Connection lost: '+e.message;}setTimeout(poll,250);}poll();
</script></body></html>'''


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


if __name__ == '__main__':
    main()
