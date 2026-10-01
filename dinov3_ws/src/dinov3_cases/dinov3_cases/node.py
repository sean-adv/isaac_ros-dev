"""Local ROS 2 / browser interface for grouped DINOv3 reference matching."""
import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import torch
import torch.nn.functional as F
from cv_bridge import CvBridge
from flask import Flask, jsonify, request, Response
from PIL import Image, ImageOps
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image as ImageMsg
from transformers import AutoModel, AutoImageProcessor
from werkzeug.serving import make_server

from .boxes import BoxConfig, extract_boxes, detection_message, draw_boxes
from vision_msgs.msg import Detection2DArray

MODEL = 'facebook/dinov3-vith16plus-pretrain-lvd1689m'
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
    def __init__(self, model_id, side):
        if side < 224 or side > 1024 or side % 16:
            raise ValueError('image_size must be a multiple of 16 between 224 and 1024.')
        self.side = side
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, dtype=self.dtype).to(self.device).eval()

    @torch.inference_mode()
    def encode(self, image):
        # Letterbox without cropping; remove padding patches from reference banks.
        w, h = image.size
        scale = self.side / max(w, h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        left, top = (self.side - nw) // 2, (self.side - nh) // 2
        canvas = Image.new('RGB', (self.side, self.side), (124, 116, 104))
        canvas.paste(image.resize((nw, nh), Image.Resampling.BICUBIC), (left, top))
        inputs = self.processor(images=canvas, do_resize=False, do_center_crop=False, return_tensors='pt')
        inputs = {k: v.to(device=self.device, dtype=self.dtype) for k, v in inputs.items()}
        outputs = self.model(**inputs).last_hidden_state
        patches = outputs[0, 1 + self.model.config.num_register_tokens:]
        grid = self.side // self.model.config.patch_size
        if patches.shape[0] != grid * grid:
            raise RuntimeError('Unexpected DINOv3 patch grid.')
        features = F.normalize(patches.float(), dim=-1)
        centers = (torch.arange(grid, device=self.device) + .5) * self.model.config.patch_size
        valid = ((centers[:, None] >= top) & (centers[:, None] < top + nh)
                 & (centers[None, :] >= left) & (centers[None, :] < left + nw)).flatten()
        return features, valid, (left, top, nw, nh, grid)

    @torch.inference_mode()
    def similarity_map(self, image, bank):
        features, valid, geometry = self.encode(image)
        grid = geometry[-1]
        # Bound matrix memory even for a large case bank.
        score = torch.full((features.shape[0],), -1., device=self.device)
        for chunk in bank.split(2048):
            score = torch.maximum(score, (features @ chunk.T).max(dim=1).values)
        raw = score.reshape(grid, grid).cpu().numpy()
        return raw, valid.reshape(grid, grid).cpu().numpy(), geometry

    def heatmap(self, image, bank):
        raw, _, geometry = self.similarity_map(image, bank)
        return self.project_heatmap(raw, geometry, image.size)

    def project_heatmap(self, raw, geometry, image_size):
        left, top, nw, nh, _ = geometry
        square = cv2.resize(raw, (self.side, self.side), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(square[top:top + nh, left:left + nw], image_size, interpolation=cv2.INTER_LINEAR)


class CaseNode(Node):
    def __init__(self):
        super().__init__('dinov3_cases')
        defaults = dict(cases_dir='/workspaces/isaac_ros-dev/dinov3_cases_data',
                        image_topic='/camera/color/image_raw', model_id=MODEL,
                        image_size=448, port=8090, host='127.0.0.1', inference_hz=5.0,
                        max_reference_patches=8192, cosine_threshold=0.70,
                        **vars(BoxConfig()))
        for k, v in defaults.items():
            self.declare_parameter(k, v)
        self.param = lambda k: self.get_parameter(k).value
        self.root = Path(self.param('cases_dir')).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.threshold = float(self.param('cosine_threshold'))
        if not 0 <= self.threshold <= 1:
            raise ValueError('cosine_threshold must be between 0 and 1.')
        self.box_config = BoxConfig(**{key: self.param(key) for key in vars(BoxConfig())})
        self.selected = '' 
        self.generation = 0
        self.source = 'camera'
        self.frame = None
        self.frame_seq = 0
        self.jpeg = None
        self.preview_jpeg = None
        self.camera_frame = None
        self.camera_seq = 0
        self.camera_received_at = None
        self.preview_error = None
        self.overlay_at = None
        self.status = 'Loading DINOv3 model…'
        self.bank = None
        self.bank_key = None
        self.pub = self.create_publisher(ImageMsg, '/dinov3/heatmap/overlay', 1)
        self.raw_pub = self.create_publisher(ImageMsg, '/dinov3/heatmap/similarity', 1)
        self.recognition_pub = self.create_publisher(Detection2DArray, '/medicine_recognition/detections', 10)
        self.recognition_debug_pub = self.create_publisher(ImageMsg, '/medicine_recognition/debug_image', 1)
        self.create_subscription(ImageMsg, self.param('image_topic'), self.on_image, qos_profile_sensor_data)
        self.app = self.make_app()
        self.server = make_server(self.param('host'), self.param('port'), self.app, threaded=True)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.preview_worker = threading.Thread(target=self.run_preview, daemon=True)
        self.preview_worker.start()
        self.worker = threading.Thread(target=self.run_worker, daemon=True)
        self.worker.start()
        self.get_logger().info(f"Browser: http://{self.param('host')}:{self.param('port')}")

    def on_image(self, msg):
        with self.lock:
            self.camera_frame = msg
            self.camera_seq += 1
            self.camera_received_at = time.monotonic()
            if self.source != 'camera':
                return
            self.frame = msg
            self.frame_seq += 1

    def run_preview(self):
        processed = -1
        while not self.stop.wait(1 / 15):
            with self.lock:
                frame, seq = self.camera_frame, self.camera_seq
            if frame is None or seq == processed:
                continue
            try:
                bgr = self.bridge.imgmsg_to_cv2(frame, desired_encoding='bgr8')
                if bgr.shape[1] > 960:
                    bgr = cv2.resize(bgr, (960, round(bgr.shape[0] * 960 / bgr.shape[1])))
                ok, data = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    raise ValueError('Could not encode camera frame')
                with self.lock:
                    self.preview_jpeg = data.tobytes()
                    self.preview_error = None
            except Exception as e:
                with self.lock:
                    self.preview_error = str(e)
            processed = seq

    def make_app(self):
        app = Flask(__name__)
        app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024

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
                               camera_topic=self.param('image_topic'), camera_age=age,
                               camera_connected=age is not None and age < 3,
                               preview_error=self.preview_error,
                               overlay_age=None if self.overlay_at is None else time.monotonic() - self.overlay_at)

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
                self.generation += 1
                self.jpeg = None
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
            return jsonify(ok=True)

        @app.get('/camera.mjpg')
        def camera_stream():
            def frames():
                previous = None
                while not self.stop.wait(1 / 15):
                    with self.lock:
                        data = self.preview_jpeg
                        fresh = self.camera_received_at is not None and time.monotonic() - self.camera_received_at < 3
                    if fresh and data is not None and data is not previous:
                        previous = data
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                               + str(len(data)).encode() + b'\r\n\r\n' + data + b'\r\n')
            return Response(frames(), mimetype='multipart/x-mixed-replace; boundary=frame',
                            headers={'Cache-Control': 'no-store'})

        @app.get('/overlay.jpg')
        def overlay():
            with self.lock:
                data = self.jpeg
            return Response(data or b'', status=200 if data else 204,
                            mimetype='image/jpeg', headers={'Cache-Control': 'no-store'})
        return app

    def run_worker(self):
        try:
            encoder = Encoder(self.param('model_id'), self.param('image_size'))
        except Exception as e:
            with self.lock:
                self.status = f'Model load failed: {type(e).__name__}: {e}'
            self.get_logger().error(self.status)
            return
        with self.lock:
            self.status = f'Model ready on {encoder.device}. Upload or select a case.'
        processed = None
        while not self.stop.wait(1 / max(.1, self.param('inference_hz'))):
            with self.lock:
                name, gen, frame, seq = self.selected, self.generation, self.frame, self.frame_seq
                threshold = self.threshold
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
                key = (name, tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in paths))
                if key != self.bank_key:
                    if not paths:
                        raise ValueError('Selected case has no supported pictures.')
                    limit = self.param('max_reference_patches')
                    if limit < len(paths):
                        raise ValueError(f'Too many pictures; maximum is {limit}.')
                    per_image = max(1, limit // len(paths))
                    banks = []
                    for path in paths:
                        features, valid, _ = encoder.encode(decode_image(path))
                        features = features[valid]
                        if not len(features):
                            raise ValueError(f'Reference image is too narrow after resizing: {path.name}')
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
                patch_scores, valid, geometry = encoder.similarity_map(image, self.bank)
                scores = encoder.project_heatmap(patch_scores, geometry, image.size)
                bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                boxes = extract_boxes(patch_scores, valid, geometry, image.size,
                                      encoder.model.config.patch_size, threshold, self.box_config, bgr)
                recognition = detection_message(boxes, name, header)
                overlay = render_heatmap(bgr, scores, threshold)
                draw_boxes(overlay, boxes)
                cv2.putText(overlay, f'{name} | cosine >= {threshold:.2f} | max {scores.max():.2f}',
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 2)
                ok, data = cv2.imencode('.jpg', overlay)
                if not ok:
                    raise RuntimeError('JPEG encoding failed.')
                with self.lock:
                    if gen != self.generation:
                        continue
                    self.jpeg = data.tobytes()
                    self.overlay_at = time.monotonic()
                    self.status = f'{name}: {len(paths)} references | {encoder.device} | {time.monotonic()-start:.2f}s inference'
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
                    self.recognition_debug_pub.publish(out)
                processed = (gen, seq)
            except Exception as e:
                with self.lock:
                    if gen == self.generation:
                        self.jpeg = None
                        self.status = f'Error: {type(e).__name__}: {e}'
                self.get_logger().error(str(e))
                processed = (gen, seq)


PAGE = '''<!doctype html><html><head><meta charset="utf-8"><title>DINOv3 Cases</title>
<style>body{font:16px system-ui;background:#101821;color:#e7eef8;max-width:1150px;margin:32px auto;padding:0 20px}section{background:#1b2836;padding:20px;border-radius:12px;margin:16px 0}button,input,select{padding:10px;margin:6px;border-radius:6px}button{background:#8ddcc4;cursor:pointer}img{max-width:100%;border-radius:10px}small{color:#b0becd}</style></head><body>
<h1>DINOv3 · Case similarity</h1><p>Choose a case to highlight visually similar patches in a camera frame or target picture.</p>
<section><h2>1. Reference groups</h2><form id="upload"><input name="case" placeholder="Case name, e.g. medicine_01" required pattern="[A-Za-z0-9_-]{1,80}"><input name="images" type="file" accept="image/*" multiple required><button>Add pictures to case</button></form>
<select id="cases"><option value="">Select case</option></select><button id="select">Use selected case</button><p><small>Use tightly cropped reference pictures of the object of interest. Backgrounds in reference pictures can also match.</small></p></section>
<section><h2>2. Target image</h2><button id="camera">Use ROS camera</button><form id="target"><input name="image" type="file" accept="image/*" required><button>Use uploaded target</button></form></section>
<section><h2>Live camera</h2><p id="camera-status">Waiting for camera…</p><img id="preview" src="/camera.mjpg" alt="Live Orbbec camera"></section>
<section><h2>Latest heatmap result</h2><label for="threshold">Cosine threshold: <output id="threshold-value">0.70</output></label><input id="threshold" type="range" min="0" max="1" step="0.01" value="0.70" style="width: min(450px, 90%)"><p><small>Threshold controls patch-mask boxes and heatmap color. Boxes use mask closing, hole filling and padding; RGB only adjusts outer borders.</small></p><p id="status">Connecting…</p><p id="error" role="alert"></p><img id="overlay" alt="Heatmap appears after a case and target are available"><p><small>Fixed cosine similarity scale: 0 (dark blue) → 0.5 (green) → 1 (red). Similarity is not a detection probability.</small></p></section>
<script>
const el=id=>document.getElementById(id);
let thresholdDirty=false;
el('threshold').oninput=()=>{thresholdDirty=true;el('threshold-value').textContent=Number(el('threshold').value).toFixed(2);};
el('threshold').onchange=async()=>{await send('/api/threshold',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({threshold:Number(el('threshold').value)})});thresholdDirty=false;};
async function send(url, options){el('error').textContent='';try{const r=await fetch(url,options);const d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);await refresh();}catch(e){el('error').textContent=e.message;}}
el('upload').onsubmit=e=>{e.preventDefault();send('/api/cases',{method:'POST',body:new FormData(e.target)});};
el('select').onclick=()=>send('/api/select',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({case:el('cases').value})});
el('target').onsubmit=e=>{e.preventDefault();let f=new FormData(e.target);f.set('source','upload');send('/api/source',{method:'POST',body:f});};
el('camera').onclick=()=>{let f=new FormData();f.set('source','camera');send('/api/source',{method:'POST',body:f});};
async function refresh(){const r=await fetch('/api/status');const d=await r.json();if(!thresholdDirty){el('threshold').value=d.threshold;el('threshold-value').textContent=Number(d.threshold).toFixed(2);}el('status').textContent=d.status+' | source: '+d.source+(d.overlay_age===null?'':' | result age: '+d.overlay_age.toFixed(1)+'s');el('camera-status').textContent=d.preview_error|| (d.camera_connected?'Live · '+d.camera_topic:'No recent camera frames · '+d.camera_topic+' · start the Orbbec driver');el('preview').style.opacity=d.camera_connected?'1':'0.3';const s=el('cases');if(JSON.stringify(d.cases)!==s.dataset.list){const value=s.value||d.selected;s.replaceChildren(new Option('Select case',''),...d.cases.map(c=>new Option(c,c)));s.value=value;s.dataset.list=JSON.stringify(d.cases);}const im=await fetch('/overlay.jpg?t='+Date.now());if(im.status===200){const url=URL.createObjectURL(await im.blob());const old=el('overlay').src;el('overlay').src=url;if(old.startsWith('blob:'))URL.revokeObjectURL(old);}else{el('overlay').removeAttribute('src');}}
async function poll(){try{await refresh();}catch(e){el('status').textContent='Connection lost: '+e.message;}setTimeout(poll,250);}poll();
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
        node.server.shutdown()
        node.preview_worker.join(timeout=2)
        node.worker.join(timeout=10)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
