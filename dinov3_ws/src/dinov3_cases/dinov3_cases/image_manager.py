"""Local image gallery and HSV background removal, independent of ROS/model loading."""

import io
import re
import threading
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file, render_template
from PIL import Image, ImageOps

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def group_path(root, name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name):
        raise ValueError("標籤限 1–80 個英文字母、數字、底線或連字號。")
    path = root / name
    if path.is_symlink() or path.resolve().parent != root.resolve():
        raise ValueError("無效的標籤路徑。")
    return path


def image_path(root, group, filename):
    folder = group_path(root, group)
    if (
        not isinstance(filename, str)
        or not filename
        or "\\" in filename
        or "/" in filename
        or Path(filename).name != filename
    ):
        raise ValueError("無效的圖片名稱。")
    path = folder / filename
    if (
        path.is_symlink()
        or path.resolve().parent != folder.resolve()
        or path.suffix.lower() not in EXTENSIONS
        or not path.is_file()
    ):
        raise ValueError("圖片不存在或格式不支援。")
    return path


def load_rgba(source):
    try:
        with Image.open(source) as image:
            if image.width * image.height > 25_000_000:
                raise ValueError("圖片不得超過 2500 萬像素。")
            return ImageOps.exif_transpose(image).convert("RGBA")
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError("無法讀取圖片。") from error


def remove_background(image, options):
    """Match circular hue and S/V, optionally flood-fill only the clicked region."""
    if not isinstance(options, dict):
        raise ValueError("請提供去背參數。")

    def number(name, default, maximum):
        value = options.get(name, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or not 0 <= value <= maximum
        ):
            raise ValueError(f"{name} 必須介於 0 到 {maximum}。")
        return float(value)

    x = number("x", None, image.width - 1)
    y = number("y", None, image.height - 1)
    if not x.is_integer() or not y.is_integer():
        raise ValueError("取色座標必須是整數。")
    x, y = int(x), int(y)
    h = number("hue", 15, 180) / 2  # UI degrees; OpenCV uses half degrees.
    s = number("saturation", 50, 255)
    v = number("value", 50, 255)
    feather = number("feather", 0, 10)
    connected = options.get("connected", True)
    if not isinstance(connected, bool):
        raise ValueError("connected 必須是布林值。")
    rgba = np.array(image.convert("RGBA"))
    if rgba[y, x, 3] == 0:
        raise ValueError("請點選尚未透明的背景位置。")
    hsv = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2HSV).astype(np.float32)
    sample = hsv[y, x]
    hue_distance = np.abs(hsv[..., 0] - sample[0])
    hue_distance = np.minimum(hue_distance, 180 - hue_distance)
    # Hue is unstable near grayscale; S and V still constrain those matches.
    hue_match = (hue_distance <= h) | ((sample[1] <= 20) & (hsv[..., 1] <= 20))
    mask = (
        hue_match
        & (np.abs(hsv[..., 1] - sample[1]) <= s)
        & (np.abs(hsv[..., 2] - sample[2]) <= v)
        & (rgba[..., 3] > 0)
    ).astype(np.uint8)
    if connected:
        cv2.floodFill(mask, None, (x, y), 2, flags=4)
        mask = (mask == 2).astype(np.uint8)
    removal = mask.astype(np.float32)
    if feather > 0:
        removal = cv2.GaussianBlur(removal, (0, 0), feather)
    rgba[..., 3] = np.rint(rgba[..., 3] * (1 - removal)).astype(np.uint8)
    # Erase hidden background RGB as well; RGB-only consumers otherwise recover it.
    rgba[rgba[..., 3] == 0, :3] = 0
    return Image.fromarray(rgba)


def png_response(image):
    output = io.BytesIO()
    image.save(output, format="PNG")
    output.seek(0)
    response = send_file(output, mimetype="image/png")
    response.headers["Cache-Control"] = "no-store"
    return response


def register_image_manager(app, root, on_change=None, capture=None):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    mutation_lock = threading.RLock()

    def changed(group):
        if on_change:
            on_change(group)

    def payload():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise ValueError("請提供 JSON 物件。")
        return data

    def save(image, folder, suffix=""):
        folder.mkdir(exist_ok=True)
        destination = folder / f"{uuid.uuid4().hex}{suffix}.png"
        temporary = destination.with_suffix(".tmp")
        try:
            image.save(temporary, format="PNG")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination.name

    @app.get("/images")
    def manager_page():
        return render_template(
            "images.html", home_label=app.config.get("IMAGE_MANAGER_HOME", "DINOv3 首頁"),
            camera_enabled=capture is not None
        )

    @app.get("/api/images")
    def list_images():
        with mutation_lock:
            groups = sorted(
                p.name
                for p in root.iterdir()
                if p.is_dir()
                and not p.is_symlink()
                and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", p.name)
            )
            group = request.args.get("case", "")
            images = []
            if group:
                folder = group_path(root, group)
                if not folder.is_dir():
                    raise ValueError("標籤不存在。")
                images = [
                    dict(name=p.name, bytes=p.stat().st_size)
                    for p in sorted(folder.iterdir())
                    if p.suffix.lower() in EXTENSIONS and p.is_file() and not p.is_symlink()
                ]
        return jsonify(cases=groups, images=images)

    @app.get("/api/images/file/<group>/<filename>")
    def picture(group, filename):
        # Normalize EXIF orientation so gallery, click coordinates and HSV agree.
        with mutation_lock:
            image = load_rgba(image_path(root, group, filename))
        return png_response(image)

    @app.post("/api/images/labels")
    def create_label():
        group = payload().get("case")
        with mutation_lock:
            group_path(root, group).mkdir(exist_ok=True)
        return jsonify(ok=True, case=group)

    @app.post("/api/images/capture")
    def capture_images():
        if capture is None:
            return jsonify(error="此圖片管理服務未連接相機，請使用相機主程式的 /images 頁面。"), 503
        data = payload()
        group = data.get("case")
        folder = group_path(root, group)
        if not folder.is_dir():
            raise ValueError("請先建立或選擇標籤。")
        prompt = data.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("請先選擇 OmDet prompt。")
        image, proposals = capture(prompt)
        # Crop the very same raw frame sent to OmDet, at its original resolution.
        crops = [image.crop(item["bbox"]) for item in proposals]
        with mutation_lock:
            names = [save(crop, folder, "_bbox") for crop in crops]
            if names:
                changed(group)
        return jsonify(ok=True, case=group, names=names, detections=proposals)

    @app.post("/api/images/upload")
    def upload_images():
        group = request.form.get("case", "")
        folder = group_path(root, group)
        files = request.files.getlist("images")
        if not files:
            raise ValueError("請選擇圖片。")
        images = [load_rgba(file.stream) for file in files]
        with mutation_lock:
            names = [save(image, folder) for image in images]
            changed(group)
        return jsonify(ok=True, names=names)

    @app.post("/api/images/preview")
    def preview():
        data = payload()
        with mutation_lock:
            image = load_rgba(image_path(root, data.get("case"), data.get("name")))
        return png_response(remove_background(image, data.get("options")))

    @app.post("/api/images/cutout")
    def cutout():
        data = payload()
        target = data.get("target_case", data.get("case"))
        folder = group_path(root, target)
        with mutation_lock:
            image = load_rgba(image_path(root, data.get("case"), data.get("name")))
            result = remove_background(image, data.get("options"))
            name = save(result, folder, "_cutout")
            changed(target)
        return jsonify(ok=True, case=target, name=name)

    @app.delete("/api/images/file/<group>/<filename>")
    def delete_picture(group, filename):
        with mutation_lock:
            image_path(root, group, filename).unlink()
            changed(group)
        return jsonify(ok=True)


def create_app(root):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
    app.config["IMAGE_MANAGER_HOME"] = "圖片管理首頁"

    @app.errorhandler(ValueError)
    def invalid(error):
        return jsonify(error=str(error)), 400

    @app.errorhandler(413)
    def too_large(error):
        return jsonify(error="上傳總大小不得超過 64 MB。"), 413

    @app.get("/")
    def index():
        return render_template("images.html", home_label=app.config["IMAGE_MANAGER_HOME"])

    register_image_manager(app, root)
    return app
