"""Local image gallery and HSV background removal, independent of ROS/model loading."""

import io
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_file, render_template
from PIL import Image, ImageDraw, ImageFont

from dinov3_cases.recognition.images import remove_background
from dinov3_cases.recognition.gallery_files import group_path, image_path, load_image as load_rgba, gallery_groups, gallery_images, save_image




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
            raise ValueError("Provide a JSON object.")
        return data

    def save(image, folder, suffix=""):
        folder.mkdir(exist_ok=True)
        destination = folder / f"{uuid.uuid4().hex}{suffix}.png"
        save_image(image, destination)
        return destination.name

    @app.get("/images")
    def manager_page():
        return render_template(
            "images.html", home_label=app.config.get("IMAGE_MANAGER_HOME", "DINOv3 Home"),
            camera_enabled=capture is not None
        )

    @app.get("/api/images")
    def list_images():
        with mutation_lock:
            groups = [path.name for path in gallery_groups(root)]
            group = request.args.get("case", "")
            images = []
            if group:
                folder = group_path(root, group)
                if not folder.is_dir():
                    raise ValueError("Label does not exist.")
                images = [
                    dict(name=p.name, bytes=p.stat().st_size)
                    for p in gallery_images(folder)
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

    @app.post("/api/images/camera-preview")
    def camera_bbox_preview():
        if capture is None:
            return jsonify(error="This image manager is not connected to a camera."), 503
        prompt = payload().get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Enter a SAM3 prompt first.")
        image, proposals = capture(prompt)
        preview = image.copy().convert("RGB")
        draw = ImageDraw.Draw(preview)
        width = max(2, round(min(preview.size) / 250))
        font = ImageFont.load_default(size=max(24, round(min(preview.size) / 30)))
        for item in proposals:
            x0, y0, x1, y1 = item["bbox"]
            text = f"{item['detection_label']} {item['detection_score']:.0%}"
            box = draw.textbbox((x0, y0), text, font=font)
            draw.rectangle(box, fill="#101821")
            draw.text((x0, y0), text, fill="#8ddcc4", font=font)
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline="#8ddcc4", width=width)
        response = png_response(preview)
        response.headers["X-Detection-Count"] = str(len(proposals))
        return response

    @app.post("/api/images/capture")
    def capture_images():
        if capture is None:
            return jsonify(error="This image manager is not connected to a camera. Use the /images page of the camera service."), 503
        data = payload()
        group = data.get("case")
        folder = group_path(root, group)
        if not folder.is_dir():
            raise ValueError("Create or select a label first.")
        prompt = data.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Enter a SAM3 prompt first.")
        image, proposals = capture(prompt)
        # Crop the very same raw frame sent to SAM3, at its original resolution.
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
            raise ValueError("Select an image.")
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
    app.config["IMAGE_MANAGER_HOME"] = "Image Manager Home"

    @app.errorhandler(ValueError)
    def invalid(error):
        return jsonify(error=str(error)), 400

    @app.errorhandler(413)
    def too_large(error):
        return jsonify(error="Total upload size must not exceed 64 MB."), 413

    @app.get("/")
    def index():
        return render_template("images.html", home_label=app.config["IMAGE_MANAGER_HOME"])

    register_image_manager(app, root)
    return app
