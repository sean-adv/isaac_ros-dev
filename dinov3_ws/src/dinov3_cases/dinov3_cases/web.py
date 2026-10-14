"""HTTP routes for the shared recognition node; inference stays in node.py."""

import re
import time
from io import BytesIO
from flask import Flask, jsonify, request, Response, render_template
from PIL import Image, ImageOps
from werkzeug.exceptions import ServiceUnavailable
from .runtime import MODEL_ID, GROUNDING_MODEL_ID, INPUT_SIZE
from .patches import reference_grid_image
from .grounded import prepare_object, validate_settings
from .image_manager import register_image_manager, image_path, EXTENSIONS


def case_path(root, name):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name):
        raise ValueError("Case name must contain 1–80 letters, numbers, underscores or hyphens.")
    path = root / name
    if path.resolve().parent != root.resolve() or path.is_symlink():
        raise ValueError("Invalid case directory.")
    return path


def decode_image(stream):
    with Image.open(stream) as image:
        if image.width * image.height > 25_000_000:
            raise ValueError("Image exceeds 25 megapixels.")
        return ImageOps.exif_transpose(image).convert("RGB")


def create_app(node):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

    @app.after_request
    def wake_after_update(response):
        if request.method != "GET":
            node.image_changed.set()
            with node.stream_changed:
                node.stream_changed.notify_all()
        return response

    def images_changed(group):
        with node.lock:
            node.generation += 1
            node.jpeg = None
            node.jpeg3d = None
            node.clear_object_debug()
            node.detections = []
            node.overlay_at = None
            node.status = "Building reference features…"

    def capture_camera(prompt):
        with node.lock:
            detector = getattr(node, "grounding_detector", None)
            frame = node.camera_frame
            received_at = node.camera_received_at
            settings = dict(node.grounding_settings)
        if detector is None:
            raise ServiceUnavailable("OmDet is loading. Please try again shortly.")
        if frame is None or received_at is None or time.monotonic() - received_at >= 3:
            raise ServiceUnavailable("The camera is disconnected or its frame is stale. Check the camera feed.")
        detector.validate_prompt(prompt)
        image = Image.fromarray(node.bridge.imgmsg_to_cv2(frame, desired_encoding="rgb8")).copy()
        proposals = detector.detect(
            image, prompt, settings["box_threshold"],
            node.param("grounding_nms_iou"), node.param("grounding_max_detections"),
            min_width=node.param("grounding_min_bbox_width"),
            min_height=node.param("grounding_min_bbox_height"),
            min_area=node.param("grounding_min_bbox_area"),
        )
        return image, proposals

    @app.errorhandler(503)
    def unavailable(error):
        return jsonify(error=error.description), 503

    register_image_manager(app, node.root, images_changed, capture=capture_camera)

    @app.errorhandler(ValueError)
    def bad_request(error):
        return jsonify(error=str(error)), 400

    @app.get("/")
    def index():
        return render_template("cases.html")

    @app.get("/api/status")
    def status():
        cases = sorted(
            p.name
            for p in node.root.iterdir()
            if p.is_dir() and not p.is_symlink() and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", p.name)
        )
        with node.lock:
            age = (
                None
                if node.camera_received_at is None
                else time.monotonic() - node.camera_received_at
            )
            return jsonify(
                cases=cases,
                selected=node.selected,
                source=node.source,
                status=node.status,
                performance=node.performance,
                overlay_ready=node.jpeg is not None,
                grounding=node.grounding_settings,
                grounding_model_id=GROUNDING_MODEL_ID,
                grounding_backend="trt",
                grounding_precision=getattr(getattr(node, "grounding_detector", None), "precision", None),
                grounding_offline_prompts=getattr(getattr(node, "grounding_detector", None), "offline_prompts", []),
                grounding_execution=node.param("grounding_execution"),
                grounding_worker=getattr(
                    getattr(node, "grounding_detector", None), "worker_info", None
                ),
                grounding_engine_path=node.param("grounding_engine_path"),
                object_max_side=INPUT_SIZE,
                object_square_input=True,
                reference_revision=node.generation,
                reference_stride=node.reference_stride,
                inference_backend="trt",
                model_id=MODEL_ID,
                backend_description=node.backend_description,
                camera_topic=node.param("image_topic"),
                camera_age=age,
                camera_connected=age is not None and age < 3,
                preview_error=node.preview_error,
                overlay_age=(
                    None if node.overlay_at is None else time.monotonic() - node.overlay_at
                ),
            )

    @app.get("/api/references")
    def references():
        name = request.args.get("case", "")
        folder = case_path(node.root, name)
        if not folder.is_dir():
            raise ValueError("Case does not exist.")
        with node.lock:
            stride = node.reference_stride
        if stride is None:
            return jsonify(error="The model is loading. Patch grids will be available shortly."), 503
        items = []
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in EXTENSIONS or not path.is_file() or path.is_symlink():
                continue
            item = dict(name=path.name, version=f"{path.stat().st_mtime_ns}-{path.stat().st_size}")
            try:
                original = decode_image(path)
                reference = prepare_object(original, INPUT_SIZE, stride)
                item.update(
                    original_size=list(original.size),
                    size=list(reference.size),
                    rows=reference.height // stride,
                    columns=reference.width // stride,
                )
            except (ValueError, OSError, Image.DecompressionBombError) as error:
                item["error"] = str(error)
            items.append(item)
        return jsonify(case=name, patch_size=stride, images=items)

    @app.get("/api/references/grid/<group>/<filename>")
    def reference_grid(group, filename):
        path = image_path(node.root, group, filename)
        with node.lock:
            stride = node.reference_stride
        if stride is None:
            return jsonify(error="The model is loading."), 503
        reference = prepare_object(decode_image(path), INPUT_SIZE, stride)
        data = BytesIO()
        reference_grid_image(reference, stride).save(data, format="PNG")
        return Response(
            data.getvalue(), mimetype="image/png", headers={"Cache-Control": "no-store"}
        )

    @app.post("/api/grounding")
    def grounding_settings():
        payload = request.get_json(silent=True)
        if (
            not isinstance(payload, dict)
            or not payload
            or set(payload) - set(node.grounding_settings)
        ):
            raise ValueError("Expected prompt and/or grounding/recognition thresholds.")
        with node.lock:
            settings = {**node.grounding_settings, **payload}
            settings["prompt"] = validate_settings(**settings)
            detector = getattr(node, "grounding_detector", None)
            if detector is None and "prompt" in payload:
                return jsonify(error="OmDet is loading. Select an offline prompt once it is ready."), 503
            if detector is not None:
                detector.validate_prompt(settings["prompt"])
            node.grounding_settings = settings
            node.generation += 1
            node.jpeg = None
            node.jpeg3d = None
            node.clear_object_debug()
            node.detections = []
            node.overlay_at = None
        return jsonify(ok=True, grounding=settings)

    @app.post("/api/cases")
    def upload():
        name = request.form.get("case", "")
        folder = case_path(node.root, name)
        files = request.files.getlist("images")
        if not files:
            raise ValueError("Choose at least one picture.")
        # Validate all pictures before modifying the case.
        try:
            images = [decode_image(f.stream) for f in files]
        except (OSError, Image.DecompressionBombError) as e:
            raise ValueError(f"Unsupported picture: {e}") from e
        folder.mkdir(exist_ok=True)
        for i, image in enumerate(images):
            dest = folder / f"{time.time_ns()}_{i}.png"
            temp = dest.with_suffix(".tmp")
            image.save(temp, format="PNG")
            temp.replace(dest)
        with node.lock:
            node.selected = name
            node.generation += 1
            node.jpeg = None
            node.jpeg3d = None
            node.clear_object_debug()
            node.detections = []
            node.status = "Building reference features…"
        return jsonify(ok=True, count=len(images))

    @app.post("/api/select")
    def select():
        name = (request.get_json(silent=True) or {}).get("case", "")
        folder = case_path(node.root, name)
        if not folder.is_dir():
            raise ValueError("Case does not exist.")
        with node.lock:
            node.selected = name
        return jsonify(ok=True)

    @app.post("/api/source")
    def source():
        mode = request.form.get("source", "camera")
        if mode not in ("camera", "upload"):
            raise ValueError("Invalid source.")
        image = None
        if mode == "upload":
            if "image" not in request.files:
                raise ValueError("Choose a target picture.")
            try:
                image = decode_image(request.files["image"].stream)
            except (OSError, Image.DecompressionBombError) as e:
                raise ValueError(f"Unsupported picture: {e}") from e
        with node.lock:
            node.source = mode
            node.frame = image
            node.frame_seq += 1
            node.generation += 1
            node.jpeg = None
            node.jpeg3d = None
            node.clear_object_debug()
            node.detections = []
        return jsonify(ok=True)

    def stream(attribute):
        def frames():
            previous = None
            while not node.stop.is_set():
                with node.stream_changed:
                    node.stream_changed.wait_for(
                        lambda: node.stop.is_set() or getattr(node, attribute) is not previous,
                        timeout=0.5,
                    )
                    data = getattr(node, attribute)
                if data is None or data is previous:
                    previous = data
                    continue
                previous = data
                # No per-client frame queue: slow clients pick the newest result next.
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(data)).encode()
                    + b"\r\n\r\n"
                    + data
                    + b"\r\n"
                )

        return Response(
            frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.get("/camera.mjpg")
    def camera_stream():
        return stream("preview_jpeg")

    def overlay_attribute():
        mode = request.args.get("bbox", "2d")
        if mode not in ("2d", "3d"):
            raise ValueError("bbox must be 2d or 3d.")
        return "jpeg3d" if mode == "3d" else "jpeg"

    @app.get("/overlay.mjpg")
    def overlay_stream():
        return stream(overlay_attribute())

    @app.get("/api/detections")
    def detections():
        with node.lock:
            return jsonify(detections=node.detections)

    @app.get("/api/object_visualizations")
    def object_visualizations():
        reference_case = request.args.get("reference_case", "")
        with node.lock:
            revision, snapshot = node.object_debug_revision, node.object_debug_snapshot
            if snapshot is not None and not any(
                v == revision for v, _ in node.object_debug_history
            ):
                node.object_debug_history.append((revision, snapshot))
        if snapshot is None:
            return jsonify(
                revision=revision,
                objects=[],
                reference_cases=[],
                reference_case="",
                ready=False,
            )
        try:
            items, _ = snapshot.render(reference_case)
        except ValueError as error:
            return jsonify(error=str(error), reference_cases=snapshot.reference_cases), 400
        return jsonify(
            revision=revision,
            objects=items,
            reference_cases=snapshot.reference_cases,
            reference_case=reference_case,
            ready=True,
        )

    @app.get("/debug/objects/<int:revision>/<int:index>/<view>.jpg")
    def object_debug_view(revision, index, view):
        with node.lock:
            snapshot = next(
                (
                    snapshot
                    for version, snapshot in node.object_debug_history
                    if version == revision
                ),
                None,
            )
            if snapshot is None:
                return Response(status=410)
        _, images = snapshot.render(request.args.get("reference_case", ""))
        data = images.get((index, view))
        return Response(
            data or b"",
            status=200 if data else 404,
            mimetype="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/overlay.jpg")
    def overlay():
        attribute = overlay_attribute()
        with node.lock:
            data = getattr(node, attribute, None)
        return Response(
            data or b"",
            status=200 if data else 204,
            mimetype="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    return app
