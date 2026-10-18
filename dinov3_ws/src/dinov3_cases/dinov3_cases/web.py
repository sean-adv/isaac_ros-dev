"""HTTP routes for the shared recognition node; inference stays in node.py."""

import time
from io import BytesIO
from flask import Flask, jsonify, request, Response, render_template
from PIL import Image
from werkzeug.exceptions import ServiceUnavailable
from .runtime import MODEL_ID, GROUNDING_MODEL_ID, INPUT_SIZE
from .patches import reference_grid_image
from .grounded import prepare_object, validate_settings
from .kcf import kcf_factory
from .image_manager import register_image_manager
from .gallery_files import (
    group_path as case_path, image_path, load_image, gallery_groups, gallery_images,
)


def decode_image(stream):
    return load_image(stream, mode="RGB")


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

    def invalidate_results():
        """Caller holds node.lock; discard output from the previous input revision."""
        node.generation += 1
        node.jpeg = None
        node.clear_object_debug()
        node.detections = []
        node.overlay_at = None

    def images_changed(group):
        with node.lock:
            invalidate_results()
            node.status = "Building reference features…"

    def capture_camera(prompt):
        with node.lock:
            detector = getattr(node, "grounding_detector", None)
            frame = node.camera_frame
            received_at = node.camera_received_at
            settings = dict(node.grounding_settings)
        if detector is None:
            raise ServiceUnavailable("SAM3 is loading. Please try again shortly.")
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
        return render_template("cases.html", input_size=INPUT_SIZE,
                               bbox_mode=node.bbox_visualization_mode)

    @app.get("/api/status")
    def status():
        cases = [path.name for path in gallery_groups(node.root)]
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
                bbox_visualization_mode=node.bbox_visualization_mode,
                grounding=node.grounding_settings,
                grounding_model_id=GROUNDING_MODEL_ID,
                grounding_backend="trt",
                grounding_precision=getattr(getattr(node, "grounding_detector", None), "precision", None),
                grounding_execution="subprocess",
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
                camera_fps=getattr(node, "camera_fps", None),
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
        for path in gallery_images(folder):
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
            raise ValueError("Expected prompt, detection_interval and/or grounding/recognition thresholds.")
        with node.lock:
            settings = {**node.grounding_settings, **payload}
            settings["prompt"] = validate_settings(**settings)
            if settings.get("detection_interval", 1) > 1:
                kcf_factory()
            detector = getattr(node, "grounding_detector", None)
            if detector is None and "prompt" in payload:
                return jsonify(error="SAM3 is loading. Please try again once it is ready."), 503
            if detector is not None:
                detector.validate_prompt(settings["prompt"])
            node.grounding_settings = settings
            invalidate_results()
        return jsonify(ok=True, grounding=settings)

    @app.post("/api/cases")
    def upload():
        name = request.form.get("case", "")
        folder = case_path(node.root, name)
        files = request.files.getlist("images")
        if not files:
            raise ValueError("Choose at least one picture.")
        # Validate all pictures before modifying the case.
        images = [decode_image(f.stream) for f in files]
        folder.mkdir(exist_ok=True)
        for i, image in enumerate(images):
            dest = folder / f"{time.time_ns()}_{i}.png"
            temp = dest.with_suffix(".tmp")
            image.save(temp, format="PNG")
            temp.replace(dest)
        with node.lock:
            node.selected = name
            invalidate_results()
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
            image = decode_image(request.files["image"].stream)
        with node.lock:
            node.source = mode
            node.frame = image
            node.frame_seq += 1
            invalidate_results()
        return jsonify(ok=True)

    def stream(attribute):
        if hasattr(node, "stream_requested_at"):
            with node.lock:
                node.stream_requested_at[attribute] = time.monotonic()
        def frames():
            tracked = hasattr(node, "stream_viewers")
            if tracked:
                with node.lock:
                    node.stream_viewers[attribute] = node.stream_viewers.get(attribute, 0) + 1
            try:
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
            finally:
                if tracked:
                    with node.lock:
                        node.stream_viewers[attribute] -= 1

        return Response(
            frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.get("/camera.mjpg")
    def camera_stream():
        return stream("preview_jpeg")

    @app.get("/overlay.mjpg")
    def overlay_stream():
        return stream("jpeg")

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
        attribute = "jpeg"
        with node.lock:
            if hasattr(node, "stream_requested_at"):
                node.stream_requested_at[attribute] = time.monotonic()
            data = getattr(node, attribute, None)
        return Response(
            data or b"",
            status=200 if data else 204,
            mimetype="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    return app
