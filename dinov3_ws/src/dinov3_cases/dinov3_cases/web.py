"""HTTP routes for the shared recognition node; inference stays in node.py."""

import time
from io import BytesIO
from flask import Flask, jsonify, request, Response, render_template
from PIL import Image
from werkzeug.exceptions import ServiceUnavailable
from .runtime import MODEL_ID, GROUNDING_MODEL_ID, INPUT_SIZE
from .patches import reference_grid_image
from .grounded import validate_settings
from .cat2real import prepare_cat2real_image
from .nvdcf import check_nvdcf
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
        if request.method != "GET" and not request.path.startswith("/api/video/"):
            node.image_changed.set()
            node.render_changed.set()
        return response

    def invalidate_results():
        """Caller holds node.lock; discard output from the previous input revision."""
        node.generation += 1
        node.video_at = None
        node.video_hub.reset()
        node.clear_object_debug()
        node.detections = []

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
                sam3_inference_events=dict(
                    stream=getattr(node, "sam3_event_stream", None),
                    events=list(getattr(node, "sam3_events", ())),
                ),
                overlay_ready=node.video_at is not None,
                bbox_visualization_mode=node.bbox_visualization_mode,
                grounding=node.grounding_settings,
                grounding_model_id=GROUNDING_MODEL_ID,
                grounding_backend="trt",
                grounding_schedule=(node.proposal_tracker.scheduler_metrics()
                                    if getattr(node, "proposal_tracker", None) is not None else None),
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
                video_error=node.video_error,
                overlay_age=(
                    None if node.video_at is None else time.monotonic() - node.video_at
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
                reference = prepare_cat2real_image(original, INPUT_SIZE)
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
        reference = prepare_cat2real_image(decode_image(path), INPUT_SIZE)
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
                check_nvdcf(node.param("nvdcf_config"))
            detector = getattr(node, "grounding_detector", None)
            if detector is None and "prompt" in payload:
                return jsonify(error="SAM3 is loading. Please try again once it is ready."), 503
            if detector is not None:
                detector.validate_prompt(settings["prompt"])
            node.grounding_settings = settings
            invalidate_results()
            node.status = "Settings applied: Preparing first detection and recognition…"
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

    @app.post("/api/video/offer")
    def video_offer():
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or not isinstance(data.get("sdp"), str):
            raise ValueError("Expected a JSON WebRTC SDP offer")
        return jsonify(node.webrtc.offer(data["sdp"], data.get("type")))

    @app.post("/api/video/peers/<peer_id>/heartbeat")
    def video_heartbeat(peer_id):
        return Response(status=204 if node.webrtc.heartbeat(peer_id) else 404)

    @app.delete("/api/video/peers/<peer_id>")
    def video_disconnect(peer_id):
        node.webrtc.disconnect(peer_id)
        return Response(status=204)

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

    return app
