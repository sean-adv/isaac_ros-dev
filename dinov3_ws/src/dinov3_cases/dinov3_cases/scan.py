"""One frozen scan session and a small HTTP contract for a grasp planner."""

from copy import deepcopy
import math
import time
from uuid import uuid4

STAGES = ("segmentation", "cloud", "preprocess", "geometry", "candidates",
          "scoring", "collision", "confidence", "ready")
STATES = STAGES + ("rescan", "moving", "fusion", "failed")


def vector(value, size):
    if (not isinstance(value, list) or len(value) != size
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(x) for x in value)):
        raise ValueError(f"Expected {size} finite numbers.")
    return value


def new_session(item, revision):
    scan = deepcopy(item.get("scan", {}))
    stage = scan.get("stage", "segmentation")
    scan.update(id=uuid4().hex, revision=revision, label=item["label"],
                bbox=item["bbox"], state=stage, view_count=1,
                confidence=None, threshold=0.8, grasp_pose=None,
                updated_at=time.monotonic(), started_at=time.monotonic(),
                checks={}, candidates=[], completed=list(STAGES[:STAGES.index(stage)]))
    if item.get("pose"):
        scan["surface_pose"] = deepcopy(item["pose"])
        scan["reason"] = "等待抓取規劃器：表面 PCA 姿態尚未通過接觸及夾爪檢查。"
    else:
        scan["state"] = "rescan" if scan.get("points") else "failed"
        scan["reason"] = ("3D bbox 尚未提供抓取規劃所需點雲。" if item.get("bbox3d")
                          else item.get("bbox3d_status", item.get("pose_status", "missing_depth")))
    return scan


def planner_update(session, payload):
    """Validate atomically; rejected results never replace the previous result."""
    state = payload.get("state")
    if state not in STATES:
        raise ValueError("Invalid scan state.")
    if payload.get("frame_id") != session.get("frame_id"):
        raise ValueError("Planner frame_id must match the scan optical frame.")
    result = deepcopy(session)
    result.update(state=state, reason=str(payload.get("reason", ""))[:500],
                  updated_at=time.monotonic(), grasp_pose=None, confidence=None)
    if state in ("rescan", "moving", "fusion", "geometry"):
        result["completed"] = list(STAGES[:3])
        result["candidates"] = []
    confidence = payload.get("confidence")
    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1.")
        result["confidence"] = confidence
    if "points" in payload:
        points = payload["points"]
        if not isinstance(points, list) or not 1 <= len(points) <= 1200:
            raise ValueError("points must contain 1–1200 preview XYZ points in meters.")
        result["points"] = [vector(p, 3) for p in points]
        count = payload.get("point_count", len(points))
        if type(count) is not int or count < len(points):
            raise ValueError("Invalid point_count.")
        result["point_count"] = count
    if "view_count" in payload:
        count = payload["view_count"]
        if type(count) is not int or count < session["view_count"] or count > 100:
            raise ValueError("view_count must be a nondecreasing integer up to 100.")
        result["view_count"] = count
    if "completed" in payload:
        completed = payload["completed"]
        if not isinstance(completed, list) or any(s not in STAGES[:-1] for s in completed):
            raise ValueError("Invalid completed stages.")
        result["completed"] = list(dict.fromkeys(completed))
    if "candidates" in payload:
        candidates = payload["candidates"]
        if not isinstance(candidates, list) or len(candidates) > 64:
            raise ValueError("At most 64 candidates.")
        result["candidates"] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError("Each candidate must be a pose object.")
            result["candidates"].append(dict(position=vector(candidate.get("position"), 3),
                                             quaternion=vector(candidate.get("quaternion"), 4)))
    checks = payload.get("checks", {})
    if not isinstance(checks, dict) or any(type(v) is not bool for v in checks.values()):
        raise ValueError("checks must contain boolean results.")
    result["checks"] = checks
    if state == "ready":
        required = ("antipodal", "contact", "clearance", "collision_free", "unknown_space_clear")
        if any(checks.get(key) is not True for key in required):
            raise ValueError("A final grasp requires all five planner checks to pass.")
        if confidence is None or confidence < result["threshold"]:
            raise ValueError("Final grasp confidence is below the required threshold.")
        pose = payload.get("grasp_pose", {})
        if not isinstance(pose, dict):
            raise ValueError("grasp_pose must be a pose object.")
        position = vector(pose.get("position"), 3)
        quaternion = vector(pose.get("quaternion"), 4)
        if abs(sum(x*x for x in quaternion) - 1) > 0.01:
            raise ValueError("grasp_pose quaternion must have unit length.")
        width = pose.get("width")
        if isinstance(width, bool) or not isinstance(width, (int, float)) or not math.isfinite(width) or width <= 0:
            raise ValueError("grasp_pose width must be positive meters.")
        result["grasp_pose"] = dict(position=position, quaternion=quaternion,
                                    width=width, frame_id=result["frame_id"])
        result["completed"] = list(STAGES)
    return result


def register_scan(app, node):
    from flask import jsonify, request
    session = None

    @app.get("/api/scan")
    def get_scan():
        with node.lock:
            if session is None:
                return jsonify(session=None)
            data = deepcopy(session)
            data["age_seconds"] = time.monotonic() - data.pop("updated_at")
            data["elapsed_seconds"] = time.monotonic() - data.pop("started_at")
        return jsonify(session=data)

    @app.post("/api/scan")
    def start_scan():
        nonlocal session
        payload = request.get_json(silent=True) or {}
        index = payload.get("index")
        with node.lock:
            if type(index) is not int or not 0 <= index < len(node.detections):
                raise ValueError("Choose a current detected object.")
            if payload.get("revision") != node.object_debug_revision:
                return jsonify(error="辨識結果已更新，請重新選擇物體。"), 409
            if node.overlay_at is None or time.monotonic() - node.overlay_at > 3:
                return jsonify(error="RGB-D 結果已過期，請等待相機更新。"), 409
            item = node.detections[index]
            if item["label"] == "unknown" or not item.get("scan", {}).get("points"):
                raise ValueError("物體需要已識別身分與有效 RGB-D 點雲。")
            session = new_session(item, node.object_debug_revision)
            return jsonify(id=session["id"])

    @app.delete("/api/scan")
    def reset_scan():
        nonlocal session
        with node.lock:
            session = None
        return jsonify(ok=True)

    @app.post("/api/scan/result")
    def update_scan():
        nonlocal session
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise ValueError("Expected a planner result object.")
        with node.lock:
            if session is None or payload.get("id") != session["id"]:
                return jsonify(error="Scan session expired or replaced."), 409
            if session["state"] in ("ready", "failed"):
                return jsonify(error="Scan session has finished; start a new scan."), 409
            session = planner_update(session, payload)
        return jsonify(ok=True)
