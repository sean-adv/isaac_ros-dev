"""Visible-surface geometric poses from registered RGB-D (no CAD object frame)."""

import cv2
import numpy as np


def stamp_seconds(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def matched_inputs(header, width, height, depths, infos, tolerance):
    """Only registered, same-size, timestamp-matched depth may use RGB boxes."""
    if not header.frame_id or stamp_seconds(header) <= 0:
        return None
    candidates = [
        depth for depth in depths
        if depth.header.frame_id == header.frame_id
        and (depth.width, depth.height) == (width, height)
        and stamp_seconds(depth.header) > 0
        and abs(stamp_seconds(depth.header) - stamp_seconds(header)) <= tolerance
    ]
    if not candidates:
        return None
    depth = min(candidates, key=lambda d: abs(stamp_seconds(d.header) - stamp_seconds(header)))
    for info in reversed(infos):
        if (info.header.frame_id == header.frame_id
                and (info.width, info.height) == (width, height)
                and stamp_seconds(info.header) <= stamp_seconds(header) + tolerance):
            return depth, info
    return None


def rotation_quaternion(rotation):
    """Convert a proper rotation matrix to unit quaternion in ROS x,y,z,w order."""
    # Eigenvector formula remains stable at 180-degree rotations.
    r = rotation
    matrix = np.array([
        [r[0, 0]-r[1, 1]-r[2, 2], r[0, 1]+r[1, 0], r[0, 2]+r[2, 0], r[2, 1]-r[1, 2]],
        [r[0, 1]+r[1, 0], r[1, 1]-r[0, 0]-r[2, 2], r[1, 2]+r[2, 1], r[0, 2]-r[2, 0]],
        [r[0, 2]+r[2, 0], r[1, 2]+r[2, 1], r[2, 2]-r[0, 0]-r[1, 1], r[1, 0]-r[0, 1]],
        [r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1], np.trace(r)],
    ]) / 3.0
    _, vectors = np.linalg.eigh(matrix)
    q = vectors[:, -1]
    return q if q[3] >= 0 else -q



def draw_pose_axes(image, pose, info, axis_length=0.08):
    """Project object-local axes into the registered RGB image (length in meters)."""
    position = np.asarray(pose["position"], dtype=np.float64)
    quaternion = np.asarray(pose["quaternion"], dtype=np.float64)
    if (not np.isfinite(position).all() or not np.isfinite(quaternion).all()
            or np.linalg.norm(quaternion) < 1e-12):
        return
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    rotation = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    points = np.vstack((position, position + axis_length * rotation.T))
    # Do not project points on or behind the camera plane.
    if np.any(points[:, 2] <= 1e-6):
        return
    k = np.asarray(info.k, dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(info.d, dtype=np.float64)
    zero = np.zeros(3, dtype=np.float64)
    if info.distortion_model == "equidistant" and distortion.size == 4:
        projected, _ = cv2.fisheye.projectPoints(
            points.reshape(-1, 1, 3), zero, zero, k, distortion
        )
    elif info.distortion_model in ("plumb_bob", "rational_polynomial", ""):
        projected, _ = cv2.projectPoints(
            points, zero, zero, k, distortion if distortion.size else None
        )
    else:
        return
    pixels = projected.reshape(-1, 2)
    if not np.isfinite(pixels).all() or np.max(np.abs(pixels)) > 1e7:
        return
    pixels = np.rint(pixels).astype(int)
    origin = tuple(map(int, pixels[0]))
    height, width = image.shape[:2]
    for endpoint, label, color in zip(
        pixels[1:], ("X", "Y", "Z"), ((0, 0, 255), (0, 255, 0), (255, 0, 0))
    ):
        endpoint = tuple(map(int, endpoint))
        visible, start, end = cv2.clipLine((0, 0, width, height), origin, endpoint)
        if not visible:
            continue
        cv2.arrowedLine(image, start, end, (0, 0, 0), 5, cv2.LINE_AA, tipLength=0.2)
        cv2.arrowedLine(image, start, end, color, 3, cv2.LINE_AA, tipLength=0.2)
        # An axis along the viewing direction projects to a dot.
        if start == end:
            cv2.circle(image, end, 4, color, -1, cv2.LINE_AA)
        text_origin = (max(0, min(end[0] + 6, width - 20)),
                       max(18, min(end[1] - 6, height - 6)))
        cv2.putText(image, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA)


def estimate_surface_pose(depth, info, bbox, *, min_depth=0.1, max_depth=5.0,
                          depth_band=0.08, min_points=80, axis_ratio=1.15):
    """Segment a central depth component, unproject, then fit its PCA frame.

    Position is the visible-surface centroid in meters. Z is the least-variance
    surface axis, pointing away from the camera; X is the longest surface axis,
    with a deterministic sign. Degenerate axes cannot yield a measured rotation.
    """
    h, w = depth.shape
    if (info.width, info.height) != (w, h):
        raise ValueError("camera_info_size_mismatch")
    k = np.asarray(info.k, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("invalid_camera_intrinsics")
    if info.binning_x > 1 or info.binning_y > 1 or info.roi.x_offset or info.roi.y_offset:
        raise ValueError("unsupported_camera_info_roi_or_binning")
    x0, y0, x1, y1 = map(int, bbox)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("empty_bbox")
    roi = depth[y0:y1, x0:x1]
    valid = np.isfinite(roi) & (roi >= min_depth) & (roi <= max_depth)
    rh, rw = roi.shape
    cy, cx = rh // 2, rw // 2
    dy, dx = max(1, rh // 10), max(1, rw // 10)
    center = roi[max(0, cy-dy):cy+dy+1, max(0, cx-dx):cx+dx+1]
    seeds = center[np.isfinite(center) & (center >= min_depth) & (center <= max_depth)]
    if seeds.size < 5:
        raise ValueError("no_valid_center_depth")
    foreground = valid & (np.abs(roi - np.median(seeds)) <= depth_band)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), 8)
    central_labels = labels[max(0, cy-dy):cy+dy+1, max(0, cx-dx):cx+dx+1]
    choices = np.unique(central_labels[central_labels > 0])
    if count <= 1 or choices.size == 0:
        raise ValueError("no_depth_component")
    component = max(choices, key=lambda label: stats[label, cv2.CC_STAT_AREA])
    v, u = np.nonzero(labels == component)
    if len(u) < min_points:
        raise ValueError("insufficient_depth_points")
    # Bound CPU work without making point density depend on distance.
    step = max(1, int(np.ceil(len(u) / 6000)))
    v, u = v[::step], u[::step]
    z = roi[v, u]
    pixels = np.stack((u + x0, v + y0), axis=1).astype(np.float64).reshape(-1, 1, 2)
    distortion = np.asarray(info.d, dtype=np.float64)
    if not np.isfinite(distortion).all():
        raise ValueError("invalid_distortion")
    if info.distortion_model in ("plumb_bob", "rational_polynomial", ""):
        rays = cv2.undistortPoints(pixels, k, distortion if distortion.size else None)
    elif info.distortion_model == "equidistant" and distortion.size == 4:
        rays = cv2.fisheye.undistortPoints(pixels, k, distortion)
    else:
        raise ValueError("unsupported_distortion_model")
    points = np.column_stack((rays[:, 0, :] * z[:, None], z))
    if not np.isfinite(points).all():
        raise ValueError("invalid_unprojected_points")
    position = points.mean(axis=0)
    centered = points - position
    values, axes = np.linalg.eigh(centered.T @ centered / len(points))
    if (values[1] <= 1e-10 or values[2] / values[1] < axis_ratio
            or values[1] / max(values[0], 1e-12) < axis_ratio):
        raise ValueError("ambiguous_surface_axes")
    x, normal = axes[:, 2].copy(), axes[:, 0].copy()
    if np.dot(normal, position) < 0:
        normal = -normal
    dominant = int(np.argmax(np.abs(x[:2])))
    if x[dominant] < 0:
        x = -x
    y = np.cross(normal, x)
    rotation = np.column_stack((x, y, normal))
    local = centered @ rotation
    return dict(
        position=position.tolist(),
        quaternion=rotation_quaternion(rotation).tolist(),
        size=(2 * np.max(np.abs(local), axis=0)).tolist(),
        point_count=len(points),
        method="visible_surface_pca",
    )


def attach_surface_poses(items, depth, info, **options):
    for item in items:
        item.pop("pose", None)
        if item["label"] == "unknown":
            item["pose_status"] = "unknown"
            continue
        try:
            item["pose"] = estimate_surface_pose(depth, info, item["bbox"], **options)
            item["pose_status"] = "ok"
        except (ValueError, cv2.error, np.linalg.LinAlgError) as error:
            item["pose_status"] = str(error)


def poses_message(items, header):
    from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

    message = Detection3DArray(header=header)
    for item in items:
        if item["label"] == "unknown" or "pose" not in item:
            continue
        data = item["pose"]
        detection = Detection3D(header=header)
        detection.id = str(item["track_id"]) if "track_id" in item else ""
        pose = detection.bbox.center
        pose.position.x, pose.position.y, pose.position.z = data["position"]
        (pose.orientation.x, pose.orientation.y,
         pose.orientation.z, pose.orientation.w) = data["quaternion"]
        (detection.bbox.size.x, detection.bbox.size.y,
         detection.bbox.size.z) = data["size"]
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = item["label"]
        hypothesis.hypothesis.score = float(item["score"] or 0.0)
        hypothesis.pose.pose = pose
        # Conservative, uncalibrated uncertainty: surface centroid != object center.
        hypothesis.pose.covariance = np.diag([0.01, 0.01, 0.01, 1.0, 1.0, 1.0]).ravel().tolist()
        detection.results = [hypothesis]
        message.detections.append(detection)
    return message
