"""Camera-aligned cuboids from RGB boxes and registered corner/nearest depths."""

import cv2
import numpy as np


def line_pixels(start, end):
    """Yield a Bresenham segment in order, including both endpoints."""
    x, y = start
    end_x, end_y = end
    dx, dy = abs(end_x-x), -abs(end_y-y)
    sx, sy = (1 if x < end_x else -1), (1 if y < end_y else -1)
    error = dx + dy
    while True:
        yield x, y
        if (x, y) == (end_x, end_y):
            break
        twice_error = 2 * error
        if twice_error >= dy:
            error += dy
            x += sx
        if twice_error <= dx:
            error += dx
            y += sy


def estimate_bbox3d(depth, info, bbox, *, min_depth=0.1, max_depth=5.0, depth_band=0.08, edge_min_jump=0.03):
    """Bound four measured corner points plus the ROI's nearest valid Z point.

    This bounds the sampled visible geometry, not an unseen object backside.
    Walk center-to-corner, choose the foreground side of the largest valid
    adjacent depth jump, and fall back to reasonable corner depth if no edge.
    """
    if not np.isfinite(depth_band) or depth_band <= 0:
        raise ValueError("invalid_bbox_depth_band")
    if not np.isfinite(edge_min_jump) or edge_min_jump <= 0:
        raise ValueError("invalid_bbox_edge_min_jump")
    h, w = depth.shape
    if (info.width, info.height) != (w, h):
        raise ValueError("camera_info_size_mismatch")
    k = np.asarray(info.k, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("invalid_camera_intrinsics")
    if info.binning_x > 1 or info.binning_y > 1 or info.roi.x_offset or info.roi.y_offset:
        raise ValueError("unsupported_camera_info_roi_or_binning")
    if len(bbox) != 4 or not np.isfinite(bbox).all():
        raise ValueError("invalid_bbox")
    x0, y0, x1, y1 = map(float, bbox)
    x0, y0, x1, y1 = max(0., x0), max(0., y0), min(float(w), x1), min(float(h), y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("empty_bbox")
    # Detection boxes use exclusive right/bottom bounds; sample inside the ROI.
    left, top = int(x0), int(y0)
    right, bottom = int(np.ceil(x1)), int(np.ceil(y1))
    roi = depth[top:bottom, left:right]
    valid = np.isfinite(roi) & (roi >= min_depth) & (roi <= max_depth)
    if not valid.any():
        raise ValueError("no_valid_bbox_depth")
    near_v, near_u = np.unravel_index(np.argmin(np.where(valid, roi, np.inf)), roi.shape)
    nearest_depth = float(roi[near_v, near_u])
    foreground_max_depth = nearest_depth + depth_band
    corner_pixels = [(left, top), (right-1, top), (right-1, bottom-1), (left, bottom-1)]
    center_pixel = (min(right-1, int((x0+x1)/2)), min(bottom-1, int((y0+y1)/2)))
    corner_depths, sampled_pixels, edge_jumps, sampling_methods = [], [], [], []
    for index, corner in enumerate(corner_pixels):
        line = list(line_pixels(center_pixel, corner))
        values = np.array([depth[v, u] for u, v in line])
        line_valid = np.isfinite(values) & (values >= min_depth) & (values <= max_depth)
        # Only consecutive valid pixels may form an edge: never bridge a hole.
        pair_valid = line_valid[:-1] & line_valid[1:]
        jumps = np.zeros(max(0, len(values)-1), dtype=np.float64)
        jumps[pair_valid] = np.abs(values[1:][pair_valid] - values[:-1][pair_valid])
        reasonable = line_valid & (values >= nearest_depth) & (values <= foreground_max_depth)
        candidates = np.flatnonzero(pair_valid & reasonable[:-1]
                                    & (jumps >= edge_min_jump))
        if candidates.size:
            # argmax picks the innermost edge on a tie. Take the center-side pixel.
            selected = int(candidates[np.argmax(jumps[candidates])])
            sampling_methods.append("max_depth_jump")
            edge_jumps.append(float(jumps[selected]))
        else:
            candidates = np.flatnonzero(reasonable)
            if not candidates.size:
                raise ValueError(f"no_reasonable_corner_depth:{index}")
            selected = int(candidates[-1])
            sampling_methods.append("reasonable_depth_fallback")
            edge_jumps.append(0.0)
        sampled_pixels.append(line[selected])
        corner_depths.append(float(values[selected]))
    pixels = np.array(sampled_pixels + [(int(near_u)+left, int(near_v)+top)],
                      dtype=np.float64).reshape(-1, 1, 2)
    depths = np.array(corner_depths + [nearest_depth])
    distortion = np.asarray(info.d, dtype=np.float64)
    if not np.isfinite(distortion).all():
        raise ValueError("invalid_distortion")
    if info.distortion_model in ("plumb_bob", "rational_polynomial", ""):
        rays = cv2.undistortPoints(pixels, k, distortion if distortion.size else None)
    elif info.distortion_model == "equidistant" and distortion.size == 4:
        rays = cv2.fisheye.undistortPoints(pixels, k, distortion)
    else:
        raise ValueError("unsupported_distortion_model")
    points = np.column_stack((rays[:, 0, :] * depths[:, None], depths))
    if not np.isfinite(points).all():
        raise ValueError("invalid_unprojected_points")
    lower, upper = points.min(axis=0), points.max(axis=0)
    size = upper - lower
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError("empty_projected_bbox")
    return dict(position=((lower + upper) / 2).tolist(), size=size.tolist(),
                corners=points[:4].tolist(), corner_depths=corner_depths,
                corner_pixels=[list(pixel) for pixel in corner_pixels],
                corner_sample_pixels=[list(pixel) for pixel in sampled_pixels],
                nearest_point=points[4].tolist(), nearest_depth=nearest_depth,
                nearest_pixel=[int(near_u)+left, int(near_v)+top],
                foreground_depth_band=float(depth_band), edge_min_jump=float(edge_min_jump),
                corner_edge_jumps=edge_jumps, corner_sampling_methods=sampling_methods,
                depth_range=[float(lower[2]), float(upper[2])],
                method="bbox_center_outward_depth_edges")


def attach_bboxes3d(items, depth, info, **options):
    for item in items:
        item.pop("pose", None)
        item.pop("pose_status", None)
        item.pop("bbox3d", None)
        try:
            item["bbox3d"] = estimate_bbox3d(depth, info, item["bbox"], **options)
            item["bbox3d_status"] = "ok"
        except (ValueError, cv2.error) as error:
            item["bbox3d_status"] = str(error)


def bbox_vertices(data):
    """Eight camera-aligned vertices, shared by ROS output and visualization."""
    lower = np.asarray(data["position"]) - np.asarray(data["size"]) / 2
    upper = np.asarray(data["position"]) + np.asarray(data["size"]) / 2
    return np.array([(x, y, z) for z in (lower[2], upper[2])
                     for x, y in [(lower[0], lower[1]), (upper[0], lower[1]),
                                  (upper[0], upper[1]), (lower[0], upper[1])]])


def bboxes3d_message(items, header):
    from dinov3_cases_msgs.msg import Detection3DPoints, Detection3DPointsArray, Point2D
    from geometry_msgs.msg import Point

    def pixel(value):
        return Point2D(x=float(value[0]), y=float(value[1]))

    def point(value):
        return Point(x=float(value[0]), y=float(value[1]), z=float(value[2]))

    message = Detection3DPointsArray(header=header)
    for item in items:
        if item["label"] == "unknown" or "bbox3d" not in item:
            continue
        data = item["bbox3d"]
        detection = Detection3DPoints(header=header)
        detection.id = str(item.get("track_id", ""))
        center = detection.bbox.center
        center.position = point(data["position"])
        center.orientation.w = 1.0  # Aligned with the camera optical axes.
        detection.bbox.size.x, detection.bbox.size.y, detection.bbox.size.z = data["size"]
        detection.hypothesis.class_id = item["label"]
        detection.hypothesis.score = float(item["score"] or 0.0)
        x0, y0, x1, y1 = item["bbox"]
        detection.bbox_2d.center.position.x = (x0+x1) / 2.0
        detection.bbox_2d.center.position.y = (y0+y1) / 2.0
        detection.bbox_2d.size_x, detection.bbox_2d.size_y = float(x1-x0), float(y1-y0)
        detection.bbox_2d_corners = [pixel(p) for p in
                                   [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]
        detection.sampled_2d_points = [pixel(p) for p in data["corner_sample_pixels"]]
        detection.sampled_3d_points = [point(p) for p in data["corners"]]
        detection.nearest_2d_point = pixel(data["nearest_pixel"])
        detection.nearest_3d_point = point(data["nearest_point"])
        detection.bbox_3d_corners = [point(p) for p in bbox_vertices(data)]
        message.detections.append(detection)
    return message
