"""Camera-aligned robust bounds of a central registered-depth component."""

import cv2
import numpy as np


def estimate_bbox3d(depth, info, bbox, *, min_depth=0.1, max_depth=5.0,
                    depth_band=0.08, min_points=80, quantile=0.02):
    """Segment around the central median Z and bound its visible point cloud.

    Quantiles trim each XYZ axis independently; this is not a strict enclosure
    or an estimate of the unseen backside. Four corner-near samples and the
    nearest component point are diagnostics only, not inputs to the bounds.
    """
    if not np.isfinite(depth_band) or depth_band <= 0:
        raise ValueError("invalid_bbox_depth_band")
    if not np.isfinite(quantile) or not 0 <= quantile < 0.5:
        raise ValueError("invalid_bbox_quantile")
    if not isinstance(min_points, (int, np.integer)) or min_points < 5:
        raise ValueError("invalid_bbox_min_points")
    if not (np.isfinite(min_depth) and np.isfinite(max_depth)
            and 0 < min_depth < max_depth):
        raise ValueError("invalid_depth_range")
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
    left, top = int(x0), int(y0)
    right, bottom = int(np.ceil(x1)), int(np.ceil(y1))
    roi = depth[top:bottom, left:right]
    valid = np.isfinite(roi) & (roi >= min_depth) & (roi <= max_depth)
    if not valid.any():
        raise ValueError("no_valid_bbox_depth")

    # Central patch spans roughly 20% of the ROI in each dimension.
    rh, rw = roi.shape
    cy, cx = rh // 2, rw // 2
    dy, dx = max(1, rh // 10), max(1, rw // 10)
    patch = (slice(max(0, cy-dy), min(rh, cy+dy+1)),
             slice(max(0, cx-dx), min(rw, cx+dx+1)))
    seeds = roi[patch][valid[patch]]
    if seeds.size < 5:
        raise ValueError("no_valid_center_depth")
    seed_depth = float(np.median(seeds))
    # Compare bounds directly to avoid arithmetic on NaN/Inf holes.
    foreground = valid & (roi >= seed_depth-depth_band) & (roi <= seed_depth+depth_band)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), 8)
    central_labels = labels[patch]
    choices = np.unique(central_labels[central_labels > 0])
    if choices.size == 0:
        raise ValueError("no_depth_component")
    component = max(choices, key=lambda label: stats[label, cv2.CC_STAT_AREA])
    v, u = np.nonzero(labels == component)
    if len(u) < min_points:
        raise ValueError("insufficient_depth_points")
    # Use the whole selected component: no diagonal-ray or sparse-grid bias.
    pixels = np.column_stack((u+left, v+top)).astype(np.float64)
    depths = roi[v, u].astype(np.float64)
    distortion = np.asarray(info.d, dtype=np.float64)
    if not np.isfinite(distortion).all():
        raise ValueError("invalid_distortion")
    image_points = pixels.reshape(-1, 1, 2)
    if info.distortion_model in ("plumb_bob", "rational_polynomial", ""):
        rays = cv2.undistortPoints(image_points, k, distortion if distortion.size else None)
    elif info.distortion_model == "equidistant" and distortion.size == 4:
        rays = cv2.fisheye.undistortPoints(image_points, k, distortion)
    else:
        raise ValueError("unsupported_distortion_model")
    points = np.column_stack((rays[:, 0, :] * depths[:, None], depths))
    if not np.isfinite(points).all():
        raise ValueError("invalid_unprojected_points")
    lower, upper = np.quantile(points, [quantile, 1-quantile], axis=0)
    size = upper-lower
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError("empty_projected_bbox")

    # Preserve internal visualization diagnostics; these points need not lie inside
    # the quantile box, and are not measured corners of the physical object.
    corner_pixels = np.array([(left, top), (right-1, top),
                              (right-1, bottom-1), (left, bottom-1)])
    samples = [int(np.argmin(np.sum((pixels-corner)**2, axis=1)))
               for corner in corner_pixels]
    nearest = int(np.argmin(depths))
    return dict(position=((lower+upper)/2).tolist(), size=size.tolist(),
                corners=points[samples].tolist(), corner_depths=depths[samples].tolist(),
                corner_pixels=corner_pixels.tolist(),
                corner_sample_pixels=pixels[samples].astype(int).tolist(),
                nearest_point=points[nearest].tolist(), nearest_depth=float(depths[nearest]),
                nearest_pixel=pixels[nearest].astype(int).tolist(),
                seed_depth=seed_depth, point_count=len(points),
                foreground_depth_band=float(depth_band), quantile=float(quantile),
                depth_range=[float(lower[2]), float(upper[2])],
                method="central_depth_component_quantiles")


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
    """Eight camera-aligned vertices, used by visualization."""
    lower = np.asarray(data["position"]) - np.asarray(data["size"]) / 2
    upper = np.asarray(data["position"]) + np.asarray(data["size"]) / 2
    return np.array([(x, y, z) for z in (lower[2], upper[2])
                     for x, y in [(lower[0], lower[1]), (upper[0], lower[1]),
                                  (upper[0], upper[1]), (lower[0], upper[1])]])


def bboxes3d_message(items, header):
    """Standard detections: visible-component box center, dimensions and class.

    Positions and dimensions are meters in header.frame_id. The pose uses the
    box center as its reference point; its identity quaternion describes the
    camera-aligned box axes, not an estimated physical object orientation.
    Covariance remains at the message default; uncertainty is not estimated.
    """
    from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
    from geometry_msgs.msg import Point

    message = Detection3DArray(header=header)
    for item in items:
        if item["label"] == "unknown" or "bbox3d" not in item:
            continue
        data = item["bbox3d"]
        detection = Detection3D(header=header)
        detection.id = str(item.get("track_id", ""))
        center = detection.bbox.center
        x, y, z = map(float, data["position"])
        center.position = Point(x=x, y=y, z=z)
        center.orientation.w = 1.0
        detection.bbox.size.x, detection.bbox.size.y, detection.bbox.size.z = map(float, data["size"])
        result = ObjectHypothesisWithPose()
        result.hypothesis.class_id = item["label"]
        result.hypothesis.score = float(item["score"] or 0.0)
        result.pose.pose.position = Point(x=x, y=y, z=z)
        result.pose.pose.orientation.w = 1.0
        detection.results = [result]
        message.detections.append(detection)
    return message
