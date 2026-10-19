"""Plane-oriented robust bounds of a central registered-depth component."""

import cv2
import numpy as np
import open3d as o3d


def _undistort_pixels(pixels, k, distortion, model):
    image_points = pixels.reshape(-1, 1, 2)
    if model in ("plumb_bob", "rational_polynomial", ""):
        return cv2.undistortPoints(
            image_points, k, distortion if distortion.size else None)[:, 0, :]
    if model == "equidistant" and distortion.size == 4:
        return cv2.fisheye.undistortPoints(image_points, k, distortion)[:, 0, :]
    raise ValueError("unsupported_distortion_model")


class ProjectionRayCache:
    """One calibration's float64 rays, owned by the 3D processing worker.

    Rebuild on size, intrinsics, distortion coefficients or model changes.
    Rows are built in chunks to bound temporary allocation at full resolution.
    """

    def __init__(self):
        self._key = None
        self._rays = None

    def get(self, width, height, k, distortion, model):
        key = (width, height, tuple(k.ravel()), tuple(distortion), model)
        if key != self._key:
            # Release the previous calibration before allocating the next one.
            self._key, self._rays = None, None
            rays = np.empty((height, width, 2), dtype=np.float64)
            for top in range(0, height, 64):
                bottom = min(top + 64, height)
                v, u = np.indices((bottom - top, width))
                pixels = np.column_stack((u.ravel(), v.ravel() + top)).astype(np.float64)
                rays[top:bottom] = _undistort_pixels(
                    pixels, k, distortion, model).reshape(bottom - top, width, 2)
            rays.flags.writeable = False
            self._rays, self._key = rays, key
        return self._rays


def _surface_rotation(points, threshold, min_points):
    """Dominant plane + longest rectangle edge; columns map local to camera.

    Open3D estimates and refines the plane on the full component in native code.
    Normal points away from the camera; the long axis has its largest camera
    component positive. These conventions cannot resolve 180-degree symmetry.
    """
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    try:
        plane, indices = cloud.segment_plane(
            distance_threshold=threshold, ransac_n=3, num_iterations=128,
            probability=0.999,
        )
    except RuntimeError as error:
        raise ValueError("plane_segmentation_failed") from error
    normal = np.asarray(plane[:3], dtype=np.float64)
    length = np.linalg.norm(normal)
    if not np.isfinite(plane).all() or length < 1e-12:
        raise ValueError("degenerate_plane")
    if len(indices) < max(min_points, int(np.ceil(len(points) * .5))):
        raise ValueError("no_dominant_plane")
    normal /= length
    inliers = points[np.asarray(indices, dtype=np.intp)]
    origin = inliers.mean(axis=0)
    if np.dot(normal, origin) < 0:
        normal = -normal
    reference = np.eye(3)[np.argmin(np.abs(normal))]
    u = reference - np.dot(reference, normal) * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    planar = np.column_stack(((inliers - origin) @ u, (inliers - origin) @ v))
    rect = cv2.boxPoints(cv2.minAreaRect(planar.astype(np.float32)))
    edges = np.roll(rect, -1, axis=0) - rect
    lengths = np.linalg.norm(edges, axis=1)
    if lengths.min() < 1e-6:
        raise ValueError("degenerate_rectangle")
    edge = edges[np.argmax(lengths)]
    x_axis = edge[0] * u + edge[1] * v
    x_axis /= np.linalg.norm(x_axis)
    if x_axis[np.argmax(np.abs(x_axis))] < 0:
        x_axis = -x_axis
    rotation = np.column_stack((x_axis, np.cross(normal, x_axis), normal))
    status = "ambiguous_in_plane" if lengths.max() / lengths.min() < 1.05 else "ok"
    return rotation, len(indices), status


def _rotation_quaternion(rotation):
    """Return an xyzw unit quaternion, including rotations near 180 degrees."""
    vector, _ = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))
    vector = vector.ravel()
    angle = np.linalg.norm(vector)
    if angle < 1e-12:
        return [0., 0., 0., 1.]
    return [*(vector * (np.sin(angle / 2) / angle)), float(np.cos(angle / 2))]


def estimate_bbox3d(depth, info, bbox, *, min_depth=0.1, max_depth=5.0,
                    depth_band=0.08, min_points=80, quantile=0.02,
                    depth_scale=1.0, ray_cache=None, plane_threshold=0.005,
                    cloud_filter=None, cloud_state=None, cloud_frame=None):
    """Segment around the central median Z and bound its visible point cloud.

    RANSAC fits the visible plane; its longest rectangle edge sets local X.
    Quantiles in that coordinate system determine the box center and sizes.
    Local Z size is visible surface variation, not unseen physical thickness.
    Failed plane fits retain camera-aligned bounds with an explicit status.
    The box is not a strict enclosure or an estimate of the unseen backside.
    Four corner-near samples and the nearest component point are diagnostics
    only, not inputs to the bounds.
    depth_scale converts input units to meters inside the clipped ROI only.
    An optional worker-owned ray_cache reuses calibrated pixel directions.
    """
    if not np.isfinite(plane_threshold) or plane_threshold <= 0:
        raise ValueError("invalid_plane_threshold")
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
    # Convert only this bbox; preserve the original float64 arithmetic/order.
    roi = depth[top:bottom, left:right].astype(np.float64, copy=False)
    if depth_scale != 1.0:
        roi = roi * depth_scale
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
    if ray_cache is None:
        rays = _undistort_pixels(pixels, k, distortion, info.distortion_model)
    else:
        ray_map = ray_cache.get(w, h, k, distortion, info.distortion_model)
        rays = ray_map[v + top, u + left]
    points = np.column_stack((rays * depths[:, None], depths))
    if not np.isfinite(points).all():
        raise ValueError("invalid_unprojected_points")
    depth_range = np.quantile(depths, [quantile, 1-quantile])
    # Depth quantiles are available before orientation. Only these retained
    # samples reach RANSAC; box-local quantiles below still define the bounds.
    plane_points = points[(depths >= depth_range[0]) & (depths <= depth_range[1])]
    if len(plane_points) < min_points:
        raise ValueError("insufficient_filtered_depth_points")
    coverage = len(plane_points) / roi.size
    cloud_depth = float(np.median(plane_points[:, 2]))
    if cloud_filter is not None:
        cloud_filter.check(coverage, cloud_depth,
                           {} if cloud_state is None else cloud_state, cloud_frame)
    rotation = np.eye(3)
    plane_inliers = 0
    try:
        rotation, plane_inliers, orientation_status = _surface_rotation(
            plane_points, plane_threshold, min_points)
    except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
        orientation_status = "camera_aligned_fallback:" + str(error)
    origin = points.mean(axis=0)
    local = (points - origin) @ rotation
    lower, upper = np.quantile(local, [quantile, 1-quantile], axis=0)
    size = upper-lower
    position = origin + rotation @ ((lower+upper)/2)
    if orientation_status.startswith("camera_aligned_fallback:"):
        position[2] = float(np.mean(depths))
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError("empty_projected_bbox")

    # Preserve internal visualization diagnostics; these points need not lie inside
    # the quantile box, and are not measured corners of the physical object.
    corner_pixels = np.array([(left, top), (right-1, top),
                              (right-1, bottom-1), (left, bottom-1)])
    samples = [int(np.argmin(np.sum((pixels-corner)**2, axis=1)))
               for corner in corner_pixels]
    nearest = int(np.argmin(depths))
    return dict(position=position.tolist(), size=size.tolist(),
                rotation=rotation.tolist(), orientation_status=orientation_status,
                plane_inlier_count=plane_inliers, plane_threshold=float(plane_threshold),
                corners=points[samples].tolist(), corner_depths=depths[samples].tolist(),
                corner_pixels=corner_pixels.tolist(),
                corner_sample_pixels=pixels[samples].astype(int).tolist(),
                nearest_point=points[nearest].tolist(), nearest_depth=float(depths[nearest]),
                nearest_pixel=pixels[nearest].astype(int).tolist(),
                seed_depth=seed_depth, point_count=len(points),
                plane_input_count=len(plane_points), cloud_coverage=float(coverage),
                cloud_depth=cloud_depth,
                foreground_depth_band=float(depth_band), quantile=float(quantile),
                depth_range=depth_range.tolist(),
                method=("central_depth_component_mean_z" if orientation_status.startswith(
                    "camera_aligned_fallback:") else "central_depth_component_plane_rectangle"))


def attach_bboxes3d(items, depth, info, **options):
    for item in items:
        item.pop("pose", None)
        item.pop("pose_status", None)
        item.pop("bbox3d", None)
        try:
            item["bbox3d"] = estimate_bbox3d(
                depth, info, item["bbox"], cloud_state=item.get("_cloud_filter_state"), **options)
            item["bbox3d_status"] = "ok"
        except (ValueError, cv2.error) as error:
            item["bbox3d_status"] = str(error)


def bbox_vertices(data):
    """Eight camera-frame vertices using the same local axes as ROS poses."""
    half = np.asarray(data["size"]) / 2
    local = np.array([(x, y, z) for z in (-half[2], half[2])
                      for x, y in [(-half[0], -half[1]), (half[0], -half[1]),
                                   (half[0], half[1]), (-half[0], half[1])]])
    return local @ np.asarray(data.get("rotation", np.eye(3))).T + data["position"]


def bboxes3d_message(items, header):
    """Standard detections: visible-component position, dimensions and class.

    Positions are meters in header.frame_id; dimensions are along local axes.
    The quaternion maps box-local axes to header.frame_id, matching visualization.
    The pose bounds visible points and does not locate the unseen physical center.
    Covariance remains at the message default; uncertainty is not estimated.
    """
    from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
    from geometry_msgs.msg import Point, Quaternion

    message = Detection3DArray(header=header)
    for item in items:
        if not item.get("bbox_visible", True) or "bbox3d" not in item:
            continue
        data = item["bbox3d"]
        source = item.get("bbox3d_source")
        measured_header = header
        if source is not None:
            measured_header = type(header)(
                frame_id=source["frame_id"], stamp=type(header.stamp)(**source["stamp"]))
        detection = Detection3D(header=measured_header)
        detection.id = str(item.get("track_id", ""))
        center = detection.bbox.center
        x, y, z = map(float, data["position"])
        center.position = Point(x=x, y=y, z=z)
        qx, qy, qz, qw = map(float, _rotation_quaternion(data.get("rotation", np.eye(3))))
        center.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        detection.bbox.size.x, detection.bbox.size.y, detection.bbox.size.z = map(float, data["size"])
        result = ObjectHypothesisWithPose()
        result.hypothesis.class_id = item["label"]
        result.hypothesis.score = float(item["score"] or 0.0)
        result.pose.pose.position = Point(x=x, y=y, z=z)
        result.pose.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        detection.results = [result]
        message.detections.append(detection)
    if message.detections:
        # The array uses the newest measurement time; each detection retains its
        # own source time when cached objects were measured on different frames.
        message.header = max((item.header for item in message.detections),
                             key=lambda h: (h.stamp.sec, h.stamp.nanosec))
    return message
