"""Depth-derived visible surface coordinate frame."""
import cv2
import numpy as np


def stamp_seconds(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def closest_depth(frames, header, tolerance):
    candidates = [f for f in frames if f.header.frame_id == header.frame_id]
    if not header.frame_id or stamp_seconds(header) <= 0 or not candidates:
        raise ValueError('Waiting for depth aligned to the RGB optical frame')
    depth = min(candidates, key=lambda f: abs(stamp_seconds(f.header) - stamp_seconds(header)))
    if abs(stamp_seconds(depth.header) - stamp_seconds(header)) > tolerance:
        raise ValueError('No synchronized depth image')
    return depth


def estimate_pose(depth, encoding, camera_matrix, distortion, bbox, mask,
                  depth_band=.04, min_points=80):
    """Masked median-depth band and PCA; this is not a model-based object pose."""
    if encoding not in ('16UC1', '32FC1'):
        raise ValueError('Depth encoding must be 16UC1 (mm) or 32FC1 (m)')
    k = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
    if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError('Invalid camera intrinsics')
    if depth.ndim != 2 or depth.shape != mask.shape:
        raise ValueError('Depth must be aligned to the RGB image dimensions')
    z = depth.astype(np.float64) * (.001 if encoding == '16UC1' else 1.)
    roi = np.zeros(depth.shape, bool)
    x0, y0, x1, y1 = map(int, bbox)
    roi[max(0, y0):min(z.shape[0], y1), max(0, x0):min(z.shape[1], x1)] = True
    valid = roi & (mask > 0) & np.isfinite(z) & (z > .05) & (z < 10.)
    if valid.sum() < min_points:
        raise ValueError('Too few valid depth points in the selected box')
    valid &= np.abs(z - np.median(z[valid])) <= depth_band
    v, u = np.nonzero(valid)
    if len(u) < min_points:
        raise ValueError('Too few foreground depth points')
    stride = max(1, int(np.ceil(len(u) / 10000)))
    u, v = u[::stride], v[::stride]
    rays = cv2.undistortPoints(np.column_stack((u, v)).astype(float).reshape(-1, 1, 2),
                               k, np.asarray(distortion, float)).reshape(-1, 2)
    points = np.column_stack((rays * z[v, u, None], z[v, u]))
    center = points.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(points - center, rowvar=False))
    if values[1] < 1e-8 or values[0] > .3 * values[1]:
        raise ValueError('Depth surface has no reliable normal')
    if values[2] < 1.15 * values[1]:
        raise ValueError('Surface is symmetric; in-plane orientation is ambiguous')
    normal, x_axis = vectors[:, 0], vectors[:, 2]
    if np.dot(normal, center) > 0:
        normal = -normal
    if x_axis[np.argmax(np.abs(x_axis))] < 0:
        x_axis = -x_axis
    rotation = np.column_stack((x_axis, np.cross(normal, x_axis), normal))
    rvec = cv2.Rodrigues(rotation)[0].ravel()
    angle = np.linalg.norm(rvec)
    quaternion = np.r_[rvec * (np.sin(angle / 2) / angle if angle > 1e-12 else .5),
                       np.cos(angle / 2)]
    return dict(position=center.tolist(), quaternion_xyzw=quaternion.tolist(),
                axes=rotation.tolist(), points=len(points), method='visible_surface_pca')


def draw_axes(image, pose, camera_matrix, distortion, length):
    center = np.asarray(pose['position'])
    rotation = np.asarray(pose['axes'])
    points = np.vstack((center, center + length * rotation.T))
    if np.any(points[:, 2] <= .01):
        return
    pixels = cv2.projectPoints(points, np.zeros(3), np.zeros(3),
                               np.asarray(camera_matrix).reshape(3, 3),
                               np.asarray(distortion, float))[0].reshape(-1, 2)
    pixels = np.rint(pixels).astype(int)
    for endpoint, color, label in zip(pixels[1:], [(0, 0, 255), (0, 255, 0), (255, 0, 0)], 'XYZ'):
        cv2.arrowedLine(image, tuple(pixels[0]), tuple(endpoint), color, 2, tipLength=.15)
        cv2.putText(image, label, tuple(endpoint), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
