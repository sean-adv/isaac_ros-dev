"""ROS-independent, metre-based RGB-D geometry. Pixel boxes use exclusive maxima."""
from dataclasses import dataclass, field
from itertools import product

import cv2
import numpy as np


@dataclass
class Config:
    min_depth: float = 0.15
    max_depth: float = 2.0
    depth_scale: float = 0.001
    pixel_stride: int = 2
    roi: list = field(default_factory=lambda: [0, 0, 0, 0])
    background_depth: float = 1.1
    foreground_threshold: float = 0.015
    connectivity: int = 8
    depth_jump_threshold: float = 0.0  # metres; zero keeps binary connectivity
    min_component_area: int = 40
    max_component_area: int = 30000
    filter_physical_size: bool = False
    min_width: float = 0.015
    max_width: float = 0.30
    min_height: float = 0.015
    max_height: float = 0.30
    min_depth_size: float = 0.005
    max_depth_size: float = 0.20
    bbox_padding_ratio: float = 0.05

    def validate(self):
        for name, value in vars(self).items():
            if not np.all(np.isfinite(value)):
                raise ValueError(f'{name} must be finite')
        for name in ('pixel_stride', 'min_component_area', 'max_component_area'):
            value = getattr(self, name)
            if int(value) != value or value < 1:
                raise ValueError(f'Invalid positive integer parameter: {name}')
        for low, high in (('min_depth', 'max_depth'), ('min_width', 'max_width'),
                          ('min_height', 'max_height'), ('min_depth_size', 'max_depth_size')):
            if not 0 <= getattr(self, low) < getattr(self, high):
                raise ValueError(f'Expected 0 <= {low} < {high}')
        if self.min_component_area > self.max_component_area:
            raise ValueError('min_component_area must not exceed max_component_area')
        if self.depth_scale <= 0 or self.background_depth <= 0 or self.foreground_threshold <= 0:
            raise ValueError('Depth scale, background depth and foreground threshold must be positive')
        if self.bbox_padding_ratio < 0 or self.connectivity not in (4, 8):
            raise ValueError('Padding must be nonnegative; connectivity must be 4 or 8')
        if self.depth_jump_threshold < 0:
            raise ValueError('depth_jump_threshold must be nonnegative (metres)')
        if len(self.roi) != 4 or any(int(v) != v or v < 0 for v in self.roi):
            raise ValueError('roi must contain four nonnegative pixel coordinates')
        if self.roi != [0, 0, 0, 0] and not (self.roi[2] > self.roi[0] and self.roi[3] > self.roi[1]):
            raise ValueError('roi must be [xmin, ymin, xmax, ymax], or all zeros')


@dataclass
class Candidate:
    centroid: np.ndarray
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    obb_center: np.ndarray
    rotation: np.ndarray
    dimensions: np.ndarray
    eigenvalues: np.ndarray
    corners: np.ndarray
    point_count: int
    bbox: tuple = ()
    candidate_id: str = ''


def depth_in_meters(depth, encoding, cfg):
    if encoding == '16UC1':
        z = depth.astype(np.float64) * cfg.depth_scale
    elif encoding == '32FC1':
        z = depth.astype(np.float64)
    else:
        raise ValueError(f'Unsupported depth encoding {encoding}; use 16UC1 or 32FC1')
    if z.ndim != 2:
        raise ValueError('Depth must have one channel')
    return z


def valid_depth(z, cfg):
    return np.isfinite(z) & (z > 0) & (z >= cfg.min_depth) & (z <= cfg.max_depth)


def foreground_mask(z, cfg):
    """Select valid surfaces sufficiently closer than the configured background."""
    return valid_depth(z, cfg) & ((cfg.background_depth - z) > cfg.foreground_threshold)


def depth_connected_components(z, mask, cfg):
    """Label foreground with local depth continuity; return labels and CC stats.

    Floating-range flood fill compares neighbouring depths, not the seed depth.
    It preserves gradual slopes but cannot separate surfaces joined by a gradual
    depth bridge. Invalid depth and pixels outside the ROI remain blocked.
    """
    if cfg.depth_jump_threshold == 0:
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=cfg.connectivity)
        return labels, stats

    height, width = z.shape
    image = np.ascontiguousarray(z, dtype=np.float32)
    # OpenCV requires a one-pixel border; nonzero mask values block traversal.
    blocked = np.ones((height + 2, width + 2), dtype=np.uint8)
    blocked[1:-1, 1:-1] = ~mask
    labels = np.zeros(z.shape, dtype=np.int32)
    stats = [(0, 0, 0, 0, 0)]  # background stats are unused by the caller
    flags = cfg.connectivity | cv2.FLOODFILL_MASK_ONLY | (2 << 8)
    for y in range(height):
        # Skip already-filled rows rather than scanning every pixel in Python.
        for x in np.flatnonzero(blocked[y + 1, 1:-1] == 0):
            if blocked[y + 1, x + 1]:
                continue
            area, _, _, rect = cv2.floodFill(
                image, blocked, (int(x), y), 0,
                loDiff=cfg.depth_jump_threshold, upDiff=cfg.depth_jump_threshold,
                flags=flags)
            left, top, w, h = rect
            region = blocked[top + 1:top + h + 1, left + 1:left + w + 1]
            filled = region == 2
            labels[top:top + h, left:left + w][filled] = len(stats)
            region[filled] = 1
            stats.append((left, top, w, h, area))
    return labels, np.asarray(stats, dtype=np.int32)


def depth_to_pointcloud(depth, encoding, intrinsics, cfg, distortion=None):
    z = depth_in_meters(depth, encoding, cfg)
    fx, fy, cx, cy = intrinsics
    if not np.all(np.isfinite(intrinsics)) or min(fx, fy) <= 0:
        raise ValueError('Invalid camera intrinsics')
    h, w = z.shape
    x0, y0, x1, y1 = cfg.roi if cfg.roi != [0, 0, 0, 0] else [0, 0, w, h]
    if x1 > w or y1 > h:
        raise ValueError('ROI exceeds image dimensions')
    v, u = np.mgrid[y0:y1:cfg.pixel_stride, x0:x1:cfg.pixel_stride]
    z = z[v, u]
    valid = np.isfinite(z) & (z > 0) & (z >= cfg.min_depth) & (z <= cfg.max_depth)
    z, u, v = z[valid], u[valid], v[valid]
    if distortion is not None and len(z):
        matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
        pixels = np.column_stack((u, v)).astype(float).reshape(-1, 1, 2)
        rays = cv2.undistortPoints(pixels, matrix, np.asarray(distortion, dtype=float)).reshape(-1, 2)
        return np.column_stack((rays * z[:, None], z))
    return np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))


def compute_object_obb(points):
    center = points.mean(axis=0)
    values, rotation = np.linalg.eigh((points - center).T @ (points - center) / max(1, len(points) - 1))
    local = (points - center) @ rotation
    low, high = local.min(axis=0), local.max(axis=0)
    # Largest, middle, smallest *observed* extents; not semantic package axes.
    order = np.argsort(high - low)[::-1]
    rotation, low, high, values = rotation[:, order], low[order], high[order], values[order]
    corners = np.array(list(product(*zip(low, high)))) @ rotation.T + center
    return Candidate(center, points.min(axis=0), points.max(axis=0),
                     ((low + high) / 2) @ rotation.T + center,
                     rotation, high - low, values, corners, len(points))


def filter_candidate(candidate, cfg):
    low = np.array([cfg.min_width, cfg.min_height, cfg.min_depth_size])
    high = np.array([cfg.max_width, cfg.max_height, cfg.max_depth_size])
    return bool(np.all(candidate.dimensions >= low) and np.all(candidate.dimensions <= high))


def project_points_to_image(points, intrinsics, shape, padding=0.0, pixel_margin=0, distortion=None):
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0)]
    if not len(points):
        return None
    fx, fy, cx, cy = intrinsics
    uv = points[:, :2] / points[:, 2, None] * [fx, fy] + [cx, cy]
    if distortion is not None:
        matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
        uv, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), matrix, np.asarray(distortion, dtype=float))
        uv = uv.reshape(-1, 2)
    low = np.floor(uv.min(axis=0)) - pixel_margin
    high = np.ceil(uv.max(axis=0)) + 1 + pixel_margin
    pad = (high - low) * padding
    h, w = shape[:2]
    low = np.maximum(0, np.floor(low - pad)).astype(int)
    high = np.minimum([w, h], np.ceil(high + pad)).astype(int)
    if np.any(high <= low):
        return None
    return int(low[0]), int(low[1]), int(high[0]), int(high[1])


def generate_proposals(depth, encoding, intrinsics, cfg, distortion=None):
    """Return candidates, sampled foreground XYZ, and a full-resolution mask."""
    z = depth_in_meters(depth, encoding, cfg)
    points = depth_to_pointcloud(depth, encoding, intrinsics, cfg, distortion)
    mask = foreground_mask(z, cfg)
    h, w = z.shape
    x0, y0, x1, y1 = cfg.roi if cfg.roi != [0, 0, 0, 0] else [0, 0, w, h]
    mask[:y0] = False
    mask[y1:] = False
    mask[:, :x0] = False
    mask[:, x1:] = False
    # Pixel stride affects 3D diagnostics only, never segmentation or bbox edges.
    sampled_valid = valid_depth(z[y0:y1:cfg.pixel_stride, x0:x1:cfg.pixel_stride], cfg)
    sampled_mask = mask[y0:y1:cfg.pixel_stride, x0:x1:cfg.pixel_stride]
    points = points[sampled_mask[sampled_valid]]
    labels, stats = depth_connected_components(z, mask, cfg)
    candidates = []
    for label in range(1, len(stats)):
        x, y, width, height, area = stats[label]
        if not cfg.min_component_area <= area <= cfg.max_component_area:
            continue
        # Backproject only this component's pixels for optional 3D metadata.
        v, u = np.nonzero(labels[y:y + height, x:x + width] == label)
        u, v = u + x, v + y
        depths = z[v, u]
        fx, fy, cx, cy = intrinsics
        rays = np.column_stack(((u - cx) / fx, (v - cy) / fy))
        if distortion is not None:
            matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
            pixels = np.column_stack((u, v)).astype(float).reshape(-1, 1, 2)
            rays = cv2.undistortPoints(pixels, matrix, np.asarray(distortion, dtype=float)).reshape(-1, 2)
        candidate = compute_object_obb(np.column_stack((rays * depths[:, None], depths)))
        if cfg.filter_physical_size and not filter_candidate(candidate, cfg):
            continue
        px, py = width * cfg.bbox_padding_ratio, height * cfg.bbox_padding_ratio
        candidate.bbox = (max(0, int(np.floor(x - px))), max(0, int(np.floor(y - py))),
                          min(w, int(np.ceil(x + width + px))), min(h, int(np.ceil(y + height + py))))
        candidates.append(candidate)
    candidates.sort(key=lambda c: (c.bbox[0], c.bbox[1], c.centroid[2]))
    for i, candidate in enumerate(candidates):
        candidate.candidate_id = f'candidate_{i}'
    return candidates, points, mask
