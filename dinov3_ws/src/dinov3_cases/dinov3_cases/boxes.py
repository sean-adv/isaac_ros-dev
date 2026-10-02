"""Patch-mask boxes in original-image half-open xyxy coordinates."""
from dataclasses import dataclass
import math
import cv2
import numpy as np


@dataclass(frozen=True)
class BoxConfig:
    closing_kernel: int = 3
    min_component_patches: int = 4
    padding_patches: float = 0.5
    rgb_border_fraction: float = 0.1
    edge_min_gradient: float = 20.0
    edge_min_support: float = 0.6
    merge_iou_threshold: float = 0.3

    def __post_init__(self):
        if self.closing_kernel < 1 or self.closing_kernel % 2 != 1:
            raise ValueError('closing_kernel must be positive and odd.')
        if self.min_component_patches < 1:
            raise ValueError('min_component_patches must be positive.')
        if not math.isfinite(self.padding_patches) or self.padding_patches < 0:
            raise ValueError('padding_patches must be finite and nonnegative.')
        if not 0 <= self.rgb_border_fraction <= .25:
            raise ValueError('rgb_border_fraction must be in [0, 0.25]; 0 disables RGB.')
        if not math.isfinite(self.edge_min_gradient) or self.edge_min_gradient <= 0:
            raise ValueError('edge_min_gradient must be finite and positive.')
        if not 0 < self.edge_min_support <= 1:
            raise ValueError('edge_min_support must be in (0, 1].')
        if not math.isfinite(self.merge_iou_threshold) or not 0 < self.merge_iou_threshold <= 1:
            raise ValueError('merge_iou_threshold must be in (0, 1].')


def clean_mask(scores, valid, threshold, kernel):
    mask = ((scores >= threshold) & valid).astype(np.uint8)
    # Explicit background prevents closing from growing along grid boundaries.
    padded = np.pad(mask, kernel)
    padded = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, np.ones((kernel, kernel), np.uint8))
    mask = padded[kernel:-kernel, kernel:-kernel] & valid.astype(np.uint8)
    flooded = np.pad(mask, 1)
    cv2.floodFill(flooded, None, (0, 0), 1)
    return (mask | (1 - flooded[1:-1, 1:-1])) & valid.astype(np.uint8)


def _components(scores, valid, threshold, kernel, min_area):
    mask = clean_mask(scores, valid, threshold, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue
        # Filled holes/closed gaps do not invent similarity evidence.
        region = np.s_[y:y+h, x:x+w]
        evidence = ((labels[region] == label) & valid[region]
                    & (scores[region] >= threshold))
        if evidence.any():
            components.append((x, y, w, h, area, float(scores[region][evidence].mean())))
    return mask, components


def similarity_to_bboxes(global_similarity, threshold=.65, min_area=2048,
                         bbox_padding=8, closing_kernel=3, *, bgr=None, cfg=None,
                         return_mask=False):
    """Post-process only the merged pixel map; areas are mask pixels, xyxy is half-open."""
    if not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError('threshold must be between -1 and 1.')
    if not isinstance(min_area, int) or min_area < 1:
        raise ValueError('min_area must be a positive integer in pixels.')
    if not math.isfinite(bbox_padding) or bbox_padding < 0:
        raise ValueError('bbox_padding must be finite and nonnegative.')
    if closing_kernel < 1 or closing_kernel % 2 != 1:
        raise ValueError('closing_kernel must be positive and odd.')
    valid = np.isfinite(global_similarity)
    mask, components = _components(global_similarity, valid, threshold, closing_kernel, min_area)
    height, width = global_similarity.shape
    boxes = []
    for x, y, w, h, area, score in components:
        box = (max(0, math.floor(x - bbox_padding)), max(0, math.floor(y - bbox_padding)),
               min(width, math.ceil(x + w + bbox_padding)), min(height, math.ceil(y + h + bbox_padding)))
        if bgr is not None and cfg is not None:
            box = refine_box(bgr, box, cfg)
        boxes.append(dict(bbox=[int(v) for v in box], score=score, area=int(area)))
    return (boxes, mask) if return_mask else boxes


def merge_boxes_by_iou(boxes, threshold=0.3):
    """Merge connected groups of overlapping half-open boxes after refinement."""
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError('IoU threshold must be in (0, 1].')
    parent = list(range(len(boxes)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i, first in enumerate(boxes):
        ax0, ay0, ax1, ay1 = first['bbox']
        a_area = (ax1 - ax0) * (ay1 - ay0)
        for j in range(i + 1, len(boxes)):
            bx0, by0, bx1, by1 = boxes[j]['bbox']
            intersection = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
            if not intersection:
                continue
            b_area = (bx1 - bx0) * (by1 - by0)
            if intersection / (a_area + b_area - intersection) >= threshold:
                parent[root(j)] = root(i)

    groups = {}
    for index, item in enumerate(boxes):
        group = groups.setdefault(root(index), [])
        group.append(item)
    merged = []
    for group in groups.values():
        merged.append(dict(
            bbox=[min(item['bbox'][0] for item in group),
                  min(item['bbox'][1] for item in group),
                  max(item['bbox'][2] for item in group),
                  max(item['bbox'][3] for item in group)],
            score=max(item['score'] for item in group),
            area=sum(item['area'] for item in group)))
    return merged


def _edge_position(bgr, positions, start, stop, vertical, origin, cfg):
    """Read only an outer strip and a one-pixel Sobel stencil halo."""
    if not len(positions) or stop - start < 3:
        return origin
    height, width = bgr.shape[:2]
    lo, hi = int(positions.min()), int(positions.max())
    if vertical:
        a, b = max(0, lo - 1), min(width, hi + 2)
        strip = bgr[start:stop, a:b]
    else:
        a, b = max(0, lo - 1), min(height, hi + 2)
        strip = bgr[a:b, start:stop]
    # Channel maximum retains chromatic edges of similar luminance.
    grad = np.abs(cv2.Sobel(strip, cv2.CV_32F, int(vertical), int(not vertical),
                           ksize=3, scale=1 / 4)).max(axis=2)
    responses = grad[:, positions - a].T if vertical else grad[positions - a, :]
    strong = responses >= cfg.edge_min_gradient
    support = strong.mean(axis=1)
    run = np.zeros(len(positions), dtype=int)
    longest = run.copy()
    for column in strong.T:
        run = np.where(column, run + 1, 0)
        longest = np.maximum(longest, run)
    reliable = (support >= cfg.edge_min_support) & (longest >= responses.shape[1] * .4)
    if not reliable.any():
        return origin
    distance = np.abs(positions - origin)
    quality = support * responses.mean(axis=1) / (1 + distance / max(1, len(positions)))
    quality[~reliable] = -1
    return int(positions[np.argmax(quality)])


def refine_box(bgr, box, cfg):
    if cfg.rgb_border_fraction == 0:
        return box
    x0, y0, x1, y1 = box
    height, width = bgr.shape[:2]
    dx = math.floor((x1 - x0) * cfg.rgb_border_fraction)
    dy = math.floor((y1 - y0) * cfg.rgb_border_fraction)
    # Outward only: RGB cannot trim DINO support or its padding.
    left = _edge_position(bgr, np.arange(max(0, x0 - dx), x0), y0, y1, True, x0, cfg)
    right = _edge_position(bgr, np.arange(x1, min(width, x1 + dx)), y0, y1, True, x1, cfg)
    top = _edge_position(bgr, np.arange(max(0, y0 - dy), y0), x0, x1, False, y0, cfg)
    bottom = _edge_position(bgr, np.arange(y1, min(height, y1 + dy)), x0, x1, False, y1, cfg)
    return left, top, right, bottom


def extract_boxes(scores, valid, geometry, image_size, patch_size, threshold, cfg, bgr=None):
    """CC on native grid, then undo letterbox and scale patch boundaries."""
    left, top, nw, nh, grid = geometry
    width, height = image_size
    if scores.shape != (grid, grid) or valid.shape != scores.shape:
        raise ValueError('Similarity and validity masks must match the native patch grid.')
    _, components = _components(scores, valid, threshold, cfg.closing_kernel, cfg.min_component_patches)
    boxes = []
    for x, y, w, h, area, score in components:
        pad = cfg.padding_patches
        x0 = max(0, math.floor(((x - pad) * patch_size - left) * width / nw))
        y0 = max(0, math.floor(((y - pad) * patch_size - top) * height / nh))
        x1 = min(width, math.ceil(((x + w + pad) * patch_size - left) * width / nw))
        y1 = min(height, math.ceil(((y + h + pad) * patch_size - top) * height / nh))
        if x1 <= x0 or y1 <= y0:
            continue
        box = (x0, y0, x1, y1)
        if bgr is not None:
            box = refine_box(bgr, box, cfg)
        boxes.append((box, score))
    return boxes


def detection_message(boxes, name, header=None):
    from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
    result = Detection2DArray()
    if header is not None:
        result.header = header
    for index, ((x0, y0, x1, y1), score) in enumerate(boxes):
        detection = Detection2D()
        detection.header = result.header
        detection.id = str(index)
        detection.bbox.center.position.x = (x0 + x1) / 2.
        detection.bbox.center.position.y = (y0 + y1) / 2.
        detection.bbox.size_x = float(x1 - x0)
        detection.bbox.size_y = float(y1 - y0)
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = name
        hypothesis.hypothesis.score = score
        detection.results = [hypothesis]
        result.detections.append(detection)
    return result


def draw_boxes(image, boxes):
    best = max(range(len(boxes)), key=lambda index: boxes[index][1], default=None)
    # Draw the highest scoring box last so its red outline stays visible.
    for index in [i for i in range(len(boxes)) if i != best] + ([best] if best is not None else []):
        (x0, y0, x1, y1), score = boxes[index]
        color = (0, 0, 255) if index == best else (60, 220, 80)
        cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), color, 2)
        cv2.putText(image, f'{score:.2f}', (x0, max(16, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
