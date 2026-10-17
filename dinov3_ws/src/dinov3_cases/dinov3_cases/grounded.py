"""Text-guided proposals followed by per-object DINOv3 gallery retrieval."""

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

from .runtime import INPUT_SIZE
from .gallery_files import EXTENSIONS, gallery_groups, gallery_images
# Preserve existing imports used by integrations and diagnostic tools.
from .object_visualization import (
    ObjectVisualizationSnapshot, build_object_diagnostic,
    encode_object_diagnostics, object_debug_views,
)

from .kcf import validate_detection_interval


DEFAULT_PROMPT = "medicine bottle, medicine box."


def validate_settings(prompt, box_threshold, recognition_threshold, margin, detection_interval=1):
    validate_detection_interval(detection_interval)
    if not isinstance(prompt, str) or not prompt.strip(" .\t\r\n") or len(prompt) > 1000:
        raise ValueError("Prompt must contain 1–1000 characters.")
    for name, value, upper in [
        ("box_threshold", box_threshold, 1),
        ("recognition_threshold", recognition_threshold, 1),
        ("recognition_margin", margin, 2),
    ]:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= value <= upper
        ):
            raise ValueError(f"{name} must be a finite number between 0 and {upper}.")
    return prompt.strip().rstrip(".") + "."


def validate_bbox_limits(min_width=0, min_height=0, min_area=0):
    for name, value in (("min_width", min_width), ("min_height", min_height),
                        ("min_area", min_area)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")


def clean_proposals(boxes, scores, labels, size, nms_iou=0.5, limit=20,
                    *, min_width=0, min_height=0, min_area=0):
    """Clip original-image XYXY boxes; suppress duplicate prompts, never union boxes."""
    validate_bbox_limits(min_width, min_height, min_area)
    width, height = size
    candidates = []
    for box, score, label in zip(boxes, scores, labels):
        if len(box) != 4 or not np.isfinite([*box, score]).all():
            continue
        x0, y0, x1, y1 = box
        x0, y0 = max(0, math.floor(x0)), max(0, math.floor(y0))
        x1, y1 = min(width, math.ceil(x1)), min(height, math.ceil(y1))
        if x1 <= x0 or y1 <= y0:
            continue
        box_width, box_height = x1 - x0, y1 - y0
        area = box_width * box_height
        if box_width < min_width or box_height < min_height or area < min_area:
            continue
        candidates.append(
            dict(
                bbox=[x0, y0, x1, y1],
                detection_score=float(score),
                detection_label=str(label),
                area=area,
            )
        )
    kept = []
    for item in sorted(candidates, key=lambda x: x["detection_score"], reverse=True):
        a = item["bbox"]
        duplicate = False
        for other in kept:
            b = other["bbox"]
            intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                0, min(a[3], b[3]) - max(a[1], b[1])
            )
            if intersection / (item["area"] + other["area"] - intersection) > nms_iou:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
            if len(kept) >= limit:
                break
    return kept


def prepare_object(image, max_side=INPUT_SIZE, stride=16, minimum=32):
    """Apply the same aspect-preserving resize/padding to references and proposals."""
    if max_side < minimum or stride <= 0:
        raise ValueError(
            "Object size must cover the minimum input size and stride must be positive."
        )
    image = image.convert("RGB")
    scale = max_side / max(image.size)
    width, height = [max(1, round(n * scale)) for n in image.size]
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    padded_w = padded_h = math.ceil(max_side / stride) * stride
    pixels = np.pad(
        np.asarray(image), ((0, padded_h - height), (0, padded_w - width), (0, 0)), mode="edge"
    )
    return Image.fromarray(pixels)


@torch.inference_mode()
def encode_object(encoder, image, max_side, stride):
    prepared = prepare_object(image, max_side, stride)
    features = encoder.encode(prepared)
    vector = features.float().mean(dim=0)
    if not torch.isfinite(vector).all() or vector.norm() < 1e-8:
        raise ValueError("Invalid object descriptor.")
    return prepared, features, F.normalize(vector, dim=0)


def compute_lab_histogram(image):
    """L1-normalized joint L/a/b histogram (8x16x16), excluding transparent pixels.

    Use the original crop before model resize/padding so padding cannot bias color.
    """
    rgba = np.asarray(image.convert("RGBA"))
    mask = (rgba[..., 3] > 0).astype(np.uint8)
    if not mask.any():
        raise ValueError("Cannot compute Lab histogram for a fully transparent image.")
    lab = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2LAB)
    histogram = cv2.calcHist([lab], [0, 1, 2], mask, [8, 16, 16], [0, 256] * 3)
    histogram = histogram.reshape(-1)
    return histogram / histogram.sum()


def gallery_signature(root):
    entries = []
    for folder in gallery_groups(root):
        for path in gallery_images(folder):
            stat = path.stat()
            entries.append((folder.name, str(path), stat.st_mtime_ns, stat.st_size))
    return tuple(entries)


class ObjectGallery:
    def __init__(self):
        self.key = None
        self.features = None
        self.labels = []
        self.paths = []
        self.patch_banks = []
        self.reference_inputs = []
        self.lab_histograms = None

    def refresh(self, root, encoder, max_side, stride):
        signature = gallery_signature(root)
        key = (signature, max_side, stride)
        if key == self.key:
            return False
        features, labels, patch_banks, reference_inputs = [], [], [], []
        lab_histograms = []
        for label, filename, _, _ in signature:
            with Image.open(filename) as source:
                if source.width * source.height > 25_000_000:
                    raise ValueError(f"Reference exceeds 25 megapixels: {filename}")
                image = ImageOps.exif_transpose(source).convert("RGBA")
            prepared, patches, descriptor = encode_object(encoder, image, max_side, stride)
            lab_histograms.append(compute_lab_histogram(image))
            features.append(descriptor)
            patch_banks.append(patches)
            reference_inputs.append(prepared)
            labels.append(label)
        # Publish a complete gallery only after all references have encoded successfully.
        self.features = torch.stack(features) if features else None
        self.lab_histograms = np.stack(lab_histograms) if lab_histograms else None
        self.labels, self.key = labels, key
        self.paths = [entry[1] for entry in signature]
        self.patch_banks = patch_banks
        self.reference_inputs = reference_inputs
        return True

    def similarities(self, descriptor, lab_histogram):
        """Fuse each reference's DINO cosine with its Lab Bhattacharyya coefficient."""
        dino = (self.features @ descriptor).clamp(-1, 1).cpu().numpy()
        lab = np.sqrt(self.lab_histograms * lab_histogram).sum(axis=1).clip(0, 1)
        return 0.8 * dino + 0.2 * lab, dino, lab

    def best_reference(self, descriptor, case="", *, lab_histogram):
        indices = [i for i, label in enumerate(self.labels) if not case or label == case]
        if not indices:
            return None, None
        scores, _, _ = self.similarities(descriptor, lab_histogram)
        best = max(indices, key=lambda i: scores[i])
        return best, float(scores[best])

    def classify(self, descriptor, threshold, margin, *, lab_histogram):
        if self.features is None:
            return dict(
                label="unknown", score=None, candidate=None, margin=None, reason="empty_gallery"
            )
        scores, dino, lab = self.similarities(descriptor, lab_histogram)
        similarities = scores.tolist()
        by_class = {}
        for label, score in zip(self.labels, similarities):
            by_class[label] = max(by_class.get(label, -1.0), score)
        ranked = sorted(by_class.items(), key=lambda item: (-item[1], item[0]))
        label, score = ranked[0]
        best_reference = max(
            (i for i, name in enumerate(self.labels) if name == label),
            key=lambda i: similarities[i],
        )
        gap = score - ranked[1][1] if len(ranked) > 1 else None
        if score < threshold:
            reason = "low_similarity"
        elif gap is not None and gap < margin:
            reason = "ambiguous"
        else:
            reason = "matched"
        return dict(
            label=label if reason == "matched" else "unknown",
            score=score,
            dino_score=float(dino[best_reference]),
            lab_score=float(lab[best_reference]),
            candidate=label,
            margin=gap,
            reason=reason,
            reference=dict(case=label, name=Path(self.paths[best_reference]).name),
        )


def recognize_objects(
    image,
    proposals,
    encoder,
    gallery,
    max_side,
    stride,
    threshold,
    margin,
    diagnostics=None,
    defer_diagnostics=False,
):
    results = []
    for proposal in proposals:
        roi = image.crop(proposal["bbox"])
        prepared, patches, descriptor = encode_object(encoder, roi, max_side, stride)
        lab_histogram = compute_lab_histogram(roi)
        classification = gallery.classify(
            descriptor, threshold, margin, lab_histogram=lab_histogram
        )
        result = dict(proposal, **classification)
        if diagnostics is not None:
            # The crop already owns its pixels; retain it for on-demand diagnostics.
            context = dict(
                roi=roi,
                roi_size=roi.size,
                prepared=prepared,
                patches=patches,
                descriptor=descriptor,
                lab_histogram=lab_histogram,
            )
            diagnostics.append(
                {"context": context}
                if defer_diagnostics
                else build_object_diagnostic(context, gallery, max_side, stride)
            )
        results.append(result)
    return results



class TrackedObjectRecognizer:
    """Reuse classifications in stable KCF list order within one tracker set.

    A new tracker-set object (or a new synchronous SAM3 frame) is a new token.
    Count changes discard cached classifications and encode the current crops.
    No spatial matching or public numeric IDs are used across tracker sets.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.token = None
        self.classifications = None
        self.diagnostics = []
        self.diagnostic_proposals = []
        self.classification_key = None
        self.encoded = False

    @property
    def diagnostic_detections(self):
        return [dict(proposal, **classification)
                for proposal, classification in zip(self.diagnostic_proposals,
                                                      self.classifications or [])]

    def update(self, pixels, proposals, encoder, gallery, max_side, stride,
               threshold, margin, *, token, gallery_revision):
        key = (gallery_revision, threshold, margin)
        self.encoded = False
        if not proposals:
            changed = (token != self.token or self.classifications != []
                       or key != self.classification_key)
            self.reset()
            self.token, self.classification_key = token, key
            self.classifications = []
            return [], [], changed

        # A surviving subset can change size without replacing the tracker object.
        # Never zip it with old labels; discard the complete cache and encode all
        # current crops, including after recovery from an empty set.
        if (token == self.token and self.classifications is not None
                and len(proposals) != len(self.classifications)):
            self.reset()
        fresh = token != self.token or self.classifications is None
        changed = fresh or key != self.classification_key
        if fresh:
            diagnostics = []
            detections = recognize_objects(
                Image.fromarray(pixels), proposals, encoder, gallery, max_side, stride,
                threshold, margin, diagnostics=diagnostics, defer_diagnostics=True,
            )
            fields = ("label", "score", "dino_score", "lab_score",
                      "candidate", "margin", "reason", "reference")
            classifications = [{k: item[k] for k in fields if k in item}
                               for item in detections]
            # Commit only a complete successful recognition of the new set.
            self.token = token
            self.diagnostics = diagnostics
            self.diagnostic_proposals = [dict(p, bbox=list(p["bbox"])) for p in proposals]
            self.classifications = classifications
            self.classification_key = key
            self.encoded = True
        elif changed:
            # Reclassify cached descriptors without another encoder forward.
            self.classifications = [
                gallery.classify(
                    item["context"]["descriptor"], threshold, margin,
                    lab_histogram=item["context"]["lab_histogram"],
                )
                for item in self.diagnostics
            ]
            self.classification_key = key
        detections = [dict(p, **c) for p, c in zip(proposals, self.classifications)]
        return detections, self.diagnostics, changed


def recognition_message(items, header=None):
    from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

    message = Detection2DArray()
    if header is not None:
        message.header = header
    for item in items:
        if item["label"] == "unknown":
            continue
        x0, y0, x1, y1 = item["bbox"]
        detection = Detection2D()
        detection.header = message.header
        detection.id = str(item.get("track_id", ""))
        detection.bbox.center.position.x = (x0 + x1) / 2.0
        detection.bbox.center.position.y = (y0 + y1) / 2.0
        detection.bbox.size_x = float(x1 - x0)
        detection.bbox.size_y = float(y1 - y0)
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = item["label"]
        hypothesis.hypothesis.score = item["score"] if item["score"] is not None else 0.0
        detection.results = [hypothesis]
        message.detections.append(detection)
    return message
