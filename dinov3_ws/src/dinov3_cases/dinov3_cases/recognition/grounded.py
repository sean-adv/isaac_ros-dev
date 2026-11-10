"""Text-guided proposals followed by per-object DINOv3 gallery retrieval."""

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

from dinov3_cases.recognition.gallery_files import gallery_groups, gallery_images
from dinov3_cases.recognition.object_visualization import build_object_diagnostic

from dinov3_cases.tracking.nvdcf import validate_detection_interval


DEFAULT_PROMPT = "medicine bottle, medicine box."


def validate_settings(prompt, box_threshold, recognition_threshold, margin, detection_interval=0.0):
    """Validate external settings, including the accepted legacy margin field."""
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


def validate_containment_threshold(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("containment_threshold must be a finite number between 0 and 1.")


def clean_proposals(boxes, scores, labels, size, limit=20,
                    *, min_width=0, min_height=0, min_area=0, containment_threshold=0.95):
    """Clip XYXY boxes; remove contained boxes, then rank by score and limit."""
    validate_containment_threshold(containment_threshold)
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
    # Prefer larger boxes when intersection / smaller area reaches the threshold, even
    # across prompt labels or when the smaller box has a higher score.
    # Equal-area duplicates keep the higher score. Filter before the result
    # limit so discarded small boxes cannot consume detection slots.
    outer_boxes = []
    for item in sorted(candidates, key=lambda x: (x["area"], x["detection_score"]),
                       reverse=True):
        a = item["bbox"]
        for other in outer_boxes:
            b = other["bbox"]
            intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                0, min(a[3], b[3]) - max(a[1], b[1])
            )
            if containment_threshold > 0 and intersection / item["area"] >= containment_threshold:
                break
        else:
            outer_boxes.append(item)
    return sorted(outer_boxes, key=lambda item: item["detection_score"], reverse=True)[:limit]


@torch.inference_mode()
def encode_object(encoder, image, max_side, stride):
    """Prepare a gallery/profile image and normalize its explicit model descriptor."""
    prepared = encoder.prepare_image(image, max_side, stride)
    features, vector = encoder.encode_with_descriptor(prepared)
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
        """Return separate DINO cosine and Lab histogram similarities."""
        dino = (self.features @ descriptor).clamp(-1, 1).cpu().numpy()
        lab = np.sqrt(self.lab_histograms * lab_histogram).sum(axis=1).clip(0, 1)
        return dino, lab

    def _rank_references(self, dino, lab, indices):
        # Lab decides; DINO, class name and reference order break exact ties.
        return sorted(indices, key=lambda i: (-float(lab[i]), -float(dino[i]),
                                              self.labels[i], i))

    def best_reference(self, descriptor, case="", *, lab_histogram, threshold=-1):
        indices = [i for i, label in enumerate(self.labels) if not case or label == case]
        if not indices:
            return None, None
        dino, lab = self.similarities(descriptor, lab_histogram)
        if not case:
            eligible = np.flatnonzero(dino >= threshold).tolist()
            # An unknown object still shows its closest DINO reference for inspection.
            indices = eligible or [max(indices, key=lambda i: float(dino[i]))]
        best = self._rank_references(dino, lab, indices)[0]
        return best, float(dino[best])

    def classify(self, descriptor, threshold, *, lab_histogram):
        """Gate references by DINO, then choose the class with the highest Lab score."""
        if self.features is None:
            return dict(
                label="unknown", score=None, candidate=None, margin=None, reason="empty_gallery"
            )
        dino, lab = self.similarities(descriptor, lab_histogram)
        return self._classify_scores(dino, lab, threshold)

    def classify_batch(self, descriptors, threshold, *, lab_histograms):
        """One GPU matrix multiply/host transfer for all current object scores."""
        if self.features is None:
            return [dict(label="unknown", score=None, candidate=None, margin=None,
                         reason="empty_gallery") for _ in descriptors]
        dino = (descriptors @ self.features.T).clamp(-1, 1).cpu().numpy()
        lab = np.sqrt(np.asarray(lab_histograms)[:,None,:] * self.lab_histograms[None,:,:]).sum(axis=2).clip(0,1)
        return [self._classify_scores(d, l, threshold) for d,l in zip(dino,lab)]

    def _classify_scores(self, dino, lab, threshold):
        eligible = np.flatnonzero(dino >= threshold).tolist()
        ranked = self._rank_references(dino, lab, eligible)
        if ranked:
            best = ranked[0]
            # Report the Lab gap between distinct eligible classes for diagnostics only.
            runner = next((i for i in ranked if self.labels[i] != self.labels[best]), None)
            gap = float(lab[best] - lab[runner]) if runner is not None else None
            reason = "matched"
        else:
            best = max(range(len(dino)), key=lambda i: float(dino[i]))
            gap = None
            reason = "low_similarity"
        label = self.labels[best]
        return dict(
            label=label if reason == "matched" else "unknown",
            score=float(dino[best]),
            dino_score=float(dino[best]),
            lab_score=float(lab[best]),
            candidate=label,
            margin=gap,
            reason=reason,
            reference=dict(case=label, name=Path(self.paths[best]).name),
        )

def recognize_objects(
    image,
    proposals,
    encoder,
    gallery,
    max_side,
    stride,
    threshold,
    diagnostics=None,
    defer_diagnostics=False,
):
    results = []
    if not proposals:
        return results
    rois = [image.crop(proposal["bbox"]) for proposal in proposals]
    encoded = encoder.encode_regions(image, [proposal["bbox"] for proposal in proposals])
    if len(encoded) != len(proposals):
        raise ValueError("Encoder did not return one result per ROI")
    histograms = [compute_lab_histogram(roi) for roi in rois]
    if hasattr(gallery, 'classify_batch'):
        classifications = gallery.classify_batch(
            torch.stack([row[2] for row in encoded]), threshold, lab_histograms=histograms)
    else:
        classifications = [gallery.classify(row[2], threshold, lab_histogram=histogram)
                           for row,histogram in zip(encoded,histograms)]
    for proposal,roi,row,lab_histogram,classification in zip(proposals,rois,encoded,histograms,classifications):
        prepared, patches, descriptor = row
        result = dict(proposal, **classification)
        if diagnostics is not None:
            # The crop already owns its pixels; retain it for on-demand diagnostics.
            context = dict(
                recognition_threshold=threshold,
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
