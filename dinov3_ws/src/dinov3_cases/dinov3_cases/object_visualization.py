"""On-demand object diagnostics and bounded immutable result snapshots."""

import copy
import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch

def object_debug_views(roi, prepared, features, reference_patches, max_side, stride, roi_size=None):
    """Use the existing forward's patch tokens; remove input padding before ROI mapping."""
    from .patches import compute_similarity_maps, reference_grid_image

    gh, gw = prepared.height // stride, prepared.width // stride
    roi_size = roi.size if roi_size is None else roi_size
    scale = max_side / max(roi_size)
    content_w, content_h = [max(1, round(n * scale)) for n in roi_size]
    views = dict(
        crop=cv2.cvtColor(np.asarray(roi), cv2.COLOR_RGB2BGR),
        grid=cv2.cvtColor(np.asarray(reference_grid_image(prepared, stride)), cv2.COLOR_RGB2BGR),
    )
    # Mark the real ROI's edge; the right/bottom remainder is replicate padding.
    cv2.rectangle(views["grid"], (0, 0), (content_w - 1, content_h - 1), (255, 130, 50), 1)
    metadata = dict(
        roi_size=list(roi_size),
        input_size=list(prepared.size),
        preview_size=list(roi.size),
        content_size=[content_w, content_h],
        patch_grid=[gh, gw],
        heatmap_available=reference_patches is not None,
    )
    if reference_patches is None:
        return views, metadata
    scores = (
        compute_similarity_maps(features.reshape(1, gh, gw, -1), reference_patches)[0].cpu().numpy()
    )
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite object patch similarities.")

    def color(values):
        return cv2.applyColorMap((np.clip(values, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

    # Patch rendering shows the full model grid. The ROI heatmap excludes padding.
    views["patches"] = cv2.resize(color(scores), prepared.size, interpolation=cv2.INTER_NEAREST)
    pixels = cv2.resize(scores, prepared.size, interpolation=cv2.INTER_LINEAR)[
        :content_h, :content_w
    ]
    mapped = cv2.resize(pixels, roi.size, interpolation=cv2.INTER_LINEAR)
    views["similarity"] = color(mapped)
    views["heatmap"] = cv2.addWeighted(views["crop"], 0.55, views["similarity"], 0.45, 0)
    metadata.update(similarity_min=float(mapped.min()), similarity_max=float(mapped.max()))
    return views, metadata


def build_object_diagnostic(
    context, gallery, max_side, stride, reference_case="", reference_bank=None
):
    index, score = gallery.best_reference(
        context["descriptor"], reference_case, lab_histogram=context["lab_histogram"]
    )
    count = (
        sum(label == reference_case for label in gallery.labels)
        if reference_case
        else int(index is not None)
    )
    bank = None if index is None else gallery.patch_banks[index]
    if reference_case and index is not None:
        bank = (
            reference_bank
            if reference_bank is not None
            else torch.cat(
                [
                    patches
                    for label, patches in zip(gallery.labels, gallery.patch_banks)
                    if label == reference_case
                ]
            )
        )
    views, metadata = object_debug_views(
        context["roi"],
        context["prepared"],
        context["patches"],
        bank,
        max_side,
        stride,
        context["roi_size"],
    )
    reference = None
    dino_score = lab_score = None
    if index is not None:
        _, dino, lab = gallery.similarities(context["descriptor"], context["lab_histogram"])
        dino_score, lab_score = float(dino[index]), float(lab[index])
        reference = dict(case=gallery.labels[index], name=Path(gallery.paths[index]).name)
        views["reference"] = cv2.cvtColor(
            np.asarray(gallery.reference_inputs[index]), cv2.COLOR_RGB2BGR
        )
    metadata.update(
        comparison_case=reference_case or (reference["case"] if reference else None),
        comparison_score=score,
        comparison_dino_score=dino_score,
        comparison_lab_score=lab_score,
        comparison_reference=reference,
        comparison_reference_count=count,
    )
    return dict(metadata=metadata, views=views, context=context)


def encode_object_diagnostics(detections, diagnostics):
    images, items = {}, []
    for index, (item, diagnostic) in enumerate(zip(detections, diagnostics)):
        for view, pixels in diagnostic["views"].items():
            ok, data = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise RuntimeError(f"Object {index} {view} JPEG encoding failed.")
            images[index, view] = data.tobytes()
        items.append(
            dict(item, index=index, **diagnostic["metadata"], views=list(diagnostic["views"]))
        )
    return items, images


class ObjectVisualizationSnapshot:
    """Render reference comparisons from one immutable inference/gallery revision."""

    def __init__(self, detections, diagnostics, gallery, max_side, stride):
        self.detections = detections
        self.contexts = [item["context"] for item in diagnostics]
        # Gallery refresh replaces its lists/tensors; the snapshot keeps the old complete set.
        self.gallery = copy.copy(gallery)
        self.reference_cases = sorted(set(gallery.labels))
        self.max_side, self.stride = max_side, stride
        self.lock = threading.Lock()
        self.cache = OrderedDict()

    def render(self, reference_case=""):
        if reference_case and reference_case not in self.reference_cases:
            raise ValueError("Reference label has no images in this result.")
        with self.lock:
            if reference_case not in self.cache:
                bank = (
                    torch.cat(
                        [
                            patches
                            for label, patches in zip(self.gallery.labels, self.gallery.patch_banks)
                            if label == reference_case
                        ]
                    )
                    if reference_case
                    else None
                )
                diagnostics = [
                    build_object_diagnostic(
                        context, self.gallery, self.max_side, self.stride, reference_case, bank
                    )
                    for context in self.contexts
                ]
                self.cache[reference_case] = encode_object_diagnostics(self.detections, diagnostics)
                # Keep automatic views and at most two selected-label comparisons per frame.
                if len(self.cache) > 3:
                    self.cache.pop(next(key for key in self.cache if key))
            self.cache.move_to_end(reference_case)
            return self.cache[reference_case]


