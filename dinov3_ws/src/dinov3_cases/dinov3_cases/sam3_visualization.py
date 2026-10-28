"""Bounded full-frame SAM3 diagnostics, encoded once per inference for HTTP/IPC."""
import base64
from io import BytesIO

import numpy as np
from PIL import Image
import torch.nn.functional as F


def encode_mask_snapshot(image, masks, selected, prompt):
    preview = image.convert("RGB").copy()
    preview.thumbnail((960, 540), Image.Resampling.BILINEAR)
    width, height = preview.size
    union = np.zeros((height, width), dtype=bool)
    overlay = np.asarray(preview).copy()
    colors = [(55, 220, 160), (255, 160, 60), (100, 150, 255), (235, 100, 200)]
    # Resize logits before thresholding, as in the official postprocessor.
    # Work one query at a time to bound CPU/GPU memory regardless of image size.
    for index, (label, query) in enumerate(selected):
        logits = masks[label, query][None, None].float()
        active = (F.interpolate(logits, (height, width), mode="bilinear",
                                align_corners=False)[0, 0] > 0).cpu().numpy()
        union |= active
        overlay[active] = (overlay[active] * .5 + np.array(colors[index % len(colors)]) * .5).astype(np.uint8)

    def encode(value, format):
        stream = BytesIO()
        value.save(stream, format=format, **({"quality": 80} if format == "JPEG" else {}))
        return base64.b64encode(stream.getvalue()).decode("ascii")

    return dict(overlay=encode(Image.fromarray(overlay), "JPEG"),
                mask=encode(Image.fromarray(union.astype(np.uint8) * 255), "PNG"),
                original_size=list(image.size), size=[width, height],
                object_count=len(selected), prompt=prompt, mask_threshold=.5)
