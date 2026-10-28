"""Optional luminance-only correction for SAM3; caller-owned RGB stays unchanged."""
import math

import cv2
import numpy as np
from PIL import Image


def validate_preprocessing(clahe_enabled=False, clahe_clip_limit=1.5,
                           clahe_tile_grid_size=8, clahe_blend=0.5):
    if type(clahe_enabled) is not bool:
        raise ValueError("SAM3 clahe_enabled must be boolean")
    if (isinstance(clahe_clip_limit, bool) or not isinstance(clahe_clip_limit, (int, float))
            or not math.isfinite(clahe_clip_limit) or clahe_clip_limit <= 0):
        raise ValueError("SAM3 clahe_clip_limit must be finite and positive")
    if type(clahe_tile_grid_size) is not int or not 1 <= clahe_tile_grid_size <= 64:
        raise ValueError("SAM3 clahe_tile_grid_size must be an integer between 1 and 64")
    if (isinstance(clahe_blend, bool) or not isinstance(clahe_blend, (int, float))
            or not math.isfinite(clahe_blend) or not 0 <= clahe_blend <= 1):
        raise ValueError("SAM3 clahe_blend must be between 0 and 1")
    return dict(clahe_enabled=clahe_enabled, clahe_clip_limit=float(clahe_clip_limit),
                clahe_tile_grid_size=clahe_tile_grid_size, clahe_blend=float(clahe_blend))


def prepare_sam3_image(image, *, clahe_enabled=False, clahe_clip_limit=1.5,
                       clahe_tile_grid_size=8, clahe_blend=0.5):
    rgb = image.convert("RGB")
    if not clahe_enabled or clahe_blend == 0:
        return rgb
    lab = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2LAB)
    lightness = lab[..., 0]
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit,
                           tileGridSize=(clahe_tile_grid_size, clahe_tile_grid_size))
    # Blend only L; a/b are preserved until conversion back to RGB.
    lab[..., 0] = cv2.addWeighted(lightness, 1 - clahe_blend,
                                 clahe.apply(lightness), clahe_blend, 0)
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
