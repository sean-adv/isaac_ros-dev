"""HSV image processing used by gallery background-removal controls."""

import cv2
import numpy as np
from PIL import Image


def remove_background(image, options):
    """Match circular hue and S/V, optionally flood-fill only the clicked region."""
    if not isinstance(options, dict):
        raise ValueError("Provide background removal settings.")

    def number(name, default, maximum):
        value = options.get(name, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or not 0 <= value <= maximum
        ):
            raise ValueError(f"{name} must be between 0 and {maximum}.")
        return float(value)

    x = number("x", None, image.width - 1)
    y = number("y", None, image.height - 1)
    if not x.is_integer() or not y.is_integer():
        raise ValueError("Sample coordinates must be integers.")
    x, y = int(x), int(y)
    h = number("hue", 15, 180) / 2  # UI degrees; OpenCV uses half degrees.
    s = number("saturation", 50, 255)
    v = number("value", 50, 255)
    feather = number("feather", 0, 10)
    connected = options.get("connected", True)
    if not isinstance(connected, bool):
        raise ValueError("connected must be a boolean.")
    rgba = np.array(image.convert("RGBA"))
    if rgba[y, x, 3] == 0:
        raise ValueError("Click a background point that is not already transparent.")
    hsv = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2HSV).astype(np.float32)
    sample = hsv[y, x]
    hue_distance = np.abs(hsv[..., 0] - sample[0])
    hue_distance = np.minimum(hue_distance, 180 - hue_distance)
    # Hue is unstable near grayscale; S and V still constrain those matches.
    hue_match = (hue_distance <= h) | ((sample[1] <= 20) & (hsv[..., 1] <= 20))
    mask = (
        hue_match
        & (np.abs(hsv[..., 1] - sample[1]) <= s)
        & (np.abs(hsv[..., 2] - sample[2]) <= v)
        & (rgba[..., 3] > 0)
    ).astype(np.uint8)
    if connected:
        cv2.floodFill(mask, None, (x, y), 2, flags=4)
        mask = (mask == 2).astype(np.uint8)
    removal = mask.astype(np.float32)
    if feather > 0:
        removal = cv2.GaussianBlur(removal, (0, 0), feather)
    rgba[..., 3] = np.rint(rgba[..., 3] * (1 - removal)).astype(np.uint8)
    # Erase hidden background RGB as well; RGB-only consumers otherwise recover it.
    rgba[rgba[..., 3] == 0, :3] = 0
    return Image.fromarray(rgba)

