"""Filter full-image boxes by a configurable camera region, without cropping pixels."""

DEFAULT_ROI = (0, 0, -1, -1)


def validate_roi(width, height, center_x, center_y):
    values = (width, height, center_x, center_y)
    for name, value, minimum in zip(
        ("width", "height", "center_x", "center_y"), values, (0, 0, -1, -1)
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"SAM3 region {name} must be an integer >= {minimum}.")
    return values


def roi_bounds(image_width, image_height, region):
    """Return clipped [left, top, right, bottom] in native camera pixels.

    Zero size keeps the full axis. Center -1 follows the image midpoint.
    Regions extending beyond the image are intersected with the image.
    """
    width, height, center_x, center_y = region

    def axis(length, size, center):
        if size == 0:
            return 0, length
        start = (length - size) // 2 if center == -1 else center - size // 2
        return max(0, min(length, start)), max(0, min(length, start + size))

    left, right = axis(image_width, width, center_x)
    top, bottom = axis(image_height, height, center_y)
    return left, top, right, bottom


def filter_roi(detections, bounds):
    """Keep only fully contained, positive-area boxes; preserve original coordinates."""
    if bounds is None:
        return detections
    left, top, right, bottom = bounds
    return [item for item in detections
            if left <= item["bbox"][0] < item["bbox"][2] <= right
            and top <= item["bbox"][1] < item["bbox"][3] <= bottom]
