"""Shared gallery paths, image decoding and deterministic file selection."""

import re
from pathlib import Path
from PIL import Image, ImageOps

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def group_path(root, name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name):
        raise ValueError("Labels must contain 1–80 letters, numbers, underscores, or hyphens.")
    path = root / name
    if path.is_symlink() or path.resolve().parent != root.resolve():
        raise ValueError("Invalid label path.")
    return path


def image_path(root, group, filename):
    folder = group_path(root, group)
    if (
        not isinstance(filename, str)
        or not filename
        or "\\" in filename
        or "/" in filename
        or Path(filename).name != filename
    ):
        raise ValueError("Invalid image filename.")
    path = folder / filename
    if (
        path.is_symlink()
        or path.resolve().parent != folder.resolve()
        or path.suffix.lower() not in EXTENSIONS
        or not path.is_file()
    ):
        raise ValueError("Image does not exist or its format is unsupported.")
    return path


def load_image(source, mode="RGBA"):
    """Decode once, normalize EXIF orientation and enforce the image size limit."""
    try:
        with Image.open(source) as image:
            if image.width * image.height > 25_000_000:
                raise ValueError("Images must not exceed 25 megapixels.")
            return ImageOps.exif_transpose(image).convert(mode)
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError("Unable to read the image.") from error


def gallery_groups(root):
    """Return valid, non-symlink label directories in stable order."""
    return [path for path in sorted(Path(root).iterdir())
            if not path.is_symlink() and path.is_dir()
            and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", path.name)]


def gallery_images(folder):
    """Use the same supported reference files for inference and both web views."""
    return [path for path in sorted(folder.iterdir())
            if not path.is_symlink() and path.is_file()
            and path.suffix.lower() in EXTENSIONS]
