"""DINOv3 patch normalization, matching and reference-grid rendering."""

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def preprocess_image(image, processor):
    """Normalize RGB CHW pixels without changing their spatial dimensions."""
    if image.ndim != 3 or image.shape[0] != 3 or min(image.shape[1:]) < 1:
        raise ValueError("Expected a nonempty RGB CHW tensor.")
    batch = image.unsqueeze(0).to(dtype=torch.float32, copy=True)
    if processor.do_rescale:
        batch.mul_(processor.rescale_factor)
    if processor.do_normalize:
        mean = batch.new_tensor(processor.image_mean).view(1, 3, 1, 1)
        std = batch.new_tensor(processor.image_std).view(1, 3, 1, 1)
        batch.sub_(mean).div_(std)
    return batch


def feature_stride(model):
    """Return the input-pixel stride of the model's final spatial features."""
    config = model.config
    if hasattr(config, "patch_size"):
        return config.patch_size
    raise ValueError(f"Unsupported DINOv3 architecture: {type(config).__name__}.")


def prefix_tokens(model):
    """Return the number of non-spatial tokens preceding the feature grid."""
    return 1 + getattr(model.config, "num_register_tokens", 0)


def patch_features(outputs, model, height, width):
    """Discard global/register tokens and retain [B, H_grid, W_grid, C]."""
    stride = feature_stride(model)
    tokens = outputs.last_hidden_state[:, prefix_tokens(model) :]
    gh, gw = height // stride, width // stride
    if tokens.shape[1] != gh * gw:
        raise RuntimeError("Unexpected DINOv3 patch grid.")
    return F.normalize(tokens.float(), dim=-1).reshape(tokens.shape[0], gh, gw, -1)


@torch.inference_mode()
def run_dino_image(image, model, processor):
    """Run one original image and return its final spatial feature grid."""
    stride = feature_stride(model)
    if min(image.shape[-2:]) < stride:
        raise ValueError(f"Image width and height must each be at least feature_stride={stride}.")
    batch = preprocess_image(image, processor)
    outputs = model(pixel_values=batch)
    features = patch_features(outputs, model, *image.shape[-2:])
    if not torch.isfinite(features).all():
        raise RuntimeError("Non-finite DINOv3 features; image rejected.")
    return features


@torch.inference_mode()
def compute_similarity_maps(features, reference_embedding):
    """Best cosine match against the existing normalized reference patch bank."""
    if reference_embedding.ndim == 1:
        reference_embedding = reference_embedding.unsqueeze(0)
    if reference_embedding.ndim != 2 or not len(reference_embedding):
        raise ValueError("Reference bank must be a nonempty [R, C] tensor.")
    flat = features.reshape(-1, features.shape[-1])
    score = flat.new_full((len(flat),), -1.0, dtype=torch.float32)
    # Bound both axes, even with large batches or reference groups.
    for start in range(0, len(flat), 2048):
        target = flat[start : start + 2048]
        best = score[start : start + len(target)]
        for chunk in reference_embedding.split(2048):
            torch.maximum(best, (target @ chunk.T).amax(dim=1), out=best)
    return score.reshape(features.shape[:-1])


def reference_grid_image(image, stride):
    """Draw model-input patch boundaries without rescaling."""
    if not isinstance(stride, int) or isinstance(stride, bool) or stride <= 0:
        raise ValueError("Patch stride must be a positive integer.")
    if min(image.size) < stride:
        raise ValueError("Reference is smaller than one patch.")
    grid = image.convert("RGB").copy()
    width, height = grid.size
    covered_w, covered_h = width // stride * stride, height // stride * stride
    draw = ImageDraw.Draw(grid)
    for x in range(0, covered_w, stride):
        draw.line((x, 0, x, covered_h - 1), fill=(255, 220, 80))
    for y in range(0, covered_h, stride):
        draw.line((0, y, covered_w - 1, y), fill=(255, 220, 80))
    draw.line((covered_w - 1, 0, covered_w - 1, covered_h - 1), fill=(255, 220, 80))
    draw.line((0, covered_h - 1, covered_w - 1, covered_h - 1), fill=(255, 220, 80))
    return grid
