"""Native-resolution, single-image DINO patch matching."""
from contextlib import nullcontext

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def inference_dtype(device):
    """DINOv3 ViT-H+ activations can overflow FP16 even for valid RGB inputs."""
    if torch.device(device).type == 'cuda' and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def preprocess_image(image, processor):
    """Normalize RGB CHW pixels without changing their spatial dimensions."""
    if image.ndim != 3 or image.shape[0] != 3 or min(image.shape[1:]) < 1:
        raise ValueError('Expected a nonempty RGB CHW tensor.')
    batch = image.unsqueeze(0).to(dtype=torch.float32, copy=True)
    if processor.do_rescale:
        batch.mul_(processor.rescale_factor)
    if processor.do_normalize:
        mean = batch.new_tensor(processor.image_mean).view(1, 3, 1, 1)
        std = batch.new_tensor(processor.image_std).view(1, 3, 1, 1)
        batch.sub_(mean).div_(std)
    return batch


def patch_features(outputs, model, height, width):
    """Discard CLS/register tokens and retain [B, H_patch, W_patch, C]."""
    patch = model.config.patch_size
    tokens = outputs.last_hidden_state[:, 1 + model.config.num_register_tokens:]
    gh, gw = height // patch, width // patch
    if tokens.shape[1] != gh * gw:
        raise RuntimeError('Unexpected DINOv3 patch grid.')
    return F.normalize(tokens.float(), dim=-1).reshape(tokens.shape[0], gh, gw, -1)


@torch.inference_mode()
def run_dino_image(image, model, processor, dtype=torch.float32):
    """One original image per forward; the model creates nonoverlapping patches."""
    patch = model.config.patch_size
    if min(image.shape[-2:]) < patch:
        raise ValueError(f'Image width and height must each be at least patch_size={patch}.')
    batch = preprocess_image(image, processor)
    amp = (torch.autocast('cuda', dtype=dtype)
           if batch.is_cuda and dtype in (torch.float16, torch.bfloat16) else nullcontext())
    with amp:
        outputs = model(pixel_values=batch.to(dtype=dtype))
    features = patch_features(outputs, model, *image.shape[-2:])
    if not torch.isfinite(features).all():
        raise RuntimeError(f'Non-finite DINOv3 features with {dtype}; image rejected.')
    return features


@torch.inference_mode()
def compute_similarity_maps(features, reference_embedding):
    """Best cosine match against the existing normalized reference patch bank."""
    if reference_embedding.ndim == 1:
        reference_embedding = reference_embedding.unsqueeze(0)
    if reference_embedding.ndim != 2 or not len(reference_embedding):
        raise ValueError('Reference bank must be a nonempty [R, C] tensor.')
    flat = features.reshape(-1, features.shape[-1])
    score = flat.new_full((len(flat),), -1., dtype=torch.float32)
    # Bound both axes, even with large batches or reference groups.
    for start in range(0, len(flat), 2048):
        target = flat[start:start + 2048]
        best = score[start:start + len(target)]
        for chunk in reference_embedding.split(2048):
            torch.maximum(best, (target @ chunk.T).amax(dim=1), out=best)
    return score.reshape(features.shape[:-1])


@torch.inference_mode()
def similarity_pixels(patch_scores, image_shape, patch_size):
    """Map complete patches to original pixels; uncovered edge strips score -1."""
    height, width = image_shape
    gh, gw = height // patch_size, width // patch_size
    if patch_scores.shape != (gh, gw) or not gh or not gw:
        raise ValueError('Unexpected similarity patch grid for image dimensions.')
    covered_h, covered_w = gh * patch_size, gw * patch_size
    pixels = F.interpolate(patch_scores[None, None].float(), (covered_h, covered_w),
                           mode='bilinear', align_corners=False)[0, 0]
    scores = pixels.new_full((height, width), -1.)
    scores[:covered_h, :covered_w] = pixels
    return scores


def debug_visualizations(bgr, patch_scores, scores, mask, boxes, patch_size):
    """Show the native patch grid and its scores in original image coordinates."""
    from .boxes import draw_boxes
    grid = bgr.copy()
    gh, gw = patch_scores.shape
    for y in range(0, gh * patch_size + 1, patch_size):
        cv2.line(grid, (0, y), (gw * patch_size - 1, y), (80, 220, 255), 1)
    for x in range(0, gw * patch_size + 1, patch_size):
        cv2.line(grid, (x, 0), (x, gh * patch_size - 1), (80, 220, 255), 1)
    def color(raw):
        return cv2.applyColorMap((np.clip(raw, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    final = bgr.copy()
    draw_boxes(final, [(item['bbox'], item['score']) for item in boxes])
    patches = cv2.resize(color(patch_scores), (gw * patch_size, gh * patch_size),
                         interpolation=cv2.INTER_NEAREST)
    return dict(grid=grid, patches=patches, heatmap=color(scores),
                mask=cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR), boxes=final)
