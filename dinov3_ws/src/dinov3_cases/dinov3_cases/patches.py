"""Native-resolution, single-image DINO patch matching."""
from contextlib import nullcontext

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def resize_reference(image, max_side, min_side):
    """Limit the reference long edge without changing its aspect ratio."""
    if max_side < 0:
        raise ValueError('reference_max_side must be nonnegative.')
    result = image.copy()
    if max_side:
        result.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    if min(result.size) < min_side:
        raise ValueError(
            f'Reference size {result.size} is smaller than feature stride {min_side}; '
            'increase reference_max_side or use a less narrow reference image.')
    return result


def prepare_reference(image, max_side, stride):
    """Resize, then replicate bottom/right edges to complete the patch grid."""
    if isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0:
        raise ValueError('Patch stride must be a positive integer.')
    reference = resize_reference(image, max_side, stride)
    right, bottom = (-reference.width) % stride, (-reference.height) % stride
    if not (right or bottom):
        return reference
    pixels = np.asarray(reference)
    padding = ((0, bottom), (0, right)) + ((0, 0),) * (pixels.ndim - 2)
    return Image.fromarray(np.pad(pixels, padding, mode='edge'))


def validate_target_padding(padding, mode, stride=None):
    """Padding must preserve the original feature-grid origin."""
    if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
        raise ValueError('target_padding must be a nonnegative integer.')
    if mode not in ('reflect', 'replicate'):
        raise ValueError('target_padding_mode must be reflect or replicate.')
    if stride is not None and padding % stride:
        raise ValueError(f'target_padding must be a multiple of feature_stride={stride}.')


def pad_target_image(image, padding, mode, stride):
    """Add context without resizing or shifting the original patch alignment."""
    validate_target_padding(padding, mode, stride)
    if min(image.size) < stride:
        raise ValueError(f'Image width and height must each be at least feature_stride={stride}.')
    if not padding:
        return image
    pixels = np.asarray(image.convert('RGB'))
    padded = np.pad(pixels, ((padding, padding), (padding, padding), (0, 0)),
                    mode='reflect' if mode == 'reflect' else 'edge')
    return Image.fromarray(padded)


def inference_dtype(device):
    """Prefer BF16 on CUDA; DINOv3 activations can overflow FP16."""
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


def feature_stride(model):
    """Return the input-pixel stride of the model's final spatial features."""
    config = model.config
    if getattr(config, 'model_type', None) == 'dinov3_convnext':
        return 32
    if hasattr(config, 'patch_size'):
        return config.patch_size
    raise ValueError(f'Unsupported DINOv3 architecture: {type(config).__name__}.')


def prefix_tokens(model):
    """Return the number of non-spatial tokens preceding the feature grid."""
    return 1 + getattr(model.config, 'num_register_tokens', 0)


def patch_features(outputs, model, height, width):
    """Discard global/register tokens and retain [B, H_grid, W_grid, C]."""
    stride = feature_stride(model)
    tokens = outputs.last_hidden_state[:, prefix_tokens(model):]
    gh, gw = height // stride, width // stride
    if tokens.shape[1] != gh * gw:
        raise RuntimeError('Unexpected DINOv3 patch grid.')
    return F.normalize(tokens.float(), dim=-1).reshape(tokens.shape[0], gh, gw, -1)


@torch.inference_mode()
def run_dino_image(image, model, processor, dtype=torch.float32):
    """Run one original image and return its final spatial feature grid."""
    stride = feature_stride(model)
    if min(image.shape[-2:]) < stride:
        raise ValueError(f'Image width and height must each be at least feature_stride={stride}.')
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


def reference_grid_image(image, stride):
    """Draw model-input patch boundaries without rescaling."""
    if not isinstance(stride, int) or isinstance(stride, bool) or stride <= 0:
        raise ValueError('Patch stride must be a positive integer.')
    if min(image.size) < stride:
        raise ValueError('Reference is smaller than one patch.')
    grid = image.convert('RGB').copy()
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
