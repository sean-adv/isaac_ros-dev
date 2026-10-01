"""GPU-resident tiled patch matching; coordinates are half-open original pixels."""
from contextlib import nullcontext
from dataclasses import dataclass
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def inference_dtype(device):
    """DINOv3 ViT-H+ activations can overflow FP16 even for valid RGB inputs."""
    if torch.device(device).type == 'cuda' and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


@dataclass(frozen=True)
class TileConfig:
    tile_size: int = 448
    overlap: float = .25
    batch_size: int = 4
    merge_mode: str = 'weighted'

    def __post_init__(self):
        if not isinstance(self.tile_size, int) or self.tile_size < 1:
            raise ValueError('tile_size must be a positive integer.')
        if not math.isfinite(self.overlap) or not 0 <= self.overlap < 1:
            raise ValueError('overlap must be in [0, 1).')
        if int(self.tile_size * (1 - self.overlap)) < 1:
            raise ValueError('overlap produces a stride smaller than one pixel.')
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError('batch_size must be a positive integer.')
        if self.merge_mode not in ('max', 'weighted'):
            raise ValueError('merge_mode must be max or weighted.')


def _starts(length, size, stride):
    end = max(0, length - size)
    starts = list(range(0, end + 1, stride))
    if starts[-1] != end:
        starts.append(end)
    return starts


def generate_tiles(image, tile_size=448, overlap=.25):
    """Return CHW tensor views and xyxy coordinates, without resizing the image."""
    TileConfig(tile_size=tile_size, overlap=overlap)
    if image.ndim != 3 or image.shape[0] != 3 or min(image.shape[1:]) < 1:
        raise ValueError('Expected a nonempty RGB CHW tensor.')
    height, width = image.shape[-2:]
    stride = int(tile_size * (1 - overlap))
    coords = [(x, y, min(x + tile_size, width), min(y + tile_size, height))
              for y in _starts(height, tile_size, stride)
              for x in _starts(width, tile_size, stride)]
    return [image[:, y1:y2, x1:x2] for x1, y1, x2, y2 in coords], coords


def preprocess_tiles(tiles, tile_size, processor):
    """Stack one batch of RGB uint8 tiles; pad small images on right/bottom.

    Normalization follows the reference processor. No spatial resize or crop is
    needed: tile_size is already an exact multiple of the model patch size.
    """
    batch = torch.stack(tiles).float()
    h, w = batch.shape[-2:]
    if h > tile_size or w > tile_size:
        raise ValueError('Tile exceeds tile_size.')
    if h != tile_size or w != tile_size:
        batch = F.pad(batch, (0, tile_size - w, 0, tile_size - h), mode='replicate')
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
    if height % patch or width % patch:
        raise ValueError('Input dimensions must be multiples of the model patch size.')
    tokens = outputs.last_hidden_state[:, 1 + model.config.num_register_tokens:]
    gh, gw = height // patch, width // patch
    if tokens.shape[1] != gh * gw:
        raise RuntimeError('Unexpected DINOv3 patch grid.')
    return F.normalize(tokens.float(), dim=-1).reshape(tokens.shape[0], gh, gw, -1)


def run_dino_batch(tiles, model, batch_size, tile_size, processor, dtype=torch.float32):
    """Yield feature batches, bounding activations and feature memory by batch_size."""
    TileConfig(tile_size=tile_size, batch_size=batch_size)
    if tile_size % model.config.patch_size:
        raise ValueError('tile_size must be a multiple of the model patch size.')
    for start in range(0, len(tiles), batch_size):
        # Keep the context inside the generator, including model execution.
        with torch.inference_mode():
            batch = preprocess_tiles(tiles[start:start + batch_size], tile_size, processor)
            amp = (torch.autocast('cuda', dtype=dtype)
                   if batch.is_cuda and dtype in (torch.float16, torch.bfloat16) else nullcontext())
            with amp:
                outputs = model(pixel_values=batch.to(dtype=dtype))
            features = patch_features(outputs, model, tile_size, tile_size)
        yield features


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
def merge_similarity_maps(similarity_maps, tile_coordinates, original_shape,
                          mode='weighted', tile_size=448):
    """Upsample to the padded tile, crop padding, then merge in original pixels."""
    if mode not in ('max', 'weighted'):
        raise ValueError('merge mode must be max or weighted.')
    if similarity_maps.ndim != 3 or len(similarity_maps) != len(tile_coordinates):
        raise ValueError('Expected one patch similarity map per tile coordinate.')
    height, width = original_shape
    scores = similarity_maps.new_full((height, width), -torch.inf if mode == 'max' else 0,
                                      dtype=torch.float32)
    weights = torch.zeros_like(scores) if mode == 'weighted' else None
    # Strictly positive at the border: outer image pixels must not divide by zero.
    axis = torch.hann_window(tile_size, periodic=False, device=scores.device).clamp_min(.01)
    window = axis[:, None] * axis[None, :] if weights is not None else None
    for raw, (x1, y1, x2, y2) in zip(similarity_maps, tile_coordinates):
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
                and x2 - x1 <= tile_size and y2 - y1 <= tile_size):
            raise ValueError('Tile coordinates lie outside the image or tile.')
        tile = F.interpolate(raw[None, None].float(), (tile_size, tile_size),
                             mode='bilinear', align_corners=False)[0, 0, :y2-y1, :x2-x1]
        region = scores[y1:y2, x1:x2]
        if weights is None:
            torch.maximum(region, tile, out=region)
        else:
            weight = window[:y2-y1, :x2-x1]
            region.addcmul_(tile, weight)
            weights[y1:y2, x1:x2].add_(weight)
    if weights is not None:
        scores.div_(weights)
    return scores


def debug_visualizations(bgr, tile_scores, coords, scores, mask, boxes, tile_size):
    """CPU-only optional diagnostics; callers transfer tile scores once."""
    from .boxes import draw_boxes
    grid = bgr.copy()
    for i, (x1, y1, x2, y2) in enumerate(coords):
        cv2.rectangle(grid, (x1, y1), (x2 - 1, y2 - 1), (80, 220, 255), 1)
        cv2.putText(grid, str(i), (x1 + 4, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (80, 220, 255), 1)
    def color(raw):
        return cv2.applyColorMap((np.clip(raw, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    # Fixed thumbnails bound debug memory even for high-resolution inputs.
    thumb = 160
    columns = min(6, len(coords))
    atlas = np.zeros((math.ceil(len(coords) / columns) * (thumb + 24), columns * thumb, 3), np.uint8)
    for i, (raw, coord) in enumerate(zip(tile_scores, coords)):
        x1, y1, x2, y2 = coord
        pixels = cv2.resize(raw, (tile_size, tile_size))[:y2-y1, :x2-x1]
        y, x = (i // columns) * (thumb + 24), (i % columns) * thumb
        atlas[y:y+thumb, x:x+thumb] = cv2.resize(color(pixels), (thumb, thumb))
        cv2.putText(atlas, f'{i}: {x1},{y1}', (x+3, y+thumb+16),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
    final = bgr.copy()
    draw_boxes(final, [(item['bbox'], item['score']) for item in boxes])
    return dict(grid=grid, tiles=atlas, heatmap=color(scores),
                mask=cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR), boxes=final)
