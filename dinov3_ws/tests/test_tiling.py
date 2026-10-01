from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from dinov3_cases.boxes import similarity_to_bboxes
from dinov3_cases.tiling import (TileConfig, generate_tiles, preprocess_tiles,
    run_dino_batch, compute_similarity_maps, merge_similarity_maps, debug_visualizations,
    patch_features, inference_dtype)


def test_inference_precision_avoids_fp16_overflow(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: True)
    assert inference_dtype('cuda') == torch.bfloat16
    assert inference_dtype('cpu') == torch.float32
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: False)
    assert inference_dtype('cuda') == torch.float32


class SpatialProbe(torch.nn.Module):
    """Use DINO's actual patch embedding code with known spatial responses.

    Deliberately omit attention: this checks geometry, independently of learned
    semantic matching. Each output channel is the mean of one input channel.
    """
    def __init__(self):
        super().__init__()
        from transformers import DINOv3ViTConfig
        from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTEmbeddings
        self.config = DINOv3ViTConfig(hidden_size=3, num_attention_heads=1,
                                     patch_size=16, num_register_tokens=4)
        self.embeddings = DINOv3ViTEmbeddings(self.config)
        with torch.no_grad():
            conv = self.embeddings.patch_embeddings
            conv.weight.zero_()
            conv.bias.zero_()
            for channel in range(3):
                conv.weight[channel, channel].fill_(1 / 256)
            self.embeddings.register_tokens.fill_(999)

    def forward(self, pixel_values):
        return SimpleNamespace(last_hidden_state=self.embeddings(pixel_values))


def test_actual_dino_patch_order_rectangular_grid():
    model = SpatialProbe()
    # Non-square grid and distinct x/y signals expose transposes and token offsets.
    image = torch.ones((1, 3, 48, 80))
    image[:, 0] = torch.arange(80)[None, :] + 1
    image[:, 1] = torch.arange(48)[:, None] + 1
    with torch.inference_mode():
        actual = patch_features(model(image), model, 48, 80)[0]
    expected = torch.empty((3, 5, 3))
    for row in range(3):
        for column in range(5):
            expected[row, column] = torch.tensor([column * 16 + 8.5, row * 16 + 8.5, 1.])
    torch.testing.assert_close(actual, F.normalize(expected, dim=-1))


@pytest.mark.parametrize('mode', ['max', 'weighted'])
@pytest.mark.parametrize('shape', [(1080, 1920), (97, 701), (701, 97)])
def test_spatial_probe_from_rgb_crop_through_final_boxes(shape, mode):
    height, width = shape
    image = torch.zeros((3, height, width), dtype=torch.uint8)
    image[2] = 1
    # Asymmetric, separated markers, including the bottom/right boundary.
    expected_boxes = [(16, 16, 64, 64), (width-64, height-64, width, height)]
    for x1, y1, x2, y2 in expected_boxes:
        image[0, y1:y2, x1:x2] = 1
    tiles, coords = generate_tiles(image, 448, .25)
    for tile, (x1, y1, x2, y2) in zip(tiles, coords):
        torch.testing.assert_close(tile, image[:, y1:y2, x1:x2])
    processor = SimpleNamespace(do_rescale=False, do_normalize=False)
    bank = torch.tensor([[1., 0., 0.]])
    maps = torch.cat([compute_similarity_maps(batch, bank)
                      for batch in run_dino_batch(tiles, SpatialProbe(), 4, 448, processor)])
    scores = merge_similarity_maps(maps, coords, shape, mode, 448).numpy()
    boxes = similarity_to_bboxes(scores, .5, 100, 0, 1)
    assert len(boxes) == len(expected_boxes)
    for actual, expected in zip(boxes, expected_boxes):
        # Patch averaging and bilinear interpolation blur an edge by at most a
        # patch; a tile offset, x/y swap or padding resize is much larger.
        assert np.max(np.abs(np.asarray(actual['bbox']) - expected)) <= 16
        x1, y1, x2, y2 = expected
        assert scores[(y1+y2)//2, (x1+x2)//2] > .65
    assert scores[height//2, width//2] == pytest.approx(0)


@pytest.mark.parametrize('shape', [(1080, 1920), (448, 448), (97, 700), (1, 1), (700, 97)])
@pytest.mark.parametrize('overlap', [0., .25, .5])
def test_full_coverage_and_views(shape, overlap):
    image = torch.zeros((3, *shape), dtype=torch.uint8)
    tiles, coords = generate_tiles(image, 448, overlap)
    covered = np.zeros(shape, bool)
    assert len(coords) == len(set(coords))
    for tile, (x1, y1, x2, y2) in zip(tiles, coords):
        covered[y1:y2, x1:x2] = True
        assert tile.shape == (3, y2-y1, x2-x1)
        assert tile.untyped_storage().data_ptr() == image.untyped_storage().data_ptr()
    assert covered.all()
    if shape == (1080, 1920) and overlap == .25:
        assert len(tiles) == 18
        assert coords[-1] == (1472, 632, 1920, 1080)


@pytest.mark.parametrize('params', [dict(tile_size=0), dict(overlap=1), dict(overlap=-.1),
    dict(overlap=float('nan')), dict(overlap=.9999), dict(batch_size=0), dict(merge_mode='bad')])
def test_invalid_settings(params):
    with pytest.raises(ValueError):
        TileConfig(**params)


PROCESSOR = SimpleNamespace(do_rescale=True, rescale_factor=1/255, do_normalize=True,
                            image_mean=[.5]*3, image_std=[.25]*3)


class FakeModel:
    config = SimpleNamespace(patch_size=16, num_register_tokens=2)

    def __init__(self):
        self.batches = []

    def __call__(self, pixel_values):
        assert not torch.is_grad_enabled()
        self.batches.append(len(pixel_values))
        patches = F.avg_pool2d(pixel_values, 16).flatten(2).transpose(1, 2)
        # Poison prefix tokens: including CLS/registers would break spatial assertions.
        prefix = patches.new_full((len(patches), 3, 3), 10000)
        return SimpleNamespace(last_hidden_state=torch.cat([prefix, patches], dim=1))


def test_batching_patch_layout_and_similarity():
    tiles = [torch.full((3, 32, 32), i, dtype=torch.uint8) for i in range(11)]
    model = FakeModel()
    features = torch.cat(list(run_dino_batch(tiles, model, 4, 32, PROCESSOR)))
    assert model.batches == [4, 4, 3]
    assert features.shape == (11, 2, 2, 3)
    bank = F.normalize(torch.randn(2051, 3), dim=-1)
    scores = compute_similarity_maps(features, bank)
    expected = (features @ bank.T).amax(-1)
    torch.testing.assert_close(scores, expected)
    # A real spatial ramp detects flipped or transposed patch axes.
    image = torch.zeros((3, 32, 32), dtype=torch.uint8)
    image[0, :16] = 255
    image[1, :, 16:] = 255
    model = FakeModel()
    actual = next(run_dino_batch([image], model, 4, 32, PROCESSOR))[0]
    expected = F.normalize(F.avg_pool2d(preprocess_tiles([image], 32, PROCESSOR), 16)[0].permute(1, 2, 0), dim=-1)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('count,batches', [(12, [4, 4, 4]), (1, [1])])
def test_full_and_single_batch(count, batches):
    model = FakeModel()
    tiles = [torch.zeros((3, 32, 32), dtype=torch.uint8)] * count
    list(run_dino_batch(tiles, model, 4, 32, PROCESSOR))
    assert model.batches == batches


def test_real_dino_batch_matches_individual_forward():
    from transformers import DINOv3ViTConfig, DINOv3ViTModel
    torch.manual_seed(42)
    model = DINOv3ViTModel(DINOv3ViTConfig(hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_register_tokens=4)).eval()
    tiles = list(torch.randint(0, 256, (5, 3, 32, 32), dtype=torch.uint8))
    batched = torch.cat(list(run_dino_batch(tiles, model, 4, 32, PROCESSOR)))
    individual = torch.cat(list(run_dino_batch(tiles, model, 1, 32, PROCESSOR)))
    torch.testing.assert_close(batched, individual, atol=1e-5, rtol=1e-5)


def test_preprocessing_matches_reference_processor():
    from transformers import DINOv3ViTImageProcessorFast
    processor = DINOv3ViTImageProcessorFast()
    tiles = torch.randint(0, 256, (3, 3, 32, 32), dtype=torch.uint8)
    actual = preprocess_tiles(list(tiles), 32, processor)
    expected = processor(images=tiles, do_resize=False, do_center_crop=False,
                         return_tensors='pt').pixel_values
    torch.testing.assert_close(actual, expected)
    small = torch.randint(0, 256, (3, 7, 13), dtype=torch.uint8)
    padded = preprocess_tiles([small], 32, processor)
    torch.testing.assert_close(padded[:, :, 7:, 13:], padded[:, :, 6:7, 12:13].expand(1, 3, 25, 19))


@pytest.mark.parametrize('mode', ['max', 'weighted'])
def test_constant_negative_maps_and_small_image_crop(mode):
    _, coords = generate_tiles(torch.zeros((3, 37, 81)), 32, .25)
    scores = merge_similarity_maps(torch.full((len(coords), 2, 2), -.7), coords,
                                   (37, 81), mode, 32)
    torch.testing.assert_close(scores, torch.full((37, 81), -.7))
    raw = torch.tensor([[[0., 1.], [0., 1.]]])
    small = merge_similarity_maps(raw, [(0, 0, 13, 7)], (7, 13), mode, 32)
    full = F.interpolate(raw[:, None], (32, 32), mode='bilinear', align_corners=False)[0, 0]
    torch.testing.assert_close(small, full[:7, :13])


def test_overlap_max_and_weighted_center_preference():
    maps = torch.tensor([[[.2]], [[.8]]])
    coords = [(0, 0, 32, 32), (16, 0, 48, 32)]
    maximum = merge_similarity_maps(maps, coords, (32, 48), 'max', 32)
    assert maximum[16, 20].item() == pytest.approx(.8)
    weighted = merge_similarity_maps(maps, coords, (32, 48), 'weighted', 32)
    assert .2 < weighted[16, 18] < .4
    assert .6 < weighted[16, 29] < .8
    torch.testing.assert_close(weighted[:, :16], torch.full((32, 16), .2))
    torch.testing.assert_close(weighted[:, 32:], torch.full((32, 16), .8))


@pytest.mark.parametrize('mode', ['max', 'weighted'])
def test_cross_tile_object_one_global_box_and_boundary_object(mode):
    image = torch.zeros((3, 48, 80))
    _, coords = generate_tiles(image, 32, .5)
    truth = torch.zeros((48, 80))
    truth[10:30, 18:62] = .9  # Crosses multiple tile seams.
    truth[40:, 72:] = .8  # Touches both the bottom and right image boundary.
    maps = torch.stack([truth[y1:y2, x1:x2] for x1, y1, x2, y2 in coords])
    merged = merge_similarity_maps(maps, coords, (48, 80), mode, 32).numpy()
    boxes, mask = similarity_to_bboxes(merged, .7, 32, 2, 1, return_mask=True)
    assert len(boxes) == 2
    assert boxes[0]['bbox'] == [16, 8, 64, 32]
    assert boxes[0]['area'] == 880
    assert boxes[0]['score'] == pytest.approx(.9)
    assert boxes[1]['bbox'] == [70, 38, 80, 48]
    views = debug_visualizations(np.zeros((48, 80, 3), np.uint8), maps.numpy(),
                                 coords, merged, mask, boxes, 32)
    assert set(views) == {'grid', 'tiles', 'heatmap', 'mask', 'boxes'}
    assert all(v.dtype == np.uint8 and v.ndim == 3 for v in views.values())


def test_pixel_postprocessing_empty_holes_noise_and_evidence():
    raw = np.zeros((40, 60), np.float32)
    assert similarity_to_bboxes(raw) == []
    raw[5:25, 8:40] = .9
    raw[10:20, 15:30] = 0
    raw[35, 50] = 1
    boxes = similarity_to_bboxes(raw, .7, 20, 0, 3)
    assert len(boxes) == 1
    assert boxes[0]['bbox'] == [8, 5, 40, 25]
    assert boxes[0]['area'] == 640
    assert boxes[0]['score'] == pytest.approx(.9)
