from io import BytesIO

import numpy as np
import pytest
import torch
from PIL import Image

from dinov3_cases.patches import (load_image, reference_patch_mask, resize_reference,
                                 compute_similarity_maps)


@pytest.mark.parametrize('stride', [16, 32])
def test_mask_matches_feature_order_and_ignores_incomplete_edges(stride):
    rgba = np.full((2 * stride + 3, 3 * stride + 5, 4), 255, np.uint8)
    rgba[:stride, stride:2 * stride, 3] = 0  # Background.
    rgba[:stride, 2 * stride:3 * stride, 3] = 128  # Mixed boundary.
    rgba[stride:2 * stride, :stride, 3] = 243  # Above 95%.
    rgba[stride:2 * stride, stride:2 * stride, 3] = 242  # Below 95%.
    rgba[2 * stride:, :, 3] = 0
    rgba[:, 3 * stride:, 3] = 0
    mask = reference_patch_mask(Image.fromarray(rgba), stride)
    assert mask.tolist() == [True, False, False, True, False, True]


def test_alpha_filter_removes_background_best_match():
    image = Image.new('RGBA', (32, 16), (0, 0, 0, 0))
    image.paste((255, 0, 0, 255), (0, 0, 16, 16))
    features = torch.tensor([[1., 0.], [0., 1.]])
    target = torch.tensor([[[[0., 1.]]]])
    mask = torch.as_tensor(reference_patch_mask(image, 16))
    assert compute_similarity_maps(target, features).item() == 1.
    assert compute_similarity_maps(target, features[mask]).item() == 0.


def test_load_and_resize_keep_alpha_and_clear_hidden_rgb():
    source = Image.new('RGBA', (64, 32), (100, 50, 20, 0))
    source.paste((10, 20, 30, 255), (0, 0, 32, 32))
    stream = BytesIO()
    source.save(stream, format='PNG')
    stream.seek(0)
    loaded = load_image(stream, preserve_alpha=True)
    assert loaded.mode == 'RGBA'
    assert loaded.getpixel((63, 0)) == (0, 0, 0, 0)
    resized = resize_reference(loaded, 32, 16)
    assert resized.size == (32, 16)
    assert reference_patch_mask(resized, 16).tolist() == [True, False]
    stream.seek(0)
    assert load_image(stream).mode == 'RGB'


def test_palette_transparency_survives_loading():
    source = Image.new('P', (32, 16), 0)
    source.putpalette([255, 0, 0, 0, 0, 0] + [0] * 762)
    source.paste(1, (16, 0, 32, 16))
    stream = BytesIO()
    source.save(stream, format='PNG', transparency=1)
    stream.seek(0)
    assert reference_patch_mask(load_image(stream, True), 16).tolist() == [True, False]


@pytest.mark.parametrize('mode', ['RGB', 'RGBA'])
def test_opaque_images_keep_all_patches(mode):
    assert reference_patch_mask(Image.new(mode, (35, 19), 'white'), 16).tolist() == [True, True]


def test_fully_transparent_image_has_no_candidates():
    assert not reference_patch_mask(Image.new('RGBA', (32, 32)), 16).any()


def test_margin_excludes_image_border_and_neighbors_of_alpha_hole():
    rgba = np.full((7 * 16, 7 * 16, 4), 255, np.uint8)
    rgba[3 * 16:4 * 16, 3 * 16:4 * 16, 3] = 0
    keep = reference_patch_mask(Image.fromarray(rgba), 16, .99, 1).reshape(7, 7)
    expected = np.zeros((7, 7), bool)
    expected[1:6, 1:6] = True
    expected[2:5, 2:5] = False
    np.testing.assert_array_equal(keep, expected)


def test_margin_can_remove_all_candidates_in_narrow_reference():
    assert not reference_patch_mask(Image.new('RGBA', (32, 80), 'white'), 16, .99, 1).any()


@pytest.mark.parametrize('margin', [-1, 0.5, float('nan')])
def test_invalid_margin_rejected(margin):
    with pytest.raises(ValueError):
        reference_patch_mask(Image.new('RGBA', (64, 64)), 16, margin_patches=margin)


@pytest.mark.parametrize('threshold', [0., -0.1, 1.1, float('nan')])
def test_invalid_coverage_rejected(threshold):
    with pytest.raises(ValueError):
        reference_patch_mask(Image.new('RGBA', (16, 16)), 16, threshold)
