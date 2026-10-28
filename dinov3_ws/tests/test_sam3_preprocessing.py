"""SAM3 luminance correction preserves inputs, geometry and the model contract."""
import numpy as np
from PIL import Image
import pytest
import torch

from dinov3_cases.sam3_preprocessing import prepare_sam3_image, validate_preprocessing
from dinov3_cases.sam3_trt import prepare_sam3_pixels
from test_sam3 import fake_detector


def shadow_image():
    row = np.linspace(30, 90, 128).astype(np.uint8)
    gray = np.tile(row, (96, 1))
    return Image.fromarray(np.repeat(gray[..., None], 3, axis=2))


def test_disabled_and_zero_blend_are_pixel_exact():
    image = shadow_image()
    baseline = prepare_sam3_pixels(image)
    np.testing.assert_array_equal(prepare_sam3_pixels(image, clahe_enabled=False), baseline)
    np.testing.assert_array_equal(prepare_sam3_pixels(image, clahe_enabled=True, clahe_blend=0), baseline)


def test_luminance_correction_changes_shadows_without_mutating_source():
    image = shadow_image()
    before = np.asarray(image).copy()
    corrected = prepare_sam3_image(image, clahe_enabled=True)
    pixels = np.asarray(corrected)
    assert corrected.size == image.size
    assert not np.array_equal(pixels, before)
    assert pixels.mean() > before.mean()
    # Neutral RGB remains neutral, apart from conversion rounding.
    assert np.max(np.ptp(pixels.astype(int), axis=2)) <= 2
    np.testing.assert_array_equal(np.asarray(image), before)
    model_input = prepare_sam3_pixels(image, clahe_enabled=True)
    assert model_input.shape == (1, 3, 1008, 1008)
    assert model_input.dtype == np.float32
    assert -1 <= model_input.min() <= model_input.max() <= 1


def test_detect_uses_corrected_pixels_and_original_coordinates():
    detector = fake_detector()
    detector.preprocessing = validate_preprocessing(clahe_enabled=True)
    image = shadow_image()
    before = np.asarray(image).copy()
    original_forward = detector.predict_tensors
    def forward(pixels, features, mask):
        expected = torch.from_numpy(prepare_sam3_pixels(image, **detector.preprocessing)).half()
        torch.testing.assert_close(pixels, expected)
        return original_forward(pixels, features, mask)
    detector.predict_tensors = forward
    proposals = detector.detect(image, 'box', .2, 1., 20)
    assert proposals[0]['bbox'] == [32, 24, 96, 72]
    np.testing.assert_array_equal(np.asarray(image), before)


@pytest.mark.parametrize('options', [
    dict(clahe_enabled='true'), dict(clahe_clip_limit=0),
    dict(clahe_clip_limit=float('nan')), dict(clahe_clip_limit=True),
    dict(clahe_tile_grid_size=0), dict(clahe_tile_grid_size=8.5),
    dict(clahe_tile_grid_size=True), dict(clahe_tile_grid_size=65),
    dict(clahe_blend=-.1), dict(clahe_blend=1.1), dict(clahe_blend=float('nan')),
])
def test_invalid_settings_rejected(options):
    with pytest.raises(ValueError):
        validate_preprocessing(**options)
