import numpy as np
import pytest
from PIL import Image

from dinov3_cases.patches import prepare_reference, resize_reference


@pytest.mark.parametrize('stride', [16, 32])
@pytest.mark.parametrize('mode', ['RGB', 'RGBA'])
@pytest.mark.parametrize('extra', [(0, 0), (5, 0), (0, 3), (5, 3)])
def test_padding_preserves_source_and_replicates_only_bottom_right(stride, mode, extra):
    width, height = stride * 2 + extra[0], stride + extra[1]
    channels = len(mode)
    pixels = np.random.default_rng(5).integers(0, 256, (height, width, channels), dtype=np.uint8)
    image = Image.fromarray(pixels, mode=mode)
    prepared = prepare_reference(image, 0, stride)
    result = np.asarray(prepared)
    expected_width = (width + stride - 1) // stride * stride
    expected_height = (height + stride - 1) // stride * stride
    assert prepared.size == (expected_width, expected_height)
    np.testing.assert_array_equal(result[:height, :width], pixels)
    expected = pixels[np.minimum(np.arange(expected_height), height - 1)[:, None],
                      np.minimum(np.arange(expected_width), width - 1)[None, :]]
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(np.asarray(image), pixels)


def test_padding_follows_resize_and_includes_last_patch():
    image = Image.new('RGB', (101, 65), 'white')
    resized = resize_reference(image, 64, 16)
    prepared = prepare_reference(image, 64, 16)
    assert resized.size == (64, 41)
    assert prepared.size == (64, 48)
    np.testing.assert_array_equal(np.asarray(prepared)[:41], resized)
