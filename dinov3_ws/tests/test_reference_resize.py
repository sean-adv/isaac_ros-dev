import pytest
from PIL import Image

from dinov3_cases.patches import resize_reference


@pytest.mark.parametrize('size,max_side,expected', [
    ((800, 600), 480, (480, 360)),
    ((600, 900), 480, (320, 480)),
    ((300, 200), 480, (300, 200)),
    ((800, 600), 0, (800, 600)),
])
def test_reference_resize_preserves_source_and_aspect(size, max_side, expected):
    source = Image.new('RGB', size, (10, 20, 30))
    result = resize_reference(source, max_side, 32)
    assert result.size == expected
    assert source.size == size
    assert result.getpixel((0, 0)) == (10, 20, 30)
    assert result is not source


@pytest.mark.parametrize('size,max_side', [
    ((800, 600), -1), ((800, 600), 16), ((2000, 100), 480), ((20, 100), 0),
])
def test_invalid_reference_size_rejected(size, max_side):
    with pytest.raises(ValueError):
        resize_reference(Image.new('RGB', size), max_side, 32)
