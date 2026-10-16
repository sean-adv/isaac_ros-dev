"""Qwen pooling and global-descriptor integration without loading 8B weights."""

import torch
import pytest
from PIL import Image

from dinov3_cases.qwen_encoder import pool_last_token
from dinov3_cases.grounded import (
    ObjectGallery, ObjectVisualizationSnapshot, recognize_objects,
)


def test_pooling_uses_last_attended_token_for_both_padding_sides():
    hidden = torch.tensor([[[9., 9.], [3., 4.], [9., 9.]],
                           [[9., 9.], [9., 9.], [4., 3.]]])
    mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
    torch.testing.assert_close(pool_last_token(hidden, mask), torch.tensor([[.6, .8], [.8, .6]]))


@pytest.mark.parametrize('value', [float('nan'), float('inf'), 0.])
def test_invalid_global_vectors_are_rejected(value):
    with pytest.raises(ValueError, match='Invalid Qwen'):
        pool_last_token(torch.full((1, 2, 3), value), torch.ones(1, 2))


def test_global_gallery_recognition_and_selected_reference_debug(tmp_path):
    class GlobalEncoder:
        supports_spatial_features = False
        def encode(self, image):
            return torch.tensor([[.6, .8]])

    folder = tmp_path / 'product'
    folder.mkdir()
    Image.new('RGB', (80, 40), 'red').save(folder / 'one.png')
    Image.new('RGB', (90, 50), 'blue').save(folder / 'two.png')
    encoder = GlobalEncoder()
    gallery = ObjectGallery()
    gallery.refresh(tmp_path, encoder, 448, 32)
    diagnostics = []
    detections = recognize_objects(
        Image.new('RGB', (100, 60)),
        [dict(bbox=[0, 0, 100, 60], detection_score=.9, detection_label='box')],
        encoder, gallery, 448, 32, .85, .05, diagnostics, defer_diagnostics=True,
    )
    assert detections[0]['label'] == 'product'
    snapshot = ObjectVisualizationSnapshot(detections, diagnostics, gallery, 448, 32)
    for selected in ('', 'product'):
        items, images = snapshot.render(selected)
        assert items[0]['embedding_type'] == 'global'
        assert not items[0]['heatmap_available']
        assert set(items[0]['views']) == {'crop', 'grid', 'reference'}
        assert (0, 'heatmap') not in images
