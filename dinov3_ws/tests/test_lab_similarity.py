"""Color fusion must affect reference selection, rejection, and cached results."""
import numpy as np
import pytest
import torch
from PIL import Image

from dinov3_cases.grounded import (
    ObjectGallery, TrackedObjectRecognizer, compute_lab_histogram,
)


class SameDescriptorEncoder:
    def encode(self, image):
        return torch.tensor([[1., 0.]])


def test_lab_histogram_normalization_transparency_and_size():
    red = compute_lab_histogram(Image.new('RGB', (1, 1), 'red'))
    rgba = Image.new('RGBA', (40, 20), (0, 0, 255, 0))
    rgba.paste((255, 0, 0, 255), (0, 0, 10, 20))
    np.testing.assert_allclose(compute_lab_histogram(rgba), red)
    assert red.shape == (2048,)
    assert red.sum() == pytest.approx(1)
    blue = compute_lab_histogram(Image.new('RGB', (20, 30), 'blue'))
    assert np.sqrt(red * blue).sum() == 0
    with pytest.raises(ValueError, match='transparent'):
        compute_lab_histogram(Image.new('RGBA', (1, 1)))


def test_fusion_changes_winner_and_drives_threshold_margin_and_debug(tmp_path):
    for label in ('blue', 'red'):
        (tmp_path / label).mkdir()
        Image.new('RGB', (32, 32), label).save(tmp_path / label / 'ref.png')
    gallery = ObjectGallery()
    gallery.refresh(tmp_path, SameDescriptorEncoder(), 32, 16)
    # DINO slightly prefers blue; color evidence must make red the winner.
    gallery.features[gallery.labels.index('red')] = torch.tensor([.9, 0.])
    histogram = compute_lab_histogram(Image.new('RGB', (32, 32), 'red'))
    descriptor = torch.tensor([1., 0.])
    result = gallery.classify(descriptor, .9, .05, lab_histogram=histogram)
    assert result['label'] == 'red'
    assert result['dino_score'] == pytest.approx(.9)
    assert result['lab_score'] == pytest.approx(1)
    assert result['score'] == pytest.approx(.92)
    assert result['margin'] == pytest.approx(.12)
    index, score = gallery.best_reference(descriptor, lab_histogram=histogram)
    assert gallery.labels[index] == 'red' and score == pytest.approx(.92)
    assert gallery.classify(descriptor, .95, .05, lab_histogram=histogram)['reason'] == 'low_similarity'
    assert gallery.classify(descriptor, .9, .15, lab_histogram=histogram)['reason'] == 'ambiguous'
    cache = TrackedObjectRecognizer()
    token = object()
    proposal = dict(bbox=[0, 0, 32, 32], detection_label='box', detection_score=.9)
    pixels = np.asarray(Image.new('RGB', (32, 32), 'red'))
    args = (pixels, [proposal], SameDescriptorEncoder(), gallery, 32, 16, .9, .05)
    first, _, _ = cache.update(*args, token=token, gallery_revision=1)
    second, _, changed = cache.update(*args, token=token, gallery_revision=1)
    assert second == first and not changed and not cache.encoded
    third, _, changed = cache.update(*args, token=token, gallery_revision=2)
    assert third == first and changed and not cache.encoded


def test_only_top_three_distinct_dino_classes_can_win():
    gallery = ObjectGallery()
    # Multiple strong references for a must not crowd c out of the shortlist.
    gallery.labels = ['a', 'a', 'a', 'b', 'c', 'c', 'd']
    gallery.paths = [f'{i}.png' for i in range(7)]
    gallery.features = torch.tensor([[s] for s in [.99, .98, .97, .96, .95, .94, .93]])
    gallery.lab_histograms = np.array(
        [[0., 1.]] * 4 + [[1., 0.], [1., 0.], [1., 0.]], dtype=np.float32
    )
    descriptor = torch.tensor([1.])
    histogram = np.array([1., 0.], dtype=np.float32)
    # The excluded fourth class would beat every eligible class with this Lab score.
    gallery.lab_histograms[4:6] = [0.81, 0.19]
    result = gallery.classify(descriptor, 0, 0, lab_histogram=histogram)
    assert result['label'] == 'c'
    assert result['score'] == pytest.approx(.8 * .95 + .2 * .9)
    assert result['margin'] == pytest.approx(result['score'] - .8 * .99)
    scores, _, lab = gallery.similarities(descriptor, histogram)
    assert np.isneginf(scores[6]) and np.isnan(lab[6])
    assert np.isfinite(scores[:6]).all()
    index, score = gallery.best_reference(descriptor, lab_histogram=histogram)
    assert index == 4 and score == pytest.approx(result['score'])
    # An explicitly selected diagnostic class can still be compared outside the shortlist.
    index, score = gallery.best_reference(descriptor, 'd', lab_histogram=histogram)
    assert index == 6 and score == pytest.approx(.8 * .93 + .2)


@pytest.mark.parametrize('count', [1, 2, 3])
def test_shortlist_with_fewer_classes_and_negative_dino_scores(count):
    gallery = ObjectGallery()
    gallery.labels = list('abc')[:count]
    gallery.paths = [f'{label}.png' for label in gallery.labels]
    gallery.features = torch.tensor([[-.1], [-.2], [-.3]])[:count]
    gallery.lab_histograms = np.ones((count, 1), dtype=np.float32)
    result = gallery.classify(torch.tensor([1.]), 0, 0, lab_histogram=np.ones(1))
    assert result['label'] == 'a'
    assert result['score'] == pytest.approx(.12)
    assert result['margin'] == (None if count == 1 else pytest.approx(.08))
