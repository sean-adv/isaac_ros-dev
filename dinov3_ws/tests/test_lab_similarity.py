"""DINO thresholds gate references; Lab selects the winner without weighted scores."""
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


def test_dino_threshold_and_lab_winner_drive_debug_and_cached_results(tmp_path):
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
    assert result['score'] == pytest.approx(.9)
    assert result['margin'] == pytest.approx(1)
    index, score = gallery.best_reference(descriptor, lab_histogram=histogram)
    assert gallery.labels[index] == 'red' and score == pytest.approx(.9)
    assert gallery.classify(descriptor, .95, .05, lab_histogram=histogram)['label'] == 'blue'
    assert gallery.classify(descriptor, .9, 2, lab_histogram=histogram)['label'] == 'red'
    cache = TrackedObjectRecognizer()
    token = object()
    proposal = dict(bbox=[0, 0, 32, 32], detection_label='box', detection_score=.9)
    pixels = np.asarray(Image.new('RGB', (32, 32), 'red'))
    args = (pixels, [proposal], SameDescriptorEncoder(), gallery, 32, 16, .9, .05)
    first, _, _ = cache.update(*args, token=token, gallery_revision=1)
    second, _, changed = cache.update(*args, token=token, gallery_revision=1)
    assert second == first and not changed and not cache.encoded
    third, diagnostics, changed = cache.update(*args, token=token, gallery_revision=2)
    assert third == first and changed and not cache.encoded
    previous_context = diagnostics[0]["context"]
    fourth, diagnostics, changed = cache.update(
        *args[:6], .95, .05, token=token, gallery_revision=2
    )
    assert fourth[0]['label'] == 'blue' and changed and not cache.encoded
    context = diagnostics[0]['context']
    assert context['recognition_threshold'] == .95
    assert previous_context['recognition_threshold'] == .9
    index, _ = gallery.best_reference(context['descriptor'],
                                      lab_histogram=context['lab_histogram'],
                                      threshold=context['recognition_threshold'])
    assert gallery.labels[index] == fourth[0]['label']


def make_gallery(scores, histograms, labels=None):
    gallery = ObjectGallery()
    gallery.labels = labels or list('abcdef')[:len(scores)]
    gallery.paths = [f'{i}.png' for i in range(len(scores))]
    gallery.features = torch.tensor([[s] for s in scores])
    gallery.lab_histograms = np.array(histograms, dtype=np.float32)
    return gallery


def test_candidates_are_not_limited_to_top_three_classes():
    gallery = make_gallery([.99, .98, .97, .96, .95, .94, .93],
                           [[0., 1.]] * 4 + [[.81, .19], [.81, .19], [1., 0.]],
                           ['a', 'a', 'a', 'b', 'c', 'c', 'd'])
    descriptor = torch.tensor([1.])
    histogram = np.array([1., 0.], dtype=np.float32)
    result = gallery.classify(descriptor, .9, 2, lab_histogram=histogram)
    assert result['label'] == 'd'
    assert result['score'] == pytest.approx(.93)
    assert result['lab_score'] == pytest.approx(1)
    assert result['margin'] == pytest.approx(.1)
    index, score = gallery.best_reference(descriptor, lab_histogram=histogram, threshold=.9)
    assert index == 6 and score == pytest.approx(.93)
    # Raising the DINO threshold excludes d, even though its Lab score is perfect.
    result = gallery.classify(descriptor, .94, 0, lab_histogram=histogram)
    assert result['label'] == 'c'
    assert result['reference']['name'] == '4.png'
    index, _ = gallery.best_reference(descriptor, lab_histogram=histogram, threshold=.94)
    assert index == 4
    # Explicit diagnostic classes can be compared regardless of the recognition threshold.
    index, score = gallery.best_reference(descriptor, 'd', lab_histogram=histogram, threshold=1)
    assert index == 6 and score == pytest.approx(.93)


def test_threshold_applies_to_each_reference_before_lab_class_selection():
    gallery = make_gallery([.99, .5, .9], [[0., 1.], [1., 0.], [.81, .19]], ['a', 'a', 'b'])
    result = gallery.classify(torch.tensor([1.]), .8, 0, lab_histogram=np.array([1., 0.]))
    assert result['label'] == 'b'
    assert result['lab_score'] == pytest.approx(.9)
    assert result['score'] == pytest.approx(.9)


@pytest.mark.parametrize('scores', [[-.1], [-.1, -.2], [-.1, -.2, -.3]])
def test_no_dino_candidates_returns_unknown_even_with_perfect_lab(scores):
    gallery = make_gallery(scores, [[1.]] * len(scores))
    result = gallery.classify(torch.tensor([1.]), 0, 0, lab_histogram=np.ones(1))
    assert result['label'] == 'unknown'
    assert result['reason'] == 'low_similarity'
    assert result['candidate'] == 'a'
    assert result['score'] == pytest.approx(-.1)
    assert result['lab_score'] == pytest.approx(1)
    assert result['margin'] is None


def test_single_candidate_needs_no_lab_threshold_and_ties_are_deterministic():
    gallery = make_gallery([.75, .7], [[0., 1.], [1., 0.]], ['b', 'a'])
    histogram = np.array([1., 0.])
    result = gallery.classify(torch.tensor([1.]), .75, 2, lab_histogram=histogram)
    assert result['label'] == 'b'  # Equality passes, even with zero Lab similarity.
    assert result['lab_score'] == 0
    assert result['margin'] is None
    gallery.lab_histograms[:] = [1., 0.]
    assert gallery.classify(torch.tensor([1.]), .7, 2, lab_histogram=histogram)['label'] == 'b'
    gallery.features[:] = .75
    result = gallery.classify(torch.tensor([1.]), .7, 2, lab_histogram=histogram)
    assert result['label'] == 'a' and result['margin'] == 0
