from dataclasses import replace
import numpy as np
import pytest
from dinov3_cases.boxes import (BoxConfig, clean_mask, extract_boxes, refine_box,
                                merge_boxes_by_iou, draw_boxes)


def boxes(scores, **kwargs):
    cfg = replace(BoxConfig(), rgb_border_fraction=0, **kwargs)
    n = scores.shape[0]
    return extract_boxes(scores, np.ones_like(scores, dtype=bool), (0, 0, n*16, n*16, n),
                         (n*16, n*16), 16, .7, cfg)


def test_closing_holes_and_noise():
    s = np.zeros((20, 20), np.float32)
    s[3:10, 3:10] = .9
    s[5:8, 5:8] = 0
    s[3:10, 6] = 0
    s[16, 16] = .9
    mask = clean_mask(s, np.ones_like(s, bool), .7, 3)
    assert mask[3:10, 3:10].all()
    result = boxes(s)
    assert len(result) == 1
    assert result[0][0] == (40, 40, 168, 168)
    assert result[0][1] == pytest.approx(.9)


def test_hole_fill_without_closing():
    s = np.zeros((12, 12))
    s[2:10, 2:10] = 1
    s[4:8, 4:8] = 0
    m = clean_mask(s, np.ones_like(s, bool), .7, 1)
    assert m[2:10, 2:10].all()
    assert m.sum() == 64


def test_separate_components_empty_full_and_clipping():
    s = np.zeros((16, 16))
    assert boxes(s) == []
    s[1:4, 1:4] = .8
    s[10:14, 10:14] = .9
    assert len(boxes(s)) == 2
    s[:] = 1
    assert boxes(s)[0][0] == (0, 0, 256, 256)


def test_letterbox_and_padding_exclusion():
    s = np.ones((14, 14), np.float32)
    valid = np.zeros_like(s, bool)
    valid[3:10] = True
    cfg = BoxConfig(rgb_border_fraction=0)
    assert extract_boxes(s, valid, (0, 56, 224, 112, 14), (192, 96), 16, .7, cfg)[0][0] == (0, 0, 192, 96)
    s[:] = 0
    s[5:7, 4:6] = .9
    result = extract_boxes(s, valid, (0, 56, 224, 112, 14), (192, 96), 16, .7, cfg)
    assert result[0][0] == (48, 13, 90, 55)
    s[~valid] = 1
    assert extract_boxes(s, valid, (0, 56, 224, 112, 14), (192, 96), 16, .7, cfg) == result


def test_rgb_disabled_flat_weak_short_and_interior_edges():
    box = (30, 30, 70, 70)
    cfg = BoxConfig()
    image = np.zeros((100, 100, 3), np.uint8)
    assert refine_box(image, box, cfg) == box
    image[28:72, 28:72] = 5
    assert refine_box(image, box, cfg) == box
    image[:] = 0
    image[40:45, 27:29] = 255
    assert refine_box(image, box, cfg) == box
    image[:] = 0
    image[40:60, 40:60] = 255
    assert refine_box(image, box, cfg) == box
    image[28:72, 28:72] = 255
    assert refine_box(image, box, replace(cfg, rgb_border_fraction=0)) == box


def test_rgb_all_four_outer_edges_and_search_limits():
    image = np.zeros((100, 100, 3), np.uint8)
    image[28:72, 28:72] = 255
    result = refine_box(image, (30, 30, 70, 70), BoxConfig())
    assert 26 <= result[0] < 30 and 26 <= result[1] < 30
    assert 70 < result[2] <= 74 and 70 < result[3] <= 74
    image[:] = 0
    image[10:90, 10:90] = 255
    assert refine_box(image, (30, 30, 70, 70), BoxConfig()) == (30, 30, 70, 70)
    assert refine_box(image, (0, 0, 100, 100), BoxConfig()) == (0, 0, 100, 100)


def test_iou_merges_transitive_overlaps_and_keeps_best_score():
    original = [
        dict(bbox=[0, 0, 10, 10], score=.7, area=30),
        dict(bbox=[4, 0, 14, 10], score=.9, area=40),
        dict(bbox=[8, 0, 18, 10], score=.8, area=20),
        dict(bbox=[30, 0, 40, 10], score=.6, area=50),
    ]
    merged = merge_boxes_by_iou(original, .4)
    assert merged == [dict(bbox=[0, 0, 18, 10], score=.9, area=90), original[3]]
    assert original[0]['bbox'] == [0, 0, 10, 10]
    assert merge_boxes_by_iou(original[:2], .5) == original[:2]
    assert merge_boxes_by_iou([], .4) == []


def test_highest_score_box_has_red_outline():
    image = np.zeros((100, 120, 3), np.uint8)
    draw_boxes(image, [((10, 30, 30, 60), .6), ((70, 30, 90, 60), .9)])
    assert image[40, 10].tolist() == [60, 220, 80]
    assert image[40, 70].tolist() == [0, 0, 255]
    draw_boxes(np.zeros_like(image), [])


@pytest.mark.parametrize('params', [dict(closing_kernel=2), dict(min_component_patches=0),
    dict(padding_patches=float('nan')), dict(rgb_border_fraction=-.1),
    dict(merge_iou_threshold=0), dict(merge_iou_threshold=float('nan')),
    dict(edge_min_gradient=0), dict(edge_min_support=1.1)])
def test_invalid_config(params):
    with pytest.raises(ValueError):
        BoxConfig(**params)
