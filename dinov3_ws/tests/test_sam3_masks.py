"""Full-frame mask pixels, empty results, filtering and revision-safe HTTP views."""
import base64
from io import BytesIO
import threading
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from dinov3_cases.sam3_visualization import encode_mask_snapshot
from dinov3_cases.web import create_app
from test_sam3 import fake_detector


def decode(snapshot, view):
    return Image.open(BytesIO(base64.b64decode(snapshot[view])))


def test_mask_matches_inference_frame_and_threshold():
    image = Image.new('RGB', (80, 40), (20, 40, 60))
    masks = torch.full((2, 1, 40, 80), -10.)
    masks[1, 0, 10:30, 20:60] = 10
    snapshot = encode_mask_snapshot(image, masks, [(1, 0)], 'box')
    binary = np.asarray(decode(snapshot, 'mask'))
    assert binary.shape == (40, 80)
    assert np.all(binary[10:30, 20:60] == 255)
    assert np.count_nonzero(binary) == 800
    assert decode(snapshot, 'overlay').size == image.size
    assert image.getpixel((30, 20)) == (20, 40, 60)
    assert snapshot['object_count'] == 1


def test_empty_inference_replaces_mask_with_black_preview():
    snapshot = encode_mask_snapshot(Image.new('RGB', (1920, 1080)),
                                    torch.ones(1, 1, 2, 2), [], 'box')
    assert decode(snapshot, 'mask').size == (960, 540)
    assert not np.asarray(decode(snapshot, 'mask')).any()
    assert snapshot['object_count'] == 0


def test_masks_follow_score_and_size_filters():
    detector = fake_detector()
    forward = detector.predict_tensors
    detector.predict_tensors = lambda *args: (*forward(*args), torch.ones(1, 1, 4, 4))
    image = Image.new('RGB', (80, 40))
    detector.detect(image, 'box', .2, 1., 20)
    assert detector.last_visualization['object_count'] == 1
    detector.detect(image, 'box', .3, 1., 20)
    assert detector.last_visualization['object_count'] == 0
    assert not np.asarray(decode(detector.last_visualization, 'mask')).any()
    detector.detect(image, 'box', .2, 1., 20, min_width=50)
    assert detector.last_visualization['object_count'] == 0


def test_http_snapshot_revision_and_reset(tmp_path):
    node = SimpleNamespace(root=tmp_path, lock=threading.Lock(), generation=2,
                           sam3_mask_revision=3, sam3_mask_snapshot=None)
    client = create_app(node).test_client()
    assert client.get('/api/sam3_masks').json['ready'] is False
    node.sam3_mask_snapshot = encode_mask_snapshot(Image.new('RGB', (80, 40)),
                                                  torch.ones(1, 1, 2, 2), [(0, 0)], 'box')
    metadata = client.get('/api/sam3_masks').json
    assert metadata['revision'] == '2-3'
    assert metadata['metadata']['original_size'] == [80, 40]
    assert 'overlay' not in metadata['metadata']
    response = client.get('/debug/sam3/2-3/mask')
    assert response.mimetype == 'image/png'
    assert np.asarray(Image.open(BytesIO(response.data))).all()
    assert client.get('/debug/sam3/2-3/overlay').mimetype == 'image/jpeg'
    assert client.get('/debug/sam3/2-2/mask').status_code == 410
    node.generation += 1
    node.sam3_mask_snapshot = None
    assert not client.get('/api/sam3_masks').json['ready']
    assert client.get('/debug/sam3/2-3/mask').status_code == 410
