"""SAM3 adapter scoring, coordinates and head-bucket integration."""
import threading
import numpy as np
import pytest
from PIL import Image
from dinov3_cases.sam3_trt import Sam3Detector, parse_prompts


@pytest.mark.parametrize('prompt', ['', 'a,b,c,d,e', None])
def test_invalid_prompt(prompt):
    with pytest.raises(ValueError):
        parse_prompts(prompt)


def test_presence_score_and_original_coordinates():
    detector = object.__new__(Sam3Detector)
    detector.lock = threading.RLock()
    detector.tokenizer = type('Tokenizer', (), {'encode': lambda self, text: [1, 2]})()
    detector._text_features = lambda labels: (4, np.zeros((32, 4, 256)), np.zeros((4, 32)))
    def vision(feeds, want):
        x = feeds['images']
        assert x.shape == (1, 3, 1008, 1008)
        assert np.all(x == -1)
        return {'fpn_2': np.zeros((1, 256, 1, 1))}
    detector.vision = vision
    detector.positions = {4: np.zeros(1)}
    def head(feeds, want):
        assert feeds['img_feat'].shape[0] == 4
        return {'scores': np.zeros((4, 1, 1)), 'presence': np.zeros((4, 1)),
                'boxes': np.tile([[[0.5, 0.5, 0.5, 0.5]]], (4, 1, 1))}
    detector.heads = {4: head}
    image = Image.new('RGB', (200, 100))
    result = detector.detect(image, 'bottle,box.', 0.2, 1.0, 20)
    assert len(result) == 2  # Padded classes are excluded; NMS disabled.
    assert result[0]['bbox'] == [50, 25, 150, 75]
    assert result[0]['detection_score'] == 0.25  # sigmoid(score) * sigmoid(presence)
    assert detector.detect(image, 'bottle,box.', 0.3, 1.0, 20) == []
