"""Official SAM3 bbox coordinates, preprocessing, cache and prompt contract."""
import threading
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
import torch
from transformers import BatchEncoding

from dinov3_cases.sam3_trt import Sam3Detector, parse_prompts, prepare_sam3_pixels


@pytest.mark.parametrize('prompt', ['', 'a,b,c,d,e', None])
def test_invalid_prompt(prompt):
    with pytest.raises(ValueError):parse_prompts(prompt)


def test_prompts_preserve_actual_class_count():
    assert parse_prompts(' bottle, box. ') == ['bottle','box']
    assert len(parse_prompts('a,b,c,d')) == 4


@pytest.mark.parametrize('color,value',[('black',-1),('white',1)])
def test_sam3_uses_minus_one_to_one_rgb_nchw(color,value):
    pixels=prepare_sam3_pixels(Image.new('RGB',(7,11),color))
    assert pixels.shape == (1,3,1008,1008)
    assert pixels.dtype == np.float32
    assert np.all(pixels==value)


def fake_detector():
    detector=Sam3Detector.__new__(Sam3Detector)
    detector.lock=threading.RLock();detector.device=torch.device('cpu')
    detector._text_features=lambda labels:(torch.zeros(len(labels),32,256),torch.ones(len(labels),32,dtype=torch.bool))
    def forward(pixels,features,mask):
        assert pixels.dtype==torch.float16 and tuple(pixels.shape)==(1,3,1008,1008)
        count=len(features)
        return torch.tensor([[[.25,.25,.75,.75]]]*count),torch.zeros(count,1),torch.zeros(count,1)
    detector.predict_tensors=forward
    return detector


def test_presence_score_and_original_coordinates():
    detector=fake_detector();image=Image.new('RGB',(200,100))
    result=detector.detect(image,'bottle,box.',.2,1.,20)
    # Identical boxes across prompts are deduplicated before classification.
    assert len(result)==1
    assert [r['detection_label'] for r in result]==['bottle']
    assert result[0]['bbox']==[50,25,150,75]
    assert result[0]['detection_score']==.25
    assert len(detector.detect(image,'bottle,box.',.2,1.,20,
                               containment_threshold=0.0))==2
    assert detector.detect(image,'bottle,box.',.3,1.,20)==[]
    assert detector.detect(image,'box.',.2,1.,20,min_width=101)==[]


def test_nonfinite_outputs_are_rejected():
    detector=fake_detector()
    detector.predict_tensors=lambda *a:(torch.full((1,1,4),float('nan')),torch.zeros(1,1),torch.zeros(1,1))
    with pytest.raises(RuntimeError,match='Non-finite'):
        detector.detect(Image.new('RGB',(20,20)),'box',.2,1.,20)


def test_prompt_cache_and_zero_padding_match_sam3_text_encoder():
    detector=Sam3Detector.__new__(Sam3Detector);detector.device='cpu';detector.cached_prompt=None
    calls=[]
    def tokenize(labels,**options):
        count=len(labels)
        return BatchEncoding(dict(input_ids=torch.tensor([[49406,123,49407,49407]]*count),
                                  attention_mask=torch.tensor([[1,1,1,0]]*count)))
    def encode(input_ids,attention_mask,return_dict):
        assert input_ids[:,3].eq(0).all()
        assert input_ids[:,2].eq(49407).all()
        calls.append(len(input_ids))
        return SimpleNamespace(last_hidden_state=torch.ones(len(input_ids),4,256))
    detector.tokenizer=tokenize;detector.text_encoder=encode;detector.text_projection=torch.nn.Identity()
    first=detector._text_features(['person'])
    assert detector._text_features(['person']) is first
    assert detector._text_features(['person','car'])[0].shape[0]==2
    assert calls==[1,2]
