"""Offline prompt validity, integrity and OmDet class-scoring semantics."""
import hashlib
import numpy as np
import pytest
import torch
from PIL import Image
from dinov3_cases.omdet_trt import OfflinePromptBank, prepare_image, postprocess


@pytest.fixture
def bank_files(tmp_path):
    path = tmp_path / 'prompts.npz'
    values = {'0_label_feats': np.zeros((2,1,512),np.float32),
              '0_task_feats': np.zeros((77,1,512),np.float32),
              '0_task_mask': np.concatenate([np.ones((1,5),np.int32),np.zeros((1,72),np.int32)],axis=1)}
    np.savez(path, **values)
    metadata = dict(class_count=2, profiles=[dict(prompt='box. bottle.', labels=['box','bottle'],task='Detect box, bottle.')],
                    prompt_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return path, metadata, values


def test_offline_bank_preserves_order_and_rejects_unencoded_prompt(bank_files):
    path, metadata, _ = bank_files
    bank = OfflinePromptBank(path, metadata)
    assert bank.resolve(' box.  bottle. ')[0]['labels'] == ['box','bottle']
    with pytest.raises(ValueError, match='offline prompts'):
        bank.resolve('bottle. box.')
    with pytest.raises(ValueError, match='offline prompts'):
        bank.resolve('car.')


def test_offline_bank_detects_wrong_feature_file(bank_files):
    path, metadata, _ = bank_files
    path.write_bytes(b'wrong')
    with pytest.raises(ValueError, match='checksum'):
        OfflinePromptBank(path, metadata)


@pytest.mark.parametrize('name,value', [('0_label_feats',np.zeros((3,1,512),np.float32)),
                                      ('0_task_mask',np.full((1,77),2,np.int32)),
                                      ('0_task_feats',np.full((77,1,512),np.nan,np.float32))])
def test_offline_bank_rejects_invalid_embeddings(bank_files,name,value):
    path, metadata, values = bank_files
    values[name] = value
    np.savez(path, **values)
    metadata['prompt_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='Invalid'):
        OfflinePromptBank(path,metadata)


def test_rgb_normalization_matches_official_pixel_scale():
    x = prepare_image(Image.new('RGB',(20,10),(255,128,0)), dict(input_size=640,mean=[123.675,116.28,103.53],std=[58.395,57.12,57.375]))
    assert x.shape == (1,3,640,640)
    np.testing.assert_allclose(x[0,:,0,0], (np.array([255,128,0])-np.array([123.675,116.28,103.53]))/np.array([58.395,57.12,57.375]),rtol=1e-6)


def test_sigmoid_class_scores_synonym_nms_and_empty_result():
    boxes = torch.tensor([[[[.5,.5,.4,.4],[.5,.5,.4,.4],[.1,.1,.1,.1]]]])
    logits = torch.tensor([[[[4.,3.],[3.,4.],[-10.,-10.]]]])
    results = postprocess(boxes,logits,['bottle','medicine bottle'],(200,100),.3,.5,20)
    assert len(results) == 1
    assert results[0]['detection_label'] in ['bottle','medicine bottle']
    np.testing.assert_allclose(results[0]['detection_score'],torch.sigmoid(torch.tensor(4.)).item())
    assert not postprocess(boxes,logits,['bottle','medicine bottle'],(200,100),1.,.5,20)
