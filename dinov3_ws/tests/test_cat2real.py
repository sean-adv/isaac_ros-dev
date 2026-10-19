"""Cat2Real must use one pooler forward and map centered padding correctly."""
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from dinov3_cases.cat2real import MODEL_ID, prepare_cat2real_image
from dinov3_cases.grounded import encode_object
from dinov3_cases.object_visualization import object_debug_views


def test_center_white_padding_and_exact_content_geometry():
    result = prepare_cat2real_image(Image.new('RGB',(16,32),'red'),64)
    assert result.size == (64,64)
    assert result.info['content_box'] == (16,0,48,64)
    assert result.getpixel((0,32)) == (255,255,255)
    assert result.getpixel((32,32)) == (255,0,0)
    assert result.getpixel((63,32)) == (255,255,255)


@pytest.mark.parametrize('size',[(1,1000),(1000,1),(77,151),(151,77)])
def test_thin_crops_keep_nonempty_content_box(size):
    prepared = prepare_cat2real_image(Image.new('RGB',size),384)
    x0,y0,x1,y1 = prepared.info['content_box']
    assert 0 <= x0 < x1 <= 384
    assert 0 <= y0 < y1 <= 384


def test_heatmap_crops_centered_padding_from_both_sides():
    roi = Image.new('RGB',(16,32),'red')
    prepared = prepare_cat2real_image(roi,64)
    patches = torch.tensor([[[0.,1.],[1.,0.],[1.,0.],[0.,1.]]]*4).reshape(16,2)
    _,metadata = object_debug_views(roi,prepared,patches,torch.tensor([[1.,0.]]),64,16)
    assert metadata['content_box'] == [16,0,48,64]
    assert metadata['content_size'] == [32,64]
    assert metadata['similarity_min'] > .5
    assert metadata['similarity_max'] == pytest.approx(1.)


def test_encode_object_uses_explicit_embedding_instead_of_patch_mean():
    class Encoder:
        def prepare_image(self,image,max_side,stride):return prepare_cat2real_image(image,max_side)
        def encode_with_descriptor(self,image):
            assert image.info['content_box'] == (16,0,48,64)
            return torch.tensor([[1.,0.],[1.,0.]]),torch.tensor([0.,2.])
    _,patches,descriptor = encode_object(Encoder(),Image.new('RGB',(16,32)),64,16)
    assert patches.shape == (2,2)
    torch.testing.assert_close(descriptor,torch.tensor([0.,1.]))


def test_cat2real_encoder_keeps_patches_and_pooler_from_one_forward(monkeypatch):
    import dinov3_cases.encoder as module
    monkeypatch.setattr(module,'MODEL_ID',MODEL_ID)
    class Model:
        config=SimpleNamespace(patch_size=16,num_register_tokens=4)
        calls=0
        def __call__(self,pixel_values):
            self.calls+=1
            assert tuple(pixel_values.shape) == (1,3,384,384)
            return SimpleNamespace(last_hidden_state=torch.ones(1,581,4),
                                   pooler_output=torch.tensor([[0.,2.,0.,0.]]))
    encoder=module.Encoder.__new__(module.Encoder)
    encoder.device='cpu';encoder.model=Model()
    encoder.processor=SimpleNamespace(do_rescale=True,rescale_factor=1/255,
        do_normalize=True,image_mean=[.485,.456,.406],image_std=[.229,.224,.225])
    patches,descriptor=encoder.encode_with_descriptor(Image.new('RGB',(384,384),'white'))
    assert encoder.model.calls == 1
    assert patches.shape == (576,4)
    torch.testing.assert_close(patches.norm(dim=-1),torch.ones(576))
    torch.testing.assert_close(descriptor,torch.tensor([0.,2.,0.,0.]))
