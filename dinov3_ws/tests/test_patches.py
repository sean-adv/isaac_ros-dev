from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from dinov3_cases.patches import (inference_dtype, patch_features, preprocess_image,
    run_dino_image, compute_similarity_maps, similarity_pixels, debug_visualizations,
    feature_stride, prefix_tokens)
from dinov3_cases.boxes import similarity_to_bboxes


def test_inference_precision_avoids_fp16_overflow(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: True)
    assert inference_dtype('cuda') == torch.bfloat16
    assert inference_dtype('cpu') == torch.float32
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: False)
    assert inference_dtype('cuda') == torch.float32


class SpatialProbe(torch.nn.Module):
    """Use DINO's actual patch embedding code with known spatial responses.

    Deliberately omit attention: this checks geometry, independently of learned
    semantic matching. Each output channel is the mean of one input channel.
    """
    def __init__(self):
        super().__init__()
        from transformers import DINOv3ViTConfig
        from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTEmbeddings
        self.config = DINOv3ViTConfig(hidden_size=3, num_attention_heads=1,
                                     patch_size=16, num_register_tokens=4)
        self.embeddings = DINOv3ViTEmbeddings(self.config)
        with torch.no_grad():
            conv = self.embeddings.patch_embeddings
            conv.weight.zero_()
            conv.bias.zero_()
            for channel in range(3):
                conv.weight[channel, channel].fill_(1 / 256)
            self.embeddings.register_tokens.fill_(999)

    def forward(self, pixel_values):
        return SimpleNamespace(last_hidden_state=self.embeddings(pixel_values))


def test_actual_dino_patch_order_rectangular_grid():
    model = SpatialProbe()
    # Non-square grid and distinct x/y signals expose transposes and token offsets.
    image = torch.ones((1, 3, 48, 80))
    image[:, 0] = torch.arange(80)[None, :] + 1
    image[:, 1] = torch.arange(48)[:, None] + 1
    with torch.inference_mode():
        actual = patch_features(model(image), model, 48, 80)[0]
    expected = torch.empty((3, 5, 3))
    for row in range(3):
        for column in range(5):
            expected[row, column] = torch.tensor([column * 16 + 8.5, row * 16 + 8.5, 1.])
    torch.testing.assert_close(actual, F.normalize(expected, dim=-1))


def test_convnext_large_geometry_and_global_token():
    from transformers import DINOv3ConvNextConfig, DINOv3ConvNextModel
    model = DINOv3ConvNextModel(DINOv3ConvNextConfig(
        depths=[1, 1, 1, 1], hidden_sizes=[8, 16, 24, 32])).eval()
    image = torch.randn(3, 65, 97)
    processor = SimpleNamespace(do_rescale=False, do_normalize=False)
    actual = run_dino_image(image, model, processor)
    assert feature_stride(model) == 32 and prefix_tokens(model) == 1
    assert actual.shape == (1, 2, 3, 32)
    torch.testing.assert_close(torch.linalg.vector_norm(actual, dim=-1),
                               torch.ones((1, 2, 3)))



@pytest.mark.parametrize('shape', [(48, 80), (53, 87), (16, 16)])
def test_original_pixels_batch_one_and_patch_geometry(shape):
    processor = SimpleNamespace(do_rescale=False, do_normalize=False)
    model = SpatialProbe()
    image = torch.ones((3, *shape))
    image[0] = torch.arange(shape[1])[None, :] + 1
    image[1] = torch.arange(shape[0])[:, None] + 1
    original = image.clone()
    calls = []
    def inspect_input(module, args, kwargs):
        pixels = kwargs['pixel_values']
        calls.append(tuple(pixels.shape))
        torch.testing.assert_close(pixels, image[None])
    model.register_forward_pre_hook(inspect_input, with_kwargs=True)
    actual = run_dino_image(image, model, processor)
    assert calls == [(1, 3, *shape)]
    expected = F.normalize(F.avg_pool2d(image[None], 16).permute(0, 2, 3, 1), dim=-1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(image, original)
    assert actual.shape == (1, shape[0] // 16, shape[1] // 16, 3)


@pytest.mark.parametrize('shape', [(1, 32), (32, 15)])
def test_too_small_image_is_not_padded(shape):
    with pytest.raises(ValueError, match='at least feature_stride'):
        run_dino_image(torch.zeros((3, *shape)), SpatialProbe(), None)


def test_preprocessing_matches_processor_without_resize():
    from transformers import DINOv3ViTImageProcessorFast
    processor = DINOv3ViTImageProcessorFast()
    image = torch.randint(0, 256, (3, 53, 87), dtype=torch.uint8)
    actual = preprocess_image(image, processor)
    expected = processor(images=image, do_resize=False, do_center_crop=False,
                         return_tensors='pt').pixel_values
    assert actual.shape == (1, 3, 53, 87)
    torch.testing.assert_close(actual, expected)


def test_real_model_matches_direct_original_image_forward():
    from transformers import DINOv3ViTConfig, DINOv3ViTModel, DINOv3ViTImageProcessorFast
    model = DINOv3ViTModel(DINOv3ViTConfig(hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_register_tokens=4)).eval()
    processor = DINOv3ViTImageProcessorFast()
    image = torch.randint(0, 256, (3, 53, 87), dtype=torch.uint8)
    actual = run_dino_image(image, model, processor)
    with torch.inference_mode():
        pixels = processor(images=image, do_resize=False, do_center_crop=False,
                           return_tensors='pt').pixel_values
        tokens = model(pixel_values=pixels).last_hidden_state[:, 5:]
        expected = F.normalize(tokens.float(), dim=-1).reshape(1, 3, 5, 32)
    torch.testing.assert_close(actual, expected)
    bank = F.normalize(torch.randn(2051, 32), dim=-1)
    torch.testing.assert_close(compute_similarity_maps(actual, bank), (actual @ bank.T).amax(-1))


def test_pixel_coordinates_and_uncovered_edges():
    patches = torch.tensor([[.1, .4, .9], [.2, .5, .8]])
    actual = similarity_pixels(patches, (37, 55), 16)
    expected = F.interpolate(patches[None, None], (32, 48), mode='bilinear', align_corners=False)[0, 0]
    torch.testing.assert_close(actual[:32, :48], expected)
    assert (actual[32:] == -1).all() and (actual[:, 48:] == -1).all()
    constant = similarity_pixels(torch.full((2, 3), -.7), (32, 48), 16)
    torch.testing.assert_close(constant, torch.full((32, 48), -.7))
    boxes, mask = similarity_to_bboxes(actual.numpy(), 0., 1, 0, 1, return_mask=True)
    assert boxes[0]['bbox'] == [0, 0, 48, 32]
    views = debug_visualizations(np.zeros((37, 55, 3), np.uint8), patches.numpy(),
                                 actual.numpy(), mask, boxes, 16)
    assert set(views) == {'grid', 'patches', 'heatmap', 'mask', 'boxes'}
    assert all(v.dtype == np.uint8 and v.ndim == 3 for v in views.values())


def test_pixel_postprocessing_empty_holes_noise_and_evidence():
    raw = np.zeros((40, 60), np.float32)
    assert similarity_to_bboxes(raw) == []
    raw[5:25, 8:40] = .9
    raw[10:20, 15:30] = 0
    raw[35, 50] = 1
    boxes = similarity_to_bboxes(raw, .7, 20, 0, 3)
    assert len(boxes) == 1
    assert boxes[0]['bbox'] == [8, 5, 40, 25]
    assert boxes[0]['area'] == 640
    assert boxes[0]['score'] == pytest.approx(.9)
