"""RoPE ablation through real multi-layer DINOv3 attention and both Encoder paths."""
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import DINOv3ViTConfig, DINOv3ViTModel, DINOv3ViTImageProcessorFast

from dinov3_cases.node import Encoder
from dinov3_cases.patches import disable_rope


def small_model():
    torch.manual_seed(42)
    return DINOv3ViTModel(DINOv3ViTConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_register_tokens=4)).eval()


def test_identity_rotations_reach_all_layers_and_both_image_paths():
    model = small_model()
    processor = DINOv3ViTImageProcessorFast()
    with patch('dinov3_cases.node.AutoModel.from_pretrained', return_value=model), \
         patch('dinov3_cases.node.AutoImageProcessor.from_pretrained', return_value=processor), \
         patch('torch.cuda.is_available', return_value=False):
        encoder = Encoder('local-test', rope_enabled=False, target_padding=0)
    calls = []
    def inspect_rotation(module, args, kwargs):
        cos, sin = kwargs['position_embeddings']
        assert torch.equal(cos, torch.ones_like(cos))
        assert torch.equal(sin, torch.zeros_like(sin))
        calls.append(module)
    handles = [layer.register_forward_pre_hook(inspect_rotation, with_kwargs=True)
               for layer in model.layer]
    try:
        rng = np.random.default_rng(7)
        reference = Image.fromarray(rng.integers(0, 256, (48, 80, 3), dtype=np.uint8))
        # A shift by whole patches permutes spatial tokens. With identity RoPE,
        # the corresponding features and cosine scores must follow that shift.
        target = Image.fromarray(np.roll(np.asarray(reference), 16, axis=1))
        bank = encoder.encode(reference)
        features = encoder.spatial_features(target)[0]
        expected = torch.roll(bank.reshape(3, 5, -1), 1, dims=1)
        torch.testing.assert_close(features, expected, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(bank.norm(dim=-1), torch.ones(15))
        _, grid, scores = encoder.similarity_map(target, bank, debug=True)
        assert grid == (3, 5)
        np.testing.assert_allclose(scores, 1., atol=2e-6)
        assert len(calls) == 3 * model.config.num_hidden_layers
        assert 'disabled (reference + target)' in encoder.description
        # Calling the helper twice must not install duplicate hooks.
        count = len(model.rope_embeddings._forward_hooks)
        disable_rope(model)
        assert len(model.rope_embeddings._forward_hooks) == count
    finally:
        for handle in handles:
            handle.remove()


def test_normal_rope_remains_available_and_changes_features():
    model = small_model()
    pixels = torch.randn(1, 3, 48, 80)
    with torch.inference_mode():
        cos, sin = model.rope_embeddings(pixels)
        assert not torch.equal(cos, torch.ones_like(cos))
        assert torch.count_nonzero(sin)
        normal = model(pixels).last_hidden_state
        disable_rope(model)
        disabled = model(pixels).last_hidden_state
    assert not torch.allclose(normal[:, 5:], disabled[:, 5:], atol=1e-6)


def test_trt_cannot_silently_reuse_rope_engine():
    with patch('dinov3_cases.trt_backend.TRTBackbone') as backbone:
        with pytest.raises(ValueError, match='requires inference_backend:=torch'):
            Encoder('local-test', backend='trt', rope_enabled=False)
        backbone.assert_not_called()


def test_unsupported_model_cannot_silently_skip_ablation():
    with pytest.raises(ValueError, match='ViT model with RoPE'):
        disable_rope(torch.nn.Linear(2, 2))
