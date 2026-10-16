"""Image retrieval with Qwen3-VL-Embedding (last-token pooling, no generation).

Matches Qwen's official image-only conversation and pooling recipe:
https://github.com/QwenLM/Qwen3-VL-Embedding/blob/main/src/models/qwen3_vl_embedding.py
Uses the built-in backbone so no downloaded Python code is executed.
"""

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen3VLModel

from .runtime import QWEN_MODEL_ID, QWEN_REVISION, QWEN_CACHE


def pool_last_token(hidden_state, attention_mask):
    positions = attention_mask.shape[1] - 1 - attention_mask.flip(1).long().argmax(1)
    rows = torch.arange(hidden_state.shape[0], device=hidden_state.device)
    vectors = hidden_state[rows, positions].float()
    if not torch.isfinite(vectors).all() or (vectors.norm(dim=-1) < 1e-8).any():
        raise ValueError("Invalid Qwen image embedding.")
    return F.normalize(vectors, dim=-1)


class QwenEncoder:
    supports_spatial_features = False
    input_size = 448
    stride = 32  # 16-pixel vision patches, merged in 2x2 groups

    def __init__(self, model_path=QWEN_MODEL_ID):
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3-VL-Embedding-8B requires CUDA in this pipeline.")
        self.device = "cuda:0"
        options = dict(cache_dir=str(QWEN_CACHE))
        if str(model_path) == QWEN_MODEL_ID:
            options["revision"] = QWEN_REVISION
        self.model = Qwen3VLModel.from_pretrained(
            str(model_path), dtype=torch.bfloat16,
            attn_implementation="sdpa", **options,
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(str(model_path), **options)
        self.description = "Qwen3-VL-Embedding-8B BF16 (global cosine similarity)"

    @torch.inference_mode()
    def encode(self, image):
        image = image.convert("RGB")
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
            {"role": "user", "content": [{"type": "image", "image": image}]},
        ]
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
        )
        # The caller supplies an already resized image divisible by the vision stride.
        if image.width % self.stride or image.height % self.stride:
            raise ValueError("Qwen input dimensions must be divisible by 32.")
        inputs = self.processor(
            text=[text], images=[image], do_resize=False,
            padding=True, return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**inputs, use_cache=False, return_dict=True)
        # [1, 4096] is one global descriptor, not a bank of spatial patch tokens.
        return pool_last_token(outputs.last_hidden_state, inputs["attention_mask"])
