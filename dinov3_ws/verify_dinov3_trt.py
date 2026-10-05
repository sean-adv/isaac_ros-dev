"""One real-image smoke check of the exported engines, not an accuracy benchmark."""
from pathlib import Path
import gc
import json

import numpy as np
from PIL import Image
import tensorrt as trt
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

from export_dinov3 import MODEL_ID, OUTPUT


def run_engine(path, pixels):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Could not load {path}")
    context = engine.create_execution_context()
    assert engine.get_tensor_dtype("pixel_values") == trt.float32
    assert engine.get_tensor_dtype("last_hidden_state") == trt.float32
    assert tuple(engine.get_tensor_shape("pixel_values")) == tuple(pixels.shape)
    output = torch.empty(tuple(engine.get_tensor_shape("last_hidden_state")),
                         device="cuda", dtype=torch.float32)
    context.set_tensor_address("pixel_values", pixels.data_ptr())
    context.set_tensor_address("last_hidden_state", output.data_ptr())
    if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
        raise RuntimeError(f"Execution failed: {path}")
    torch.cuda.synchronize()
    return output.cpu()


@torch.inference_mode()
def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    image_path = sorted((Path(__file__).resolve().parents[1] /
                         "dinov3_cases_data/stomach").glob("*.png"))[0]
    image = Image.open(image_path).convert("RGB")
    # Same edge-replicated letterbox used by Encoder.encode().
    w, h = image.size
    scale = 448 / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    left, top = (448 - nw) // 2, (448 - nh) // 2
    resized = np.asarray(image.resize((nw, nh), Image.Resampling.BICUBIC))
    canvas = Image.fromarray(np.pad(resized, ((top, 448-nh-top),
                                (left, 448-nw-left), (0, 0)), mode="edge"))
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    pixels = processor(images=canvas, do_resize=False, do_center_crop=False,
                       return_tensors="pt")["pixel_values"].cuda().contiguous()
    model = AutoModel.from_pretrained(MODEL_ID, local_files_only=True,
                dtype=torch.float32, attn_implementation="eager").cuda().eval()
    expected = model(pixel_values=pixels).last_hidden_state.cpu()
    assert torch.isfinite(expected).all()
    prefix = 1 + model.config.num_register_tokens
    expected_patches = F.normalize(expected[:, prefix:].float(), dim=-1)
    # Existing application selects BF16 on Thor.
    model = model.to(dtype=torch.bfloat16)
    model.set_attn_implementation("sdpa")
    current = model(pixel_values=pixels.to(torch.bfloat16)).last_hidden_state.float().cpu()
    assert torch.isfinite(current).all()
    current_patches = F.normalize(current[:, prefix:], dim=-1)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    report = {"image": str(image_path), "input_shape": list(pixels.shape),
              "scope": "Single reference image; not end-to-end case matching validation.",
              "engines": {}}
    for precision in ("fp32", "bf16"):
        actual = run_engine(OUTPUT / f"dinov3_{precision}.engine", pixels)
        assert actual.shape == expected.shape
        assert torch.isfinite(actual).all(), f"Non-finite output from {precision}"
        error = (actual - expected).abs()
        patches = F.normalize(actual[:, prefix:], dim=-1)
        similarity = (patches * expected_patches).sum(-1)
        current_similarity = (patches * current_patches).sum(-1)
        report["engines"][precision] = {
            "output_shape": list(actual.shape), "all_finite": True,
            "max_abs_error_vs_pytorch_fp32": error.max().item(),
            "mean_abs_error_vs_pytorch_fp32": error.mean().item(),
            "patch_cosine_mean_vs_pytorch_fp32": similarity.mean().item(),
            "patch_cosine_min_vs_pytorch_fp32": similarity.min().item(),
            "patch_cosine_mean_vs_current_bf16": current_similarity.mean().item(),
            "patch_cosine_min_vs_current_bf16": current_similarity.min().item(),
        }
    (OUTPUT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
