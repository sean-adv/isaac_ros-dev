"""Compare the new TRTMC engine through the application adapter to PyTorch."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import AutoModel
from dinov3_cases.grounded import prepare_object
from dinov3_cases.encoder import Encoder
from dinov3_cases.patches import preprocess_image


@torch.inference_mode()
def main():
    root = Path(__file__).resolve().parents[2]
    artifacts = root / "dinov3_ws/artifacts/tensorrt11"
    model_id = "facebook/dinov3-vits16-pretrain-lvd1689m"
    encoder = Encoder(artifacts / "dinov3_vits16_fp16_224.engine")
    image = prepare_object(
        Image.open(root / "dinov3_cases_data/stomach/1790733215386907348_2.png"), 224, 16
    )
    actual = encoder.encode(image)
    pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1).cuda()
    model = (
        AutoModel.from_pretrained(
            model_id, cache_dir=str(root / "dinov3_ws/artifacts/huggingface"), local_files_only=True
        )
        .cuda()
        .eval()
    )
    expected = model(pixel_values=preprocess_image(pixels, encoder.processor)).last_hidden_state[
        :, 5:
    ][0]
    expected = torch.nn.functional.normalize(expected, dim=-1)
    cosine = torch.nn.functional.cosine_similarity(actual, expected, dim=-1)
    assert actual.shape == (196, 384)
    assert cosine.mean() > 0.999 and cosine.min() > 0.99
    report = dict(
        passed=True,
        input=[1, 3, 224, 224],
        patches=list(actual.shape),
        mean_patch_cosine=float(cosine.mean()),
        min_patch_cosine=float(cosine.min()),
    )
    (artifacts / "dinov3-adapter-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
