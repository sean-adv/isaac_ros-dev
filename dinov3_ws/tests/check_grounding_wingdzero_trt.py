"""Compare wingdzero ONNX/TRT through the production input and postprocess adapter."""

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch
from dinov3_cases.grounded import GroundingDetector, clean_proposals
from dinov3_cases.trt_backend import ARTIFACTS


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine", type=Path, default=ARTIFACTS / "grounding_dino_wingdzero_fp16.engine"
    )
    parser.add_argument(
        "--report", type=Path, default=ARTIFACTS / "verification_grounding_wingdzero.json"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    image = Image.open(root / "dinov3_cases_data/stomach/1790733215386907348_2.png").convert("RGB")
    detector = GroundingDetector(args.engine, "cuda")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(ARTIFACTS / "grounding_dino_wingdzero.onnx"),
        options,
        providers=["CPUExecutionProvider"],
    )
    report = {
        "scope": "One real medicine image, full/short prompts, and rotated image; not a dataset accuracy benchmark.",
        "engine": str(args.engine),
        "checks": [],
    }
    for picture, prompt in [
        (image, "medicine bottle. medicine box. bottle. box."),
        (image, "box. bottle."),
        (image.rotate(90, expand=True), "medicine bottle. medicine box."),
    ]:
        inputs = detector.model.prepare_inputs(detector.processor, picture, prompt)
        assert tuple(inputs.img.shape) == (1, 3, 800, 1200)
        assert tuple(inputs.input_ids.shape) == (1, 12)
        feed = {item.name: inputs[item.name].cpu().numpy() for item in session.get_inputs()}
        reference_values = session.run(["logits", "boxes"], feed)
        expected = SimpleNamespace(
            logits=torch.from_numpy(reference_values[0]).cuda(),
            pred_boxes=torch.from_numpy(reference_values[1]).cuda(),
        )
        for _ in range(3):
            actual = detector.model(**inputs)
        start = time.perf_counter()
        for _ in range(5):
            actual = detector.model(**inputs)
        elapsed = (time.perf_counter() - start) * 1000 / 5
        assert not torch.isnan(actual.logits).any()
        assert torch.isfinite(actual.pred_boxes).all()
        proposals = []
        for outputs in (expected, actual):
            result = detector.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=0.3,
                text_threshold=0.25,
                target_sizes=[(picture.height, picture.width)],
            )[0]
            proposals.append(
                clean_proposals(
                    result["boxes"].cpu().tolist(),
                    result["scores"].cpu().tolist(),
                    result["text_labels"],
                    picture.size,
                    0.5,
                    20,
                )
            )
        before, after = proposals
        assert len(before) > 0 and len(before) == len(after), (prompt, before, after)
        a = np.asarray([item["bbox"] for item in before])
        b = np.asarray([item["bbox"] for item in after])
        costs = np.abs(a[:, None] - b[None, :]).sum(-1)
        rows, cols = linear_sum_assignment(costs)
        intersection = np.maximum(
            0, np.minimum(a[rows, 2:], b[cols, 2:]) - np.maximum(a[rows, :2], b[cols, :2])
        ).prod(-1)
        union = (
            (a[rows, 2:] - a[rows, :2]).prod(-1)
            + (b[cols, 2:] - b[cols, :2]).prod(-1)
            - intersection
        )
        min_iou = float((intersection / union).min())
        score_error = max(
            abs(before[i]["detection_score"] - after[j]["detection_score"])
            for i, j in zip(rows, cols)
        )
        row = {
            "prompt": prompt,
            "image_wh": list(picture.size),
            "real_text_tokens": int(inputs.attention_mask.sum()),
            "engine_text_tokens": 12,
            "detections_after_nms": len(after),
            "matched_iou_min": min_iou,
            "matched_score_error_max": score_error,
            "trt_adapter_ms": elapsed,
        }
        report["checks"].append(row)
        print(json.dumps(row), flush=True)
        assert min_iou > 0.95 and score_error < 0.03, row
        assert [before[i]["detection_label"] for i in rows] == [
            after[j]["detection_label"] for j in cols
        ]
    try:
        detector.validate_prompt("medicine bottle. medicine box. bottle. box. car. truck.")
        raise AssertionError("Over-length prompt accepted")
    except ValueError as error:
        assert "tokens" in str(error)
    report["passed"] = True
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "PASS: wingdzero ONNX/TRT, shorter padded prompts, rotated image, over-length prompt rejection"
    )


if __name__ == "__main__":
    main()
