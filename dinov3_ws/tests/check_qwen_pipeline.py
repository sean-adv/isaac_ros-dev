"""Real-GPU gallery retrieval smoke test. Reports latency, not retrieval accuracy."""

import json
from pathlib import Path
import time

from PIL import Image, ImageOps
import torch

from dinov3_cases.qwen_encoder import QwenEncoder
from dinov3_cases.grounded import ObjectGallery, prepare_object
from dinov3_cases.runtime import WORKSPACE, ARTIFACTS, QWEN_MODEL_ID, QWEN_REVISION


def main():
    started = time.monotonic()
    encoder = QwenEncoder()
    load_seconds = time.monotonic() - started
    gallery = ObjectGallery()
    started = time.monotonic()
    gallery.refresh(WORKSPACE / 'dinov3_cases_data', encoder, encoder.input_size, encoder.stride)
    torch.cuda.synchronize()
    gallery_seconds = time.monotonic() - started
    assert gallery.features is not None and gallery.features.shape[1] == 4096
    rows = []
    # Use one existing image per class. This verifies plumbing and consistency;
    # these images belong to the gallery and are not an independent accuracy set.
    for label in sorted(set(gallery.labels)):
        path = Path(gallery.paths[gallery.labels.index(label)])
        with Image.open(path) as image:
            prepared = prepare_object(ImageOps.exif_transpose(image), encoder.input_size, encoder.stride)
        times = []
        for _ in range(3):
            torch.cuda.synchronize()
            started = time.monotonic()
            embedding = encoder.encode(prepared)
            torch.cuda.synchronize()
            times.append((time.monotonic() - started) * 1000)
        assert embedding.shape == (1, 4096) and embedding.isfinite().all()
        torch.testing.assert_close(embedding.norm(dim=-1), torch.ones(1, device=embedding.device))
        match = gallery.classify(embedding[0], .85, .05, lab_histogram=gallery.lab_histograms[gallery.labels.index(label)])
        index, score = gallery.best_reference(embedding[0], lab_histogram=gallery.lab_histograms[gallery.labels.index(label)])
        assert score > .999 and gallery.labels[index] == label
        rows.append(dict(case=label, image=str(path), encode_ms=times, classification=match))
    report = dict(model=QWEN_MODEL_ID, revision=QWEN_REVISION, device=torch.cuda.get_device_name(),
                  load_seconds=load_seconds, gallery_seconds=gallery_seconds,
                  gallery_images=len(gallery.labels), dimensions=4096, input_size=encoder.input_size,
                  peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                  evaluation='Gallery self-retrieval smoke test; not independent accuracy evaluation.',
                  results=rows)
    target = ARTIFACTS / 'qwen3_vl_embedding_validation.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
