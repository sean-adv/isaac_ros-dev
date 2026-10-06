"""Offline reference-bank attribution; does not change the ROS pipeline or data.

Use alpha coverage >= .95 for foreground, <= .05 for background, and
otherwise boundary. With no --target, leave each cutout out of the bank
and use it as a diagnostic target, not as a substitute for a camera frame.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoImageProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/dinov3_cases'))
from dinov3_cases.patches import (resize_reference, run_dino_image,
                                compute_similarity_maps, similarity_pixels)
from dinov3_cases.trt_backend import TRTBackbone, MODEL_VARIANTS, DEFAULT_VARIANT


def load(path, max_side, stride):
    with Image.open(path) as source:
        rgba = ImageOps.exif_transpose(source).convert('RGBA')
        # Match production: convert RGB BEFORE resizing. Resize alpha separately.
        rgb = resize_reference(rgba.convert('RGB'), max_side, stride)
        alpha = np.asarray(rgba.getchannel('A').resize(rgb.size, Image.Resampling.BOX)) / 255.
    h, w = rgb.height // stride, rgb.width // stride
    coverage = alpha[:h * stride, :w * stride].reshape(h, stride, w, stride).mean((1, 3))
    labels = np.where(coverage >= .95, 0, np.where(coverage <= .05, 1, 2))
    return rgb, labels


def stats(values, region, threshold):
    v = values[region]
    if not len(v):
        return {'n': 0}
    return dict(n=int(len(v)), mean=float(v.mean()), p95=float(np.quantile(v, .95)),
                above_threshold_fraction=float((v >= threshold).mean()))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', type=Path, default=Path('dinov3_cases_data/punateng'))
    parser.add_argument('--target', type=Path)
    parser.add_argument('--model', choices=MODEL_VARIANTS, default=DEFAULT_VARIANT)
    parser.add_argument('--max-side', type=int, default=640)
    parser.add_argument('--threshold', type=float, default=.7)
    parser.add_argument('--out', type=Path, default=Path('dinov3_ws/diagnostics/background_results'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    spec = MODEL_VARIANTS[args.model]
    model = TRTBackbone(spec['engine_path'], spec['model_id'])
    processor = AutoImageProcessor.from_pretrained(spec['model_id'], local_files_only=True)
    stride = model.config.patch_size

    def encode(rgb):
        pixels = torch.from_numpy(np.asarray(rgb).copy()).permute(2, 0, 1).cuda()
        return run_dino_image(pixels, model, processor)

    records = []
    for path in sorted(args.references.iterdir()):
        if path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}:
            continue
        rgb, labels = load(path, args.max_side, stride)
        feat = encode(rgb)
        records.append((path, rgb, labels, feat))
        print('encoded', path.name, rgb.size, np.bincount(labels.ravel(), minlength=3).tolist(), flush=True)
    if len(records) < 2 and args.target is None:
        raise ValueError('Leave-one-out diagnostics need at least two references.')
    report = dict(model=spec['model_id'], backend=model.description, threshold=args.threshold,
                  reference_max_side=args.max_side,
                  scope='User target' if args.target else 'Leave-one-out cutouts; not the original camera frame',
                  labels=['foreground', 'background', 'boundary'],
                  references=[dict(file=str(p), size=list(im.size),
                                   patch_counts=np.bincount(l.ravel(), minlength=3).tolist())
                              for p, im, l, _ in records], results=[])
    tasks = [(p, im, lab, feat, i) for i, (p, im, lab, feat) in enumerate(records)]
    if args.target:
        # Target is kept at native resolution, matching production.
        im, lab = load(args.target, 0, stride)
        tasks = [(args.target, im, lab, encode(im), None)]
    rng = np.random.default_rng(0)
    for path, im, target_labels, feat, excluded in tasks:
        selected = [r for i, r in enumerate(records) if i != excluded]
        limit = 8192 // len(selected)  # Same default per-image bank sampling as production.
        banks, categories = [], []
        for _, _, lab, f in selected:
            flat = f[0].reshape(-1, f.shape[-1])
            idx = torch.linspace(0, len(flat)-1, min(limit, len(flat)), device=flat.device).long()
            banks.append(flat[idx]); categories.append(lab.ravel()[idx.cpu().numpy()])
        bank = torch.cat(banks)
        cats = np.concatenate(categories)
        maps = {'all': compute_similarity_maps(feat, bank)[0].cpu().numpy()}
        for label, name in enumerate(report['labels']):
            chosen = bank[torch.as_tensor(cats == label, device=bank.device)]
            if len(chosen):
                maps[name] = compute_similarity_maps(feat, chosen)[0].cpu().numpy()
        if 'foreground' in maps and 'background' in maps:
            n = min(int((cats == 0).sum()), int((cats == 1).sum()))
            for label, name in enumerate(['foreground', 'background']):
                indices = rng.choice(np.flatnonzero(cats == label), n, replace=False)
                maps['equal_count_' + name] = compute_similarity_maps(
                    feat, bank[torch.as_tensor(indices, device=bank.device)])[0].cpu().numpy()
        names = [n for n in report['labels'] if n in maps]
        winner = np.asarray(names)[np.stack([maps[n] for n in names]).argmax(0)]
        edge = np.zeros(target_labels.shape, bool)
        edge[[0, -1], :] = True; edge[:, [0, -1]] = True
        regions = {'outer_patch_ring': edge, 'alpha_background': target_labels == 1,
                   'alpha_foreground': target_labels == 0}
        result = dict(target=str(path), excluded_from_bank=excluded is not None,
                      bank_counts=np.bincount(cats, minlength=3).tolist(),
                      regions={})
        for region_name, region in regions.items():
            result['regions'][region_name] = dict(
                maps={name: stats(v, region, args.threshold) for name, v in maps.items()},
                winner_fractions={name: float((winner[region] == name).mean()) if region.any() else None
                                  for name in names})
        # Bilinear interpolation cannot exceed the original patch-score extrema.
        pixels = similarity_pixels(torch.from_numpy(maps['all']), (im.height, im.width), stride).numpy()
        covered = pixels[:target_labels.shape[0]*stride, :target_labels.shape[1]*stride]
        assert covered.max() <= maps['all'].max() + 1e-5
        assert covered.min() >= maps['all'].min() - 1e-5
        assert np.allclose(maps['all'], np.stack([maps[n] for n in names]).max(0), atol=1e-5)
        result['interpolation'] = dict(patch_max=float(maps['all'].max()), pixel_max=float(covered.max()))
        tiles = []
        for name, raw in maps.items():
            np.save(args.out / f'{path.stem}_{name}.npy', raw)
            color = cv2.applyColorMap((np.clip(raw, 0, 1)*255).astype(np.uint8), cv2.COLORMAP_TURBO)
            tile = cv2.resize(color, im.size, interpolation=cv2.INTER_NEAREST)
            tile = cv2.copyMakeBorder(tile, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30,30,30))
            cv2.putText(tile, name, (4,21), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255),1)
            tiles.append(tile)
        cv2.imwrite(str(args.out / f'{path.stem}_comparison.png'), np.concatenate(tiles, axis=1))
        report['results'].append(result)
        print('done', path.name, json.dumps(result['regions']['outer_patch_ring']), flush=True)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('report:', args.out / 'report.json', flush=True)


if __name__ == '__main__':
    main()
