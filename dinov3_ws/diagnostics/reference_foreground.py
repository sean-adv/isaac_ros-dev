"""Audit current foreground-only matching, with leave-one-out targets by default.

Uses production loading/resizing/filtering and sampling. Eroded banks are diagnostic
ablations, not changes to the service. Target foreground uses mean alpha >= .95.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoImageProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/dinov3_cases'))
from dinov3_cases.patches import (load_image, resize_reference, reference_patch_mask,
                                run_dino_image, compute_similarity_maps)
from dinov3_cases.trt_backend import TRTBackbone, MODEL_VARIANTS, DEFAULT_VARIANT


def summary(scores, region, threshold):
    values = scores[region]
    if not len(values):
        return {'n': 0}
    return dict(n=int(len(values)), mean=float(values.mean()),
                above_threshold_fraction=float((values >= threshold).mean()))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', type=Path, default=Path('dinov3_cases_data/punateng'))
    parser.add_argument('--target', type=Path)
    parser.add_argument('--model', choices=MODEL_VARIANTS, default=DEFAULT_VARIANT)
    parser.add_argument('--max-side', type=int, default=640)
    parser.add_argument('--threshold', type=float, default=.7)
    parser.add_argument('--out', type=Path, default=Path('/tmp/dinov3_foreground_audit'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    spec = MODEL_VARIANTS[args.model]
    model = TRTBackbone(spec['engine_path'], spec['model_id'])
    processor = AutoImageProcessor.from_pretrained(spec['model_id'], local_files_only=True)
    stride = model.config.patch_size

    def record(path, max_side):
        image = resize_reference(load_image(path, preserve_alpha=True), max_side, stride)
        rgb = np.asarray(image.convert('RGB')).copy()
        features = run_dino_image(torch.from_numpy(rgb).permute(2, 0, 1).cuda(), model, processor)
        alpha = np.asarray(image.getchannel('A'), dtype=np.float32) / 255.
        gh, gw = features.shape[1:3]
        coverage = alpha[:gh*stride, :gw*stride].reshape(gh, stride, gw, stride).mean((1, 3))
        flat = features[0].reshape(-1, features.shape[-1])
        masks = {'all': np.ones((gh, gw), bool)}
        for threshold in (.95, .99, 1.):
            masks[f'fg_{threshold:g}'] = reference_patch_mask(image, stride, threshold).reshape(gh, gw)
        for margin in (1, 2):
            masks[f'fg_0.99_erode_{margin}'] = reference_patch_mask(
                image, stride, .99, margin).reshape(gh, gw)
        print('encoded', path.name, image.size, {k:int(v.sum()) for k,v in masks.items()}, flush=True)
        return dict(path=path, image=image, features=features, flat=flat, coverage=coverage, masks=masks)

    records = [record(p, args.max_side) for p in sorted(args.references.iterdir())
               if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}]
    if not records or (not args.target and len(records) < 2):
        raise ValueError('Need references and at least two images for leave-one-out.')
    tasks = [(record(args.target, 0), None)] if args.target else list(zip(records, range(len(records))))
    report = dict(model=spec['model_id'], reference_max_side=args.max_side,
                  threshold=args.threshold, scope='native target' if args.target else 'resized leave-one-out cutouts',
                  results=[])
    for target, excluded in tasks:
        refs = [r for i,r in enumerate(records) if i != excluded]
        limit = 8192 // len(refs)
        coverage = target['coverage']
        edge = np.zeros(coverage.shape, bool)
        edge[[0,-1],:] = True
        edge[:,[0,-1]] = True
        regions = dict(outer_patch_ring=edge, alpha_background=coverage <= .05,
                       alpha_boundary=(coverage > .05) & (coverage < .95), alpha_foreground=coverage >= .95,
                       outer_background=edge & (coverage <= .05), outer_boundary=edge & (coverage > .05) & (coverage < .95),
                       outer_foreground=edge & (coverage >= .95))
        result = dict(target=str(target['path']), excluded_from_bank=excluded is not None, banks={})
        for name in refs[0]['masks']:
            banks = []
            counts = []
            for ref in refs:
                keep = torch.as_tensor(ref['masks'][name].reshape(-1), device=ref['flat'].device)
                selected = ref['flat'][keep]
                counts.append(dict(file=str(ref['path']), eligible=int(len(selected))))
                if len(selected):
                    indices = torch.linspace(0, len(selected)-1, min(limit,len(selected)), device=selected.device).long()
                    banks.append(selected[indices])
            if not banks:
                result['banks'][name] = dict(error='no eligible patches', references=counts)
                continue
            bank = torch.cat(banks)
            scores = compute_similarity_maps(target['features'], bank)[0].cpu().numpy()
            np.save(args.out / f'{target["path"].stem}_{name}.npy', scores)
            result['banks'][name] = dict(sampled=int(len(bank)), references=counts,
                                        regions={k:summary(scores,v,args.threshold) for k,v in regions.items()})
        report['results'].append(result)
        print('done', target['path'].name, flush=True)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('report:', args.out / 'report.json', flush=True)


if __name__ == '__main__':
    main()
