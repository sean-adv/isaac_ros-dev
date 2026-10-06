"""Controlled edge diagnostics. Leaves production and source images unchanged.

Default targets are synthetic controls, NOT the user's camera image.
Use --target for native-resolution reproduction. All statistics use raw patches.
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/dinov3_cases'))
from dinov3_cases.patches import run_dino_image, compute_similarity_maps, resize_reference
from dinov3_cases.trt_backend import TRTBackbone, MODEL_VARIANTS
from reference_background import load


def regions(shape):
    y, x = np.indices(shape)
    d = np.minimum.reduce([y, x, shape[0]-1-y, shape[1]-1-x])
    return {'outer1': d == 0, 'ring1_2': (d >= 1) & (d < 3), 'interior': d >= 3}


def summary(raw):
    return {name: {'n': int(m.sum()), 'mean': float(raw[m].mean()),
                   'p95': float(np.quantile(raw[m], .95)),
                   'fraction_ge_070': float((raw[m] >= .7).mean())}
            for name, m in regions(raw.shape).items() if m.any()}


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', type=Path)
    p.add_argument('--references', type=Path, default=Path('dinov3_cases_data/punateng'))
    p.add_argument('--model', choices=MODEL_VARIANTS, default='vith16plus')
    p.add_argument('--backends', nargs='+', choices=['trt', 'torch'], default=['trt', 'torch'])
    p.add_argument('--out', type=Path, default=Path('dinov3_ws/diagnostics/large_edge_results'))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    spec = MODEL_VARIANTS[args.model]
    processor = AutoImageProcessor.from_pretrained(spec['model_id'], local_files_only=True)
    refs = []
    for path in sorted(args.references.iterdir()):
        if path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
            rgb, labels = load(path, 640, 16)
            refs.append((path, rgb, labels))
    if not refs:
        raise ValueError('No references')
    targets = {}
    for size in [(640, 384), (1280, 768), (1920, 1152)]:
        for color, value in [('gray', 128), ('black', 0)]:
            targets[f'{color}_{size[0]}'] = Image.new('RGB', size, (value,)*3)
    rng = np.random.default_rng(42)
    targets['noise_1280'] = Image.fromarray(rng.integers(0,256,(768,1280,3),dtype=np.uint8))
    if args.target:
        with Image.open(args.target) as im:
            original = ImageOps.exif_transpose(im).convert('RGB')
        targets['target_native'] = original
        targets['target_640'] = resize_reference(original, 640, 16)
        # Whole-patch padding preserves the original patch alignment. Remove any
        # incomplete bottom/right strip to keep the compared region exact.
        w, h = original.size
        aligned = np.asarray(original)[:h//16*16, :w//16*16]
        targets['target_aligned'] = Image.fromarray(aligned)
        for mode in ['reflect', 'edge']:
            targets[f'target_pad_{mode}'] = Image.fromarray(np.pad(aligned, ((64,64),(64,64),(0,0)),mode=mode))
    report = {'model': spec['model_id'], 'target': str(args.target) if args.target else None,
              'scope': 'Synthetic controls plus user target if supplied; no camera-image causal claim without target.',
              'references': [str(r[0]) for r in refs], 'results': {}, 'backend_comparison': {}}
    baseline = {}
    for backend in args.backends:
        print('loading', backend, flush=True)
        if backend == 'trt':
            model = TRTBackbone(spec['engine_path'], spec['model_id']); dtype = torch.float32
        else:
            dtype = torch.bfloat16
            model = AutoModel.from_pretrained(spec['model_id'], local_files_only=True,
                    dtype=dtype, attn_implementation='sdpa').cuda().eval()
        def encode(im):
            pixels = torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).cuda()
            return run_dino_image(pixels, model, processor, dtype)
        banks, labels_all, edge_all, origin = [], [], [], []
        for path, im, labels in refs:
            features = encode(im)[0]
            flat = features.flatten(0,1)
            indices = torch.linspace(0,len(flat)-1,min(8192//len(refs),len(flat)),device='cuda').long()
            idx = indices.cpu().numpy()
            banks.append(flat[indices]); labels_all.extend(labels.ravel()[idx])
            y,x = np.indices(labels.shape)
            near = (y<2)|(x<2)|(y>=labels.shape[0]-2)|(x>=labels.shape[1]-2)
            edge_all.extend(near.ravel()[idx])
            origin.extend([{'file':path.name,'row':int(i//labels.shape[1]),'col':int(i%labels.shape[1])} for i in idx])
        bank = torch.cat(banks)
        labels_all, edge_all = np.asarray(labels_all), np.asarray(edge_all)
        masks = {'all': np.ones(len(bank),bool), 'foreground': labels_all == 0,
                 'foreground_no_ref_edge2': (labels_all == 0)&~edge_all}
        report.setdefault('bank_counts', {})[backend] = {k:int(v.sum()) for k,v in masks.items()}
        for name, im in targets.items():
            print(backend, name, im.size, flush=True)
            feat = encode(im)
            maps, row = {}, {'size':list(im.size), 'banks':{}}
            for kind, mask in masks.items():
                if not mask.any():
                    continue
                raw = compute_similarity_maps(feat, bank[torch.as_tensor(mask,device='cuda')])[0].cpu().numpy()
                maps[kind] = raw
                row['banks'][kind] = summary(raw)
                if name.startswith('target_pad_'):
                    row['banks'][kind]['original_area'] = summary(raw[4:-4,4:-4])
                np.save(args.out / f'{backend}_{name}_{kind}.npy', raw)
            # Attribution to the actual winning reference, before interpolation.
            flat = feat[0].flatten(0,1)
            winners = torch.cat([(chunk @ bank.T).argmax(-1) for chunk in flat.split(1024)]).cpu().numpy()
            for region, mask in regions(maps['all'].shape).items():
                chosen = winners[mask.ravel()]
                values, counts = np.unique(chosen,return_counts=True)
                order = np.argsort(counts)[-5:][::-1]
                row.setdefault('winners',{})[region] = {
                    'reference_outer2_fraction':float(edge_all[chosen].mean()),
                    'alpha_category_fractions': [float((labels_all[chosen]==j).mean()) for j in range(3)],
                    'top5': [dict(origin[int(values[i])], count=int(counts[i])) for i in order]}
            if name in baseline:
                ref_feat, ref_maps = baseline.pop(name)
                cosine = (feat.cpu()*ref_feat).sum(-1)[0].numpy()
                report['backend_comparison'][name] = {
                    'feature_cosine':summary(cosine),
                    'score_abs_difference': {k:summary(np.abs(v-ref_maps[k])) for k,v in maps.items()}}
            elif backend == args.backends[0] and len(args.backends)>1:
                baseline[name] = (feat.cpu(), maps)
            panels = []
            for kind, raw in maps.items():
                colored = cv2.applyColorMap((np.clip(raw,0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO)
                colored = cv2.resize(colored,(480,round(480*im.height/im.width)),interpolation=cv2.INTER_NEAREST)
                colored = cv2.copyMakeBorder(colored,28,0,0,0,cv2.BORDER_CONSTANT)
                cv2.putText(colored,kind,(5,19),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                panels.append(colored)
            cv2.imwrite(str(args.out/f'{backend}_{name}.jpg'),np.concatenate(panels,axis=1))
            report['results'][f'{backend}/{name}'] = row
            (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(row['banks']),flush=True)
            del feat, flat
        del model, bank, banks, features
        gc.collect(); torch.cuda.empty_cache()
    print('report:',args.out/'report.json',flush=True)

if __name__ == '__main__':
    main()
