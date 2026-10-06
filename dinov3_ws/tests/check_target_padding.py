"""Real TensorRT check of padded target output geometry and gray control scores."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dinov3_cases.node import Encoder
from dinov3_cases.patches import resize_reference
from dinov3_cases.trt_backend import MODEL_VARIANTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODEL_VARIANTS, default='vith16plus')
    args = parser.parse_args()
    spec = MODEL_VARIANTS[args.model]
    encoder = Encoder(spec['model_id'], 'trt', spec['engine_path'], target_padding=64)
    root = Path(__file__).resolve().parents[2]
    banks = []
    paths = sorted((root/'dinov3_cases_data/punateng').glob('*.png'))
    for path in paths:
        with Image.open(path) as source:
            im = resize_reference(source.convert('RGB'), 640, 16)
        features = encoder.encode(im)
        assert len(features) == (im.height//16)*(im.width//16)
        idx = torch.linspace(0,len(features)-1,min(8192//len(paths),len(features)),device=features.device).long()
        banks.append(features[idx])
    bank = torch.cat(banks)
    report = {'model':args.model, 'scope':'Synthetic gray control plus existing reference used as geometry-only target; not user camera accuracy.', 'results':[]}
    gray = Image.new('RGB',(1280,768),(128,)*3)
    with Image.open(paths[0]) as source:
        real = source.convert('RGB')  # Incomplete bottom/right patches.
    for mode in ['reflect','replicate']:
        encoder.target_padding_mode = mode
        for padding in [0,64]:
            encoder.target_padding = padding
            for name,im in [('gray',gray),('existing_image',real)]:
                scores,grid,patches = encoder.similarity_map(im,bank,debug=True)
                assert scores.shape == (im.height,im.width)
                assert grid == (im.height//16,im.width//16)
                assert patches.shape == grid and np.isfinite(scores).all()
                assert scores.min() >= -1.0001 and scores.max() <= 1.0001
                assert (scores[grid[0]*16:] == -1).all()
                assert (scores[:,grid[1]*16:] == -1).all()
                edge = np.zeros(grid,bool);edge[[0,-1],:]=True;edge[:,[0,-1]]=True
                row = dict(target=name,mode=mode,padding=padding,size=list(im.size),
                           output_shape=list(scores.shape),grid=list(grid),
                           outer_mean=float(patches[edge].mean()))
                report['results'].append(row)
                print(json.dumps(row),flush=True)
    out = root/'dinov3_ws/diagnostics'/f'target_padding_{args.model}.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: real TRT target padding, reference sizes, original-coordinate outputs and uncovered strips.',out,flush=True)


if __name__ == '__main__':
    main()
