"""Measure OmDet preprocessing, TensorRT and postprocessing on a saved frame."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from dinov3_cases.grounded import OmDetDetector, DEFAULT_PROMPT
from dinov3_cases import omdet_trt
from dinov3_cases.runtime import GROUNDING_ENGINE, TRT_PYTHON
from dinov3_cases.grounding_ipc import SharedMemoryGroundingDetector


def install_gpu_preprocessing():
    """Keep official PIL resize; normalize RGB values on CUDA for experiments."""
    import torch
    original = omdet_trt.prepare_image
    def prepare(image, metadata):
        rgb = np.array(image.convert('RGB').resize((metadata['input_size'],)*2, Image.Resampling.BILINEAR))
        pixels = torch.from_numpy(rgb).cuda().float()
        pixels = (pixels-pixels.new_tensor(metadata['mean']))/pixels.new_tensor(metadata['std'])
        result = pixels.permute(2,0,1).unsqueeze(0).contiguous()
        torch.cuda.current_stream().synchronize()
        return result
    omdet_trt.prepare_image = prepare
    return original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['single','split'],default='single')
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=60)
    parser.add_argument('--gpu-preprocess',action='store_true')
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    if args.gpu_preprocess and args.mode != 'single':
        parser.error('--gpu-preprocess requires single mode')
    original = install_gpu_preprocessing() if args.gpu_preprocess else None
    detector = (OmDetDetector(GROUNDING_ENGINE,'cuda') if args.mode=='single' else
                SharedMemoryGroundingDetector(GROUNDING_ENGINE,'cuda',python=str(TRT_PYTHON)))
    image = Image.open(args.image).convert('RGB')
    if original:
        import torch
        torch.testing.assert_close(omdet_trt.prepare_image(image,detector.metadata).cpu(),
                                   original(image,detector.metadata),rtol=1e-6,atol=1e-6)
    rows = []
    try:
        for i in range(10+args.samples):
            start = time.monotonic()
            proposals = detector.detect(image,DEFAULT_PROMPT,.3,.5,20)
            elapsed = (time.monotonic()-start)*1000
            if i>=10:
                rows.append(dict(getattr(detector,'last_metrics',{}),total_ms=elapsed,objects=len(proposals)))
    finally:
        if hasattr(detector,'close'):
            detector.close()
    summary = {key:dict(median=float(np.median([r[key] for r in rows])),p95=float(np.percentile([r[key] for r in rows],95)))
               for key in rows[0] if key.endswith('_ms')}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(model='OmDet-Turbo-Tiny',mode=args.mode,samples=args.samples,summary=summary,rows=rows),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
