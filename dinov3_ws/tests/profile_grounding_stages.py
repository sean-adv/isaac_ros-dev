"""Warmup and split Grounding latency on a saved real camera frame."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from dinov3_cases.grounded import GroundingDetector, DEFAULT_PROMPT
from dinov3_cases.grounding_trt import GroundingTRT
from dinov3_cases.runtime import LEGACY_GROUNDING_ENGINE, GROUNDING_ENGINE


def install_gpu_preprocessing():
    """Keep CPU resize, cache text, move pixel normalization to CUDA."""
    import cv2
    import torch
    from transformers import BatchEncoding
    original = GroundingTRT.prepare_inputs
    def prepare(self, processor, image, prompt):
        cached = getattr(self, '_profile_text_inputs', None)
        if cached is None or cached[0] != prompt:
            baseline = original(self, processor, image, prompt)
            text = {k: v for k, v in baseline.items() if k != 'img'}
            self._profile_text_inputs = (prompt, text)
        else:
            text = cached[1]
        rgb = cv2.resize(np.asarray(image.convert('RGB')), (self.width, self.height))
        pixels = torch.from_numpy(rgb).to(self.device).float().div_(255.0)
        mean = pixels.new_tensor([0.485, 0.456, 0.406])
        std = pixels.new_tensor([0.229, 0.224, 0.225])
        pixels.sub_(mean).div_(std)
        pixels = pixels.permute(2, 0, 1).unsqueeze(0).contiguous()
        # Finish normalization here so timing does not shift into the TRT stage.
        torch.cuda.current_stream(self.device).synchronize()
        return BatchEncoding(dict(text, img=pixels))
    GroundingTRT.prepare_inputs = prepare
    return original


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['split','single'], default='split')
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=60)
    p.add_argument('--gpu-preprocess', action='store_true')
    args = p.parse_args()
    baseline_prepare = install_gpu_preprocessing() if args.gpu_preprocess else None
    current = {}
    def timed(method, key):
        original = getattr(GroundingTRT, method)
        def wrapper(self, *a, **kw):
            start=time.monotonic()
            out=original(self,*a,**kw)
            current[key]=(time.monotonic()-start)*1000
            return out
        setattr(GroundingTRT, method, wrapper)
    timed('prepare_inputs','prepare_ms')
    timed('__call__','trt_call_ms')
    detector=GroundingDetector(LEGACY_GROUNDING_ENGINE if args.mode=='split' else GROUNDING_ENGINE, 'cuda')
    image=Image.open(args.image).convert('RGB')
    parity = None
    if baseline_prepare:
        import torch
        reference = baseline_prepare(detector.model, detector.processor, image, DEFAULT_PROMPT)
        candidate = detector.model.prepare_inputs(detector.processor, image, DEFAULT_PROMPT)
        parity = max(float((reference[k].float()-candidate[k].float()).abs().max()) for k in reference)
        assert parity < 1e-6, f'Preprocessing mismatch: {parity}'
        baseline_outputs = detector.model(**reference)
        gpu_outputs = detector.model(**candidate)
        output_diff = {}
        for key in ['logits','pred_boxes']:
            ref, out = getattr(baseline_outputs,key), getattr(gpu_outputs,key)
            torch.testing.assert_close(ref, out, rtol=1e-3, atol=5e-3, equal_nan=True)
            finite = torch.isfinite(ref) & torch.isfinite(out)
            assert torch.equal(torch.isfinite(ref), torch.isfinite(out))
            output_diff[key] = float((ref[finite]-out[finite]).abs().max()) if finite.any() else 0.0
    rows=[]
    for index in range(10+args.samples):
        current.clear()
        start=time.monotonic()
        proposals=detector.detect(image,DEFAULT_PROMPT,0.3,0.25,0.5,20)
        total=(time.monotonic()-start)*1000
        if index>=10:
            rows.append(dict(current,total_ms=total,postprocess_ms=total-sum(current.values()),objects=len(proposals)))
    summary={k:dict(mean=float(np.mean([r[k] for r in rows])),median=float(np.median([r[k] for r in rows])),p95=float(np.percentile([r[k] for r in rows],95))) for k in rows[0]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(mode=args.mode,gpu_preprocess=args.gpu_preprocess,input_max_abs_diff=parity,output_max_abs_diff=output_diff if baseline_prepare else None,samples=len(rows),summary=summary,rows=rows),indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
