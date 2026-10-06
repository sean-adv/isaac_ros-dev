"""Position-only interventions on identical pixels; diagnostic, not a model fix."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
from large_image_edges import summary
from reference_background import load
from dinov3_cases.patches import run_dino_image, compute_similarity_maps
from dinov3_cases.trt_backend import MODEL_VARIANTS

@torch.inference_mode()
def main():
    out = Path('dinov3_ws/diagnostics/position_results'); out.mkdir(parents=True, exist_ok=True)
    spec = MODEL_VARIANTS['vith16plus']
    model = AutoModel.from_pretrained(spec['model_id'],local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    processor = AutoImageProcessor.from_pretrained(spec['model_id'],local_files_only=True)
    def encode(im):
        return run_dino_image(torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).cuda(), model,processor,torch.bfloat16)
    banks=[]; labels=[]
    paths=sorted(Path('dinov3_cases_data/punateng').glob('*.png'))
    for path in paths:
        im, lab=load(path,640,16); f=encode(im)[0].flatten(0,1)
        idx=torch.linspace(0,len(f)-1,min(8192//len(paths),len(f)),device='cuda').long()
        banks.append(f[idx]);labels.extend(lab.ravel()[idx.cpu().numpy()])
    bank=torch.cat(banks); fg=bank[torch.as_tensor(np.asarray(labels)==0,device='cuda')]
    image=Image.new('RGB',(1280,768),(128,)*3)
    def score(im):
        f=encode(im)
        return {name:compute_similarity_maps(f,b)[0].cpu().numpy() for name,b in [('all',bank),('foreground',fg)]}
    original=score(image)
    rope=next(m for m in model.modules() if type(m).__name__=='DINOv3ViTRopePositionEmbedding')
    original_forward=rope.forward
    def shifted(pixels):
        cos,sin=original_forward(pixels)
        h,w=pixels.shape[-2]//16,pixels.shape[-1]//16
        return tuple(torch.roll(t.reshape(h,w,-1),(h//2,w//2),(0,1)).reshape(h*w,-1) for t in (cos,sin))
    rope.forward=shifted
    rolled=score(image)
    def disabled(pixels):
        cos,sin=original_forward(pixels)
        return torch.ones_like(cos),torch.zeros_like(sin)
    rope.forward=disabled
    no_rope=score(image)
    rope.forward=original_forward
    # The original constant image is identical inside this larger constant image.
    # Compare the same aligned 1280x768 pixels, now 64 pixels from the border.
    padded=score(Image.new('RGB',(1408,896),(128,)*3))
    report={'scope':'Uniform gray target; reference features unchanged. Removing RoPE is a causal diagnostic, not a validated production fix.', 'results':{}}
    for name, maps in [('normal',original),('rolled_rope',rolled),('disabled_target_rope',no_rope),('padded_gray',padded)]:
        report['results'][name]={}
        for kind,raw in maps.items():
            np.save(out/f'{name}_{kind}.npy',raw)
            report['results'][name][kind]=summary(raw)
            if name=='padded_gray':report['results'][name][kind]['original_area']=summary(raw[4:-4,4:-4])
    report['roll_equivariance']={}
    for kind,raw in rolled.items():
        expected=np.roll(original[kind],(24,40),(0,1))
        report['roll_equivariance'][kind]={'mean_abs_error':float(np.abs(raw-expected).mean()),'max_abs_error':float(np.abs(raw-expected).max()),'correlation':float(np.corrcoef(raw.ravel(),expected.ravel())[0,1])}
        assert np.abs(raw-expected).mean()<.01
        assert no_rope[kind].max()-no_rope[kind].min()<.005
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
