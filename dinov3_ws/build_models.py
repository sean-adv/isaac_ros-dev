"""Build/validate official Torch-TensorRT SAM3 and Cat2Real without sibling repos."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src/dinov3_cases'))
from dinov3_cases.cat2real import MODEL_ID as CAT_ID, REVISION as CAT_REVISION
from dinov3_cases.runtime import SAM3_REVISION

SPECS = {
    'cat2real': dict(model_id=CAT_ID, revision=CAT_REVISION, directory='cat2real-dinov3-384',
                     size=384, dtype='fp32', filename='cat2real-fp32-trt.ts'),
    'sam3': dict(model_id='facebook/sam3', revision=SAM3_REVISION, directory='sam3',
                 size=1008, dtype='fp16', filename='vision-fp16.ts'),
}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def metrics(reference, actual):
    import torch
    a,b=reference.float(),actual.float()
    finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:return dict(finite=False,passed=False)
    relative=float((a-b).norm()/a.norm().clamp_min(1e-12))
    cosine=float(torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0))
    return dict(finite=True,relative_l2=relative,cosine=cosine)


def build(target, args):
    import torch
    import torch_tensorrt
    import tensorrt as trt
    from transformers import AutoModel, AutoImageProcessor, Sam3Model
    from PIL import Image
    import numpy as np
    from dinov3_cases.model_modules import Cat2RealFeatures, Sam3VisionFeatures
    from dinov3_cases.cat2real import prepare_cat2real_image
    from dinov3_cases.sam3_trt import prepare_sam3_pixels
    spec=SPECS[target];out=args.artifacts/spec['directory'];source=out/'source'
    out.mkdir(parents=True,exist_ok=True)
    if args.download:
        from huggingface_hub import snapshot_download
        snapshot_download(spec['model_id'],revision=spec['revision'],local_dir=source,
            allow_patterns=['*.json','*.safetensors','*.txt','*.model'])
    if not (source/'model.safetensors').is_file():
        raise FileNotFoundError(f'Missing {source}; use --download after HF access is configured.')
    torch.manual_seed(0);torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dtype=torch.float32 if spec['dtype']=='fp32' else torch.float16
    path=out/spec['filename']
    def load(dtype):
        cls=AutoModel if target=='cat2real' else Sam3Model
        model=cls.from_pretrained(str(source),local_files_only=True,dtype=dtype).eval()
        return (Cat2RealFeatures(model) if target=='cat2real' else Sam3VisionFeatures(model.vision_encoder)).cuda().eval()
    compile_seconds=None
    if not args.validate_only:
        path.with_suffix('.json').unlink(missing_ok=True)
        model=load(dtype)
        example=torch.zeros((1,3,spec['size'],spec['size']),device='cuda',dtype=dtype)
        start=time.perf_counter()
        compiled=torch_tensorrt.compile(model,ir='dynamo',inputs=[example],require_full_compilation=True)
        compile_seconds=time.perf_counter()-start
        torch_tensorrt.save(compiled,str(path),output_format='torchscript',inputs=[example])
        del compiled,model
        torch.cuda.empty_cache()
    compiled=torch.jit.load(str(path)).eval().cuda()
    reference=load(torch.float32)
    processor=AutoImageProcessor.from_pretrained(str(source),local_files_only=True) if target=='cat2real' else None
    images=[('black',Image.new('RGB',(73,151),'black')),('white',Image.new('RGB',(151,73),'white')),
            ('noise',Image.fromarray(np.random.default_rng(0).integers(0,256,(231,317,3),dtype=np.uint8)))]
    for p in sorted((ROOT/'data/gallery').rglob('*.png'))[:4]:
        with Image.open(p) as image:images.append((p.name,image.convert('RGB')))
    sample=ROOT/'artifacts/validation/person.jpg'
    if sample.is_file():
        with Image.open(sample) as image:images.append(('person.jpg',image.convert('RGB')))
    checks=[]
    with torch.inference_mode():
        for name,image in images:
            if target=='cat2real':
                x=processor(images=prepare_cat2real_image(image),return_tensors='pt')['pixel_values'].cuda()
            else:x=torch.from_numpy(prepare_sam3_pixels(image)).cuda()
            a,b=reference(x),compiled(x.to(dtype))
            if target=='cat2real':
                values=[metrics(a,b),metrics(a[:,0],b[:,0])]
                passed=all(v['finite'] and v['cosine']>=.999 and v['relative_l2']<=.01 for v in values)
            else:
                # Compare the FPN features used by bbox and segmentation; positions are deterministic.
                values=[metrics(a[i],b[i]) for i in (1,2,3)]
                passed=all(v['finite'] and v['cosine']>=.99 and v['relative_l2']<=.15 for v in values)
            checks.append(dict(image=name,passed=passed,outputs=values))
            print(target, name, 'PASS' if passed else 'FAIL',flush=True)
        for _ in range(10):compiled(x.to(dtype))
        torch.cuda.synchronize();times=[]
        for _ in range(50):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();compiled(x.to(dtype));end.record();end.synchronize();times.append(start.elapsed_time(end))
    metadata=dict(model_id=spec['model_id'],revision=spec['revision'],dtype=spec['dtype'],
        input_shape=[1,3,spec['size'],spec['size']],checkpoint_sha256=digest(source/'model.safetensors'),
        artifact_sha256=digest(path),torch=torch.__version__,torch_tensorrt=torch_tensorrt.__version__,
        tensorrt=trt.__version__,gpu=torch.cuda.get_device_name(),require_full_compilation=True)
    if target=='cat2real':metadata.update(hidden_size=1024,patch_size=16,num_register_tokens=4,descriptor='normalized CLS/pooler_output')
    report=dict(passed=all(v['passed'] for v in checks),checks=checks,metadata=metadata,compile_seconds=compile_seconds,
        benchmark=dict(mean_ms=float(np.mean(times)),p50_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),warmup=10,iterations=50),
        scope='Reloaded TRT vs fresh original FP32; model-only GPU timing; no AP claim')
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:raise RuntimeError(f'{target} validation failed; see {out}/validation.json')
    path.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(dict(target=target,passed=True,benchmark=report['benchmark'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',choices=['all','sam3','cat2real'],default='all')
    p.add_argument('--artifacts',type=Path,default=ROOT/'artifacts')
    p.add_argument('--download',action='store_true',help='Download pinned official checkpoints/tokenizers')
    p.add_argument('--validate-only',action='store_true',help='Validate existing artifacts without recompiling')
    args=p.parse_args();args.artifacts=args.artifacts.resolve()
    for target in (SPECS if args.target=='all' else [args.target]):build(target,args)


if __name__=='__main__':main()
