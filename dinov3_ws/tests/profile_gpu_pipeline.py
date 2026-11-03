"""Compare identical saved RGB-D inputs on legacy and CUDA/native paths.

Usage: python tests/profile_gpu_pipeline.py --samples frames.npz --metadata calibration.json --output report.json [--models]
The model option loads the original and dynamic-batch engines, preserving FP32.
"""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import cv2,numpy as np


def median_ms(function,repeats=7):
    function();values=[]
    for _ in range(repeats):
        start=time.perf_counter();function();values.append((time.perf_counter()-start)*1000)
    return dict(median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),samples=values)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--baseline-depth',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--models',action='store_true');args=p.parse_args()
    cv2.setNumThreads(1)
    raw=np.load(args.samples);rgb,depth=raw['rgb'],raw['depth'];meta=json.loads(args.metadata.read_text())
    info=SimpleNamespace(**(meta['info']|{'roi':SimpleNamespace(**meta['info']['roi'])}))
    boxes=[item['bbox'] for item in meta['detections']]
    from dinov3_cases.geometry.depth_bbox import ProjectionRayCache, attach_bboxes3d
    from dinov3_cases.gpu_geometry import CudaPlaneFitter
    from dinov3_cases.tracking.nvdcf import NvDCF
    from dinov3_cases.api.visualization import render_display_overlay
    report=dict(scope='Identical saved-input stage timing, not live camera end-to-end latency',capture_stamps=meta.get('stamps'),stages={})
    stages=report['stages']
    import torch
    properties=torch.cuda.get_device_properties(0)
    report['device']=dict(name=properties.name,multi_processor_count=properties.multi_processor_count,
                          capability=[properties.major,properties.minor],cuda=torch.version.cuda)
    if args.baseline_depth:
        import importlib.util
        spec=importlib.util.spec_from_file_location('baseline_depth_bbox',args.baseline_depth)
        baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
        old_cache=baseline.ProjectionRayCache()
        def old_measure():
            items=[dict(bbox=list(b)) for b in boxes]
            baseline.attach_bboxes3d(items,depth,info,min_depth=.2,max_depth=1.,quantile=.05,
                                    depth_scale=.001,ray_cache=old_cache)
            return items
        stages['bbox3d_legacy']=median_ms(old_measure)
    fitter=CudaPlaneFitter();cache=ProjectionRayCache()
    cache.get(info.width,info.height,np.asarray(info.k).reshape(3,3),np.asarray(info.d),info.distortion_model)
    try:
        for mode,backend in [('cpu',None),('cuda',fitter)]:
            def measure():
                items=[dict(bbox=list(b)) for b in boxes]
                attach_bboxes3d(items,depth,info,min_depth=.2,max_depth=1.,quantile=.05,
                               depth_scale=.001,ray_cache=cache,plane_fitter=backend)
                return items
            stages['bbox3d_'+mode]=median_ms(measure)
    finally:fitter.close()
    for mode in ('pipe','shm'):
        tracker=NvDCF(transport=mode);serial=[0]
        try:
            tracker.update(rgb,0,[dict(item) for item in meta['detections']])
            def measure():
                serial[0]+=1;return tracker.update(rgb,serial[0])
            stages['nvdcf_'+mode]=median_ms(measure,15)
        finally:tracker.close()
    overlay=render_display_overlay(rgb,meta['detections'],info)
    stages['image_overlay']=median_ms(lambda: render_display_overlay(rgb,meta['detections'],info),15)
    stages['image_jpeg']=median_ms(lambda: cv2.imencode('.jpg',overlay,[cv2.IMWRITE_JPEG_QUALITY,90]),15)
    if args.models:
        import torch
        from PIL import Image
        from dinov3_cases.inference.encoder import Encoder
        from dinov3_cases.recognition.grounded import encode_object
        from dinov3_cases.runtime import DINO_ENGINE_SINGLE, DINO_ENGINE_BATCH
        from dinov3_cases.inference.sam3_trt import prepare_sam3_pixels, prepare_sam3_tensor
        image=Image.fromarray(rgb);selected=boxes[:4]
        for mode,path in [('single',DINO_ENGINE_SINGLE),('batch',DINO_ENGINE_BATCH)]:
            encoder=Encoder(path)
            def measure():
                output=([encode_object(encoder,image.crop(b),384,16) for b in selected] if mode=='single'
                        else encoder.encode_regions(image,selected));torch.cuda.synchronize();return output
            rows=measure();stages['dino_'+mode]=median_ms(measure,5)
            if mode=='single':reference=[r[2].cpu() for r in rows]
            else:report['descriptor_cosines']=[float(torch.nn.functional.cosine_similarity(a,r[2].cpu(),dim=0)) for a,r in zip(reference,rows)]
            del encoder;torch.cuda.empty_cache()
        for mode in ('cpu','cuda'):
            def measure():
                output=(torch.from_numpy(prepare_sam3_pixels(image)).to('cuda',dtype=torch.float16)
                        if mode=='cpu' else prepare_sam3_tensor(image,'cuda'))
                torch.cuda.synchronize();return output
            stages['sam3_preprocess_'+mode]=median_ms(measure)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    for name,values in stages.items():print(name,round(values['median_ms'],2),'ms',flush=True)


if __name__=='__main__':main()
