"""Compare saved pre-change 3D code, ROI conversion, and cached rays on identical depth frames."""
import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import time

import cv2
import numpy as np
from dinov3_cases.depth_bbox import ProjectionRayCache, attach_bboxes3d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True, help='Saved pre-change depth_bbox.py')
    parser.add_argument('--samples', type=Path, required=True, help='Directory containing frames.npz and calibration.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    if args.rounds < 1 or args.repeats < 1:
        parser.error('rounds and repeats must be positive')
    spec = importlib.util.spec_from_file_location('baseline_depth_bbox', args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    meta = json.loads((args.samples / 'calibration.json').read_text())
    info_values = meta['info']
    info = SimpleNamespace(**(info_values | {'roi': SimpleNamespace(**info_values['roi'])}))
    frames = np.load(args.samples / 'frames.npz')['depths']
    scale = {'16UC1': .001, '32FC1': 1.}[meta['encoding']]
    h, w = frames.shape[1:]
    def box(cx, cy, bw, bh):
        return [int(cx-bw/2), int(cy-bh/2), int(cx+bw/2), int(cy+bh/2)]
    cases = {
        'one_300x300': [box(w*.5, h*.5, min(300,w), min(300,h))],
        'one_600x400': [box(w*.5, h*.5, min(600,w), min(400,h))],
        'three_300x300': [box(w*x, h*.5, min(300,w), min(300,h)) for x in (.3,.5,.7)],
    }
    cv2.setNumThreads(1)  # Match production opencv_num_threads.
    reports = {}
    for case, boxes in cases.items():
        cache = ProjectionRayCache()
        def run(mode, raw):
            items = [dict(bbox=b) for b in boxes]
            if mode == 'before':
                depth = raw.astype(np.float64)
                if scale != 1.:
                    depth = depth * scale
                baseline.attach_bboxes3d(items, depth, info)
            else:
                attach_bboxes3d(items, raw, info, depth_scale=scale,
                               ray_cache=cache if mode == 'roi_and_rays' else None)
            return items
        start = time.perf_counter()
        cold_result = run('roi_and_rays', frames[0])
        cold_ms = (time.perf_counter()-start)*1000
        max_abs_error = 0.
        statuses = {}
        for raw in frames:
            expected = run('before', raw)
            for item in expected:
                key = item['bbox3d_status']
                statuses[key] = statuses.get(key, 0)+1
            for mode in ('roi_only', 'roi_and_rays'):
                actual = run(mode, raw)
                for old, new in zip(expected, actual):
                    assert old['bbox3d_status'] == new['bbox3d_status']
                    assert old.keys() == new.keys()
                    if 'bbox3d' not in old:
                        continue
                    assert old['bbox3d'].keys() == new['bbox3d'].keys()
                    for key, value in old['bbox3d'].items():
                        if isinstance(value, str):
                            assert value == new['bbox3d'][key]
                        else:
                            a, b = np.asarray(value), np.asarray(new['bbox3d'][key])
                            np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
                            max_abs_error = max(max_abs_error, float(np.max(np.abs(a-b))))
        if not statuses.get('ok'):
            raise RuntimeError(f'{case}: no valid 3D result; choose other sample ROIs')
        timings = {m: [] for m in ('before', 'roi_only', 'roi_and_rays')}
        round_stats = []
        for repeat in range(args.rounds):
            modes = list(timings) if repeat % 2 == 0 else list(reversed(timings))
            per_round = {}
            # Alternate order per input too, reducing drift bias.
            for i in range(len(frames)*args.repeats):
                order = modes if i % 2 == 0 else modes[::-1]
                for mode in order:
                    start = time.perf_counter()
                    run(mode, frames[i % len(frames)])
                    ms = (time.perf_counter()-start)*1000
                    timings[mode].append(ms)
                    per_round.setdefault(mode, []).append(ms)
            round_stats.append({m: float(np.mean(v)) for m,v in per_round.items()})
        summary = {m: dict(samples=len(v), mean_ms=float(np.mean(v)), median_ms=float(np.median(v)),
                           p95_ms=float(np.percentile(v,95))) for m,v in timings.items()}
        report = dict(boxes=boxes, statuses=statuses, max_abs_output_error=max_abs_error,
                      cold_first_call_ms=cold_ms, cache_bytes=cache._rays.nbytes if cache._rays is not None else 0,
                      timings=summary, rounds=round_stats, samples_ms=timings)
        report['speedup'] = summary['before']['mean_ms']/summary['roi_and_rays']['mean_ms']
        reports[case] = report
        print(json.dumps({k:v for k,v in report.items() if k != 'samples_ms'} | {'case':case}), flush=True)
    result = dict(options={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  image_size=[w,h], encoding=meta['encoding'], frames=len(frames), opencv_version=cv2.__version__,
                  opencv_threads=cv2.getNumThreads(), reports=reports,
                  limitation='Replay of real depth frames with fixed test ROIs, single-thread OpenCV. Measures full-frame/ROI conversion plus 3D attach work, not total pipeline FPS. Camera driver stays running; no SAM3/DINO or JPEG load. Cache construction reported separately.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
