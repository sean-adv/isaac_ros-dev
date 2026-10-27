"""Compare OpenCV thread counts with identical fixed-ROI KCF inputs."""
import argparse
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np
from PIL import Image

from dinov3_cases.kcf import KCFProposals
from dinov3_cases.h264_stream import H264Encoder
from dinov3_cases.visualization import render_display_overlay


def cpu_ticks():
    values = [int(v) for v in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
    return sum(values), values[3] + values[4]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--boxes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 2, 4, 14])
    parser.add_argument('--max-roi-side', type=int, default=0, help='Limit each KCF working ROI; 0 disables resizing')
    parser.add_argument('--preview', action='store_true', help='Include shared 960x540 H.264 preview encoding at 30 FPS')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--seconds', type=float, default=3)
    args = parser.parse_args()
    if min(args.threads) < 1 or args.rounds < 1 or args.seconds <= 0:
        parser.error('threads, rounds and seconds must be positive')
    pixels = np.asarray(Image.open(args.image).convert('RGB'))
    boxes = json.loads(args.boxes.read_text())
    if not boxes:
        parser.error('boxes must be nonempty')
    reports = []
    original_threads = cv2.getNumThreads()
    try:
        for repeat in range(args.rounds):
            order = args.threads if repeat % 2 == 0 else list(reversed(args.threads))
            for count in order:
                # Only change OpenCV's global pool while all benchmark workers are stopped.
                cv2.setNumThreads(count)
                modes = ('foreground_plus_replay_and_preview',) if args.preview else ('foreground_only', 'foreground_plus_replay')
                for mode in modes:
                    trackers = [KCFProposals(roi_padding=.5, max_roi_side=args.max_roi_side) for _ in range(2)]
                    for tracker in trackers:
                        tracker.update(pixels, 0, 'threads', 2**63, lambda: boxes)
                        for n in range(1, 11):
                            tracker.update(pixels, n, 'threads', 2**63, lambda: boxes)
                    stopped, started = threading.Event(), threading.Event()
                    errors, replay_times = [], []

                    def replay():
                        try:
                            started.set()
                            n = 11
                            while not stopped.is_set():
                                start = time.perf_counter()
                                items, source = trackers[1].update(pixels, n, 'threads', 2**63, lambda: [])
                                replay_times.append((time.perf_counter()-start)*1000)
                                if source != 'kcf' or len(items) != len(boxes):
                                    raise RuntimeError('Background tracking lost')
                                n += 1
                        except Exception as error:
                            errors.append(repr(error))

                    def preview():
                        try:
                            encoder = H264Encoder()
                            deadline = time.perf_counter()
                            while not stopped.is_set():
                                bgr = render_display_overlay(pixels, [], mode='2d')
                                encoder.encode(bgr, None, 0)
                                deadline += 1/30
                                stopped.wait(max(0, deadline-time.perf_counter()))
                        except Exception as error:
                            errors.append(repr(error))

                    preview_worker = None
                    worker = None
                    if mode.startswith('foreground_plus_replay'):
                        worker = threading.Thread(target=replay, name='benchmark-replay')
                        worker.start()
                        started.wait(2)
                    if args.preview:
                        preview_worker = threading.Thread(target=preview, name='benchmark-preview')
                        preview_worker.start()
                    values = []
                    before_ticks = cpu_ticks()
                    before_cpu, before = time.process_time(), time.perf_counter()
                    deadline = before + args.seconds
                    try:
                        n = 11
                        while time.perf_counter() < deadline:
                            start = time.perf_counter()
                            items, source = trackers[0].update(pixels, n, 'threads', 2**63, lambda: [])
                            values.append((time.perf_counter()-start)*1000)
                            if source != 'kcf' or len(items) != len(boxes):
                                raise RuntimeError('Foreground tracking lost')
                            n += 1
                        elapsed = time.perf_counter()-before
                        cpu_seconds = time.process_time()-before_cpu
                        after_ticks = cpu_ticks()
                    finally:
                        stopped.set()
                        if worker:
                            worker.join(5)
                            if worker.is_alive():
                                raise RuntimeError('Background worker failed to stop')
                        if preview_worker:
                            preview_worker.join(5)
                            if preview_worker.is_alive():
                                raise RuntimeError('Preview worker failed to stop')
                    if errors:
                        raise RuntimeError(errors)
                    ticks = after_ticks[0]-before_ticks[0]
                    idle = after_ticks[1]-before_ticks[1]
                    row = dict(round=repeat+1, threads=count, mode=mode, objects=len(boxes),
                               max_roi_side=args.max_roi_side, samples=len(values), median_ms=float(np.median(values)),
                               p95_ms=float(np.percentile(values,95)),
                               foreground_fps=len(values)/elapsed,
                               replay_fps=len(replay_times)/elapsed,
                               process_cpu_percent=cpu_seconds/elapsed*100,
                               system_cpu_percent=(1-idle/ticks)*100 if ticks else None,
                               foreground_rows_ms=values)
                    reports.append(row)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(dict(image=str(args.image), boxes=str(args.boxes),
                        original_threads=original_threads, reports=reports,
                        limitation='Static real-camera image and manual boxes; replay is continuously busy stress, not full SAM3/DINOv3 throughput.'), indent=2)+'\n')
                    print(json.dumps({k:v for k,v in row.items() if k!='foreground_rows_ms'}), flush=True)
    finally:
        cv2.setNumThreads(original_threads)


if __name__ == '__main__':
    main()
