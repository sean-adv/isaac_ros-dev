"""Sample a running pipeline and its MJPEG output without loading models.

Receive times describe this HTTP client, not camera-to-browser latency.
Idle static sources may repeat cached JPEGs; received frames are not unique RGB FPS.
"""
import argparse
import json
from pathlib import Path
import threading
import time
import urllib.request

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8090')
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('--seconds must be positive')
    base = args.url.rstrip('/')
    samples, frames, events, errors = [], [], {}, []
    stop = threading.Event()

    def receive():
        try:
            with urllib.request.urlopen(base+'/image_annotated.mjpg', timeout=5) as response:
                while not stop.is_set():
                    line = response.readline()
                    if not line:
                        break
                    if line.strip() != b'--frame':
                        continue
                    headers = {}
                    while True:
                        line = response.readline()
                        if not line or line == b'\r\n':
                            break
                        key, value = line.decode('ascii').split(':', 1)
                        headers[key.lower()] = value.strip()
                    length = int(headers['content-length'])
                    jpeg = response.read(length)
                    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError('Invalid MJPEG image')
                    frames.append(dict(at=time.monotonic(), width=image.shape[1],
                                       height=image.shape[0], bytes=len(jpeg)))
        except Exception as error:
            errors.append(str(error))

    worker = threading.Thread(target=receive, daemon=True)
    start = time.monotonic()
    worker.start()
    try:
        while time.monotonic()-start < args.seconds:
            before = time.monotonic()
            with urllib.request.urlopen(base+'/api/status', timeout=5) as response:
                status = json.load(response)
            samples.append(dict(at=time.monotonic(), http_ms=(time.monotonic()-before)*1000,
                                status=status))
            for event in status.get('sam3_inference_events', {}).get('events', []):
                events[event['sequence']] = event
            stop.wait(max(0, .02-(time.monotonic()-before)))
    finally:
        stop.set()
        worker.join(timeout=6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(start=start, end=time.monotonic(), samples=samples,
                                         image_frames=frames, image_errors=errors,
                                         sam3_events=list(events.values())), indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), polls=len(samples), image_frames=len(frames),
                          image_errors=errors)))


if __name__ == '__main__':
    main()
