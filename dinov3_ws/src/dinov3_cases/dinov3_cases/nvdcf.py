"""DeepStream 9.1 NvDCF in a native process isolated from Torch/TensorRT libraries."""
from pathlib import Path
import os
import select
import struct
import subprocess
import time

import cv2
import numpy as np

from .runtime import PROJECT

DEEPSTREAM = Path('/opt/nvidia/deepstream/deepstream-9.1')
BRIDGE = PROJECT / 'artifacts/nvdcf/nvdcf_bridge'
if not BRIDGE.is_file():
    BRIDGE = Path('/usr/local/bin/dinov3-nvdcf-bridge')
DEFAULT_CONFIG = PROJECT / 'config/nvdcf.yaml'


def validate_detection_interval(value):
    if type(value) is not int or value < 1:
        raise ValueError('detection_interval must be an integer >= 1.')


def check_nvdcf(config=DEFAULT_CONFIG):
    for path in (BRIDGE, Path(config), DEEPSTREAM / 'lib/libnvds_nvmultiobjecttracker.so'):
        if not path.is_file():
            raise ValueError(f'NvDCF dependency missing: {path}. Run native/install_nvdcf.sh.')


class NvDCF:
    """One chronological, whole-frame multi-object tracker context.

    Native IDs index classification metadata within a context; the scheduler
    assigns public IDs when a detector generation catches up to the live frame.
    """
    def __init__(self, config=DEFAULT_CONFIG, width=640, height=384, timeout=30):
        check_nvdcf(config)
        if any(type(v) is not int or v < 32 or v > 4096 or v % 32 for v in (width, height)):
            raise ValueError('nvdcf width/height must be multiples of 32 in [32, 4096]')
        self.width, self.height, self.timeout = width, height, timeout
        self.process = None
        self.metadata = {}
        self.proposals = []
        self.frame_key = None
        self.serial = 0
        self.initialized = False
        env = os.environ.copy()
        # run.sh prioritizes pip TensorRT 11; DeepStream uses system dependencies.
        env['LD_LIBRARY_PATH'] = f'{DEEPSTREAM}/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu'
        env.pop('LD_PRELOAD', None)
        self.process = subprocess.Popen(
            [str(BRIDGE), str(DEEPSTREAM / 'lib/libnvds_nvmultiobjecttracker.so'),
             str(Path(config).resolve()), str(width), str(height)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, bufsize=0,
        )
        os.set_blocking(self.process.stdin.fileno(), False)
        try:
            if struct.unpack('<I', self._read(4))[0] != 0x4e564443:
                raise RuntimeError('Invalid NvDCF bridge handshake')
        except BaseException:
            self.close()
            raise

    def _read(self, size):
        chunks = bytearray()
        deadline = time.monotonic() + self.timeout
        while len(chunks) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise TimeoutError('NvDCF bridge response timed out')
            chunk = os.read(self.process.stdout.fileno(), size - len(chunks))
            if not chunk:
                raise RuntimeError(f'NvDCF bridge exited ({self.process.poll()}); see stderr')
            chunks.extend(chunk)
        return chunks

    def update(self, pixels, frame_key, detections=None):
        if self.initialized and frame_key == self.frame_key:
            return self.proposals
        height, width = pixels.shape[:2]
        sx, sy = self.width / width, self.height / height
        rgba = cv2.cvtColor(cv2.resize(pixels, (self.width, self.height)), cv2.COLOR_RGB2RGBA)
        items = [] if detections is None else detections
        if len(items) > 150:
            raise ValueError('NvDCF supports at most 150 detections per frame')
        self.serial += 1
        data = bytearray(struct.pack('<III', self.serial, detections is not None, len(items)))
        data.extend(np.ascontiguousarray(rgba).tobytes())
        for item in items:
            x0, y0, x1, y1 = item['bbox']
            data.extend(struct.pack('<5f', x0*sx, y0*sy, (x1-x0)*sx, (y1-y0)*sy,
                                    item.get('detection_score', 1.0)))
        try:
            # FileIO writes can be partial on pipes.
            view = memoryview(data)
            deadline = time.monotonic() + self.timeout
            while view:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([], [self.process.stdin], [], remaining)[1]:
                    raise TimeoutError('NvDCF bridge input timed out')
                try:
                    written = os.write(self.process.stdin.fileno(), view[:65536])
                except BlockingIOError:
                    continue
                view = view[written:]
            count = struct.unpack('<I', self._read(4))[0]
            if count > 150:
                raise RuntimeError('Invalid NvDCF output count')
            records = self._read(count * 32)
        except BaseException:
            self.close()
            raise
        output = []
        for native_id, x, y, w, h, confidence, detection in struct.iter_unpack('<Q5fi', records):
            if 0 <= detection < len(items):
                self.metadata[native_id] = dict(items[detection])
            if native_id not in self.metadata:
                continue
            box = [max(0, min(width, round(x/sx))), max(0, min(height, round(y/sy))),
                   max(0, min(width, round((x+w)/sx))), max(0, min(height, round((y+h)/sy)))]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            output.append(dict(self.metadata[native_id], bbox=box,
                               area=(box[2]-box[0])*(box[3]-box[1]),
                               nvdcf_id=native_id, tracker_confidence=confidence,
                               track_status='confirmed' if detection >= 0 else 'coasting',
                               bbox_source='sam3' if detections is not None else 'nvdcf'))
        self.proposals = output
        self.frame_key = frame_key
        self.initialized = True
        return output

    def close(self):
        process, self.process = self.process, None
        if process is not None:
            process.stdin.close()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()
