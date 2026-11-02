"""Native GPU display/encode process with small geometry metadata over the pipe."""
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
import os,select,struct,subprocess,time
import cv2
import numpy as np
from .runtime import PROJECT
from .depth_bbox import bbox_vertices, project_box_points
from .h264_stream import H264Encoder, AccessUnit

BRIDGE=PROJECT/'artifacts/video/video_gpu_bridge'
if not BRIDGE.is_file():BRIDGE=Path('/usr/local/bin/dinov3-video-gpu-bridge')


@dataclass
class GpuDisplayFrame:
    rgb: np.ndarray
    lines: list
    texts: list


def prepare_gpu_display_frame(rgb, detections, info=None, mode='3d'):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError('Expected uint8 RGB image')
    if mode not in ('2d','3d'):
        raise ValueError('Invalid bbox visualization mode.')
    height,width=rgb.shape[:2];scale=np.array([960/width,540/height])
    lines,texts=[],[]
    edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for item in detections:
        if not item.get('bbox_visible',True):continue
        color=(255,190,0) if item['label']=='unknown' else (80,220,60)
        if mode=='3d':
            if 'bbox3d' not in item or info is None:continue
            points=bbox_vertices(item['bbox3d'])
            if not np.isfinite(points).all() or np.any(points[:,2]<=1e-6):continue
            try:projected=project_box_points(points,info)*scale
            except (ValueError,cv2.error):continue
            if not np.isfinite(projected).all() or abs(projected).max()>1e7:continue
            pixels=np.rint(projected).astype(int)
            pairs=edges
        else:
            x0,y0,x1,y1=item['bbox'];lower=np.rint(np.array([x0,y0])*scale).astype(int)
            upper=np.rint(np.array([x1-1,y1-1])*scale).astype(int)
            pixels=np.array([lower,[upper[0],lower[1]],upper,[lower[0],upper[1]]]);pairs=edges[:4]
        drawn=False
        for a,b in pairs:
            visible,p0,p1=cv2.clipLine((0,0,960,540),tuple(pixels[a]),tuple(pixels[b]))
            if visible:
                lines.append((*p0,*p1,*(c/255 for c in color)));drawn=True
        if not drawn:continue
        label=item['detection_label'] if item['label']=='unknown' else f"{item['label']} | {item['detection_label']}"
        x,y=np.rint(np.asarray(item['bbox'][:2])*scale).astype(int)
        texts.append((max(4,min(900,int(x))),max(0,min(520,int(y)-24)),label,color))
    # Own the pixels: the shared input lease can be recycled after rendering.
    return GpuDisplayFrame(np.array(rgb,copy=True,order='C'),lines,texts)


class GpuH264Encoder:
    gpu_render=True

    def __init__(self,codec='auto',fps=30,bitrate=2_000_000):
        self.fallback=H264Encoder(codec,fps,bitrate)
        self.fps,self.bitrate=fps,bitrate
        self.process=None;self.input_shm=None;self.last_pts=-1;self.last_backend=None
        if not BRIDGE.is_file():
            raise RuntimeError('Missing GPU video bridge; run bash dinov3_ws/native/build_video_gpu.sh')

    @property
    def codec(self):
        return 'deepstream_gpu_nvenc' if self.last_backend == 'gpu' else self.fallback.codec

    def _read(self,count):
        data=bytearray();end=time.monotonic()+10
        while len(data)<count:
            timeout=end-time.monotonic()
            if timeout<=0 or not select.select([self.process.stdout],[],[],timeout)[0]:
                raise TimeoutError('GPU video bridge timed out')
            chunk=os.read(self.process.stdout.fileno(),count-len(data))
            if not chunk:raise RuntimeError('GPU video bridge exited; see stderr')
            data.extend(chunk)
        return bytes(data)

    def _open(self):
        env=os.environ.copy()
        env['LD_LIBRARY_PATH']='/opt/nvidia/deepstream/deepstream-9.1/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu'
        env.pop('LD_PRELOAD',None)
        self.input_shm=shared_memory.SharedMemory(create=True,size=4096*4096*3)
        self.process=subprocess.Popen([str(BRIDGE),'/'+self.input_shm.name,str(self.fps),str(self.bitrate)],
                                      stdin=subprocess.PIPE,stdout=subprocess.PIPE,env=env,bufsize=0)
        if struct.unpack('<I',self._read(4))[0]!=0x47564944:
            raise RuntimeError('Invalid GPU video handshake')

    def encode(self,pixels,header,generation,force_keyframe=False):
        if not isinstance(pixels,GpuDisplayFrame):
            if self.last_backend == 'gpu':
                self.fallback.reset()
                force_keyframe = True
            self.fallback.last_pts = max(self.fallback.last_pts,self.last_pts)
            unit = self.fallback.encode(pixels,header,generation,force_keyframe)
            self.last_pts,self.last_backend = unit.pts,'cpu'
            return unit
        if self.last_backend == 'cpu':
            force_keyframe = True
        try:
            if self.process is None:self._open()
            rgb=pixels.rgb
            if rgb.nbytes>self.input_shm.size:raise ValueError('Video input exceeds shared buffer')
            np.copyto(np.ndarray(rgb.shape,dtype=np.uint8,buffer=self.input_shm.buf),rgb)
            pts=max(self.last_pts+1,round(time.monotonic()*90000))
            data=bytearray(struct.pack('<7I',rgb.shape[1],rgb.shape[0],len(pixels.lines),len(pixels.texts),
                                       bool(force_keyframe),pts>>32,pts&0xffffffff))
            for line in pixels.lines:data.extend(struct.pack('<7f',*line))
            for x,y,label,color in pixels.texts:
                label=label.encode('utf-8');data.extend(struct.pack('<6I',x,y,len(label),*color));data.extend(label)
            view=memoryview(data)
            while view:
                n=os.write(self.process.stdin.fileno(),view)
                if not n:raise RuntimeError('GPU video pipe closed')
                view=view[n:]
            size,keyframe=struct.unpack('<II',self._read(8))
            if size>16_000_000:raise RuntimeError('Invalid GPU video packet length')
            packet=self._read(size);self.last_pts=pts;self.last_backend="gpu"
            return AccessUnit(packet,pts,bool(keyframe),header,generation)
        except BaseException:
            self.reset();raise

    def reset(self):
        if self.process:
            process,self.process=self.process,None
            process.stdin.close()
            try:process.wait(timeout=3)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            process.stdout.close()
        if self.input_shm:
            self.input_shm.close();self.input_shm.unlink();self.input_shm=None
        self.fallback.reset()
        self.last_backend=None
