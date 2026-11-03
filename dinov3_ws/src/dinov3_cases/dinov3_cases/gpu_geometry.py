"""CUDA geometry ABI, independent of the TensorRT and DeepStream processes."""
import ctypes as C
from pathlib import Path
import threading
import numpy as np
from .runtime import PROJECT

LIBRARY = PROJECT / 'artifacts/cuda/libgpu_geometry.so'
if not LIBRARY.is_file():
    LIBRARY = Path('/usr/local/lib/libgpu_geometry.so')
_library = None
_lock = threading.Lock()


def library():
    global _library
    with _lock:
        if _library is None:
            if not LIBRARY.is_file():
                raise RuntimeError('Missing CUDA kernels; run bash dinov3_ws/native/build.sh gpu_geometry')
            lib = C.CDLL(str(LIBRARY))
            lib.geometry_error.restype = C.c_char_p
            lib.plane_create.restype = C.c_void_p
            lib.plane_destroy.argtypes = [C.c_void_p]
            lib.plane_fit.argtypes = [C.c_void_p,C.c_void_p,C.c_int,C.c_float,C.c_int,C.c_uint,C.c_void_p]
            lib.prepare_rgb_rois.argtypes = [C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_void_p,C.c_int,C.c_int,C.c_void_p,C.c_void_p]
            _library = lib
        return _library


class CudaPlaneFitter:
    """Worker-owned stream and reusable buffers; returns refined CPU coefficients."""
    backend = 'cuda_ransac'

    def __init__(self, iterations=128):
        if type(iterations) is not int or not 1 <= iterations <= 1024:
            raise ValueError("RANSAC iterations must be in [1, 1024]")
        self.lib = library()
        self.handle = self.lib.plane_create()
        if not self.handle:
            raise RuntimeError(self.lib.geometry_error().decode())
        self.iterations = iterations
        self.lock = threading.Lock()

    def fit(self, points, threshold):
        points = np.asarray(points,dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all():
            raise ValueError("Expected finite Nx3 points with N >= 3")
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("Invalid plane threshold")
        xyz = np.ascontiguousarray(points,dtype=np.float32)
        mask = np.empty(len(points),dtype=np.uint8)
        with self.lock:
            if not self.handle:
                raise RuntimeError('CUDA plane fitter is closed')
            result = self.lib.plane_fit(self.handle,xyz.ctypes.data,len(xyz),threshold,
                                       self.iterations,0x12345678,mask.ctypes.data)
            if result:
                raise ValueError('cuda_plane_segmentation_failed:'+self.lib.geometry_error().decode())
        indices = np.flatnonzero(mask)
        if len(indices)<3:
            raise ValueError('degenerate_plane')
        inliers = points[indices]
        center = inliers.mean(axis=0)
        local = inliers-center
        values,axes = np.linalg.eigh(local.T@local/len(local))
        if values[1]<=max(values[2]*1e-10,1e-16):
            raise ValueError('degenerate_plane')
        normal = axes[:,0]
        return np.r_[normal,-normal@center],indices

    def close(self):
        with self.lock:
            if self.handle:
                self.lib.plane_destroy(self.handle)
                self.handle=None

    def __del__(self):
        if getattr(self,'handle',None):
            self.close()


def prepare_rois(rgb, boxes, size=384):
    """One resident RGB image -> a batch of uint8 HWC centered white crops."""
    import torch
    if rgb.device.type!='cuda' or rgb.dtype!=torch.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:
        raise ValueError('Expected CUDA uint8 HWC RGB')
    rgb = rgb.contiguous()
    host_boxes = np.asarray(boxes, dtype=np.int32)
    if (host_boxes.ndim != 2 or host_boxes.shape[1] != 4 or np.any(host_boxes[:, :2] < 0)
            or np.any(host_boxes[:, 2:] <= host_boxes[:, :2])
            or np.any(host_boxes[:, 2] > rgb.shape[1]) or np.any(host_boxes[:, 3] > rgb.shape[0])):
        if host_boxes.size == 0:
            return torch.empty((0,size,size,3),dtype=torch.uint8,device=rgb.device)
        raise ValueError('Invalid ROI bounds')
    boxes = torch.as_tensor(host_boxes,dtype=torch.int32,device=rgb.device).contiguous()
    if boxes.ndim!=2 or boxes.shape[1]!=4:
        raise ValueError('Expected Nx4 boxes')
    if not len(boxes):
        return torch.empty((0,size,size,3),dtype=torch.uint8,device=rgb.device)
    output = torch.empty((len(boxes),size,size,3),dtype=torch.uint8,device=rgb.device)
    with torch.cuda.device(rgb.device):
        lib=library()
        result=lib.prepare_rgb_rois(rgb.data_ptr(),rgb.shape[1],rgb.shape[0],rgb.stride(0),
                                   boxes.data_ptr(),len(boxes),size,output.data_ptr(),
                                   torch.cuda.current_stream(rgb.device).cuda_stream)
        if result:
            raise RuntimeError(lib.geometry_error().decode())
    return output
