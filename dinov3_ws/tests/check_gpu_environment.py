"""Report CUDA driver loading and check PyTorch and TensorRT initialization."""
import ctypes
import os
from pathlib import Path
import socket
import sys


def loaded_cuda_libraries():
    maps = Path('/proc/self/maps')
    if maps.exists():
        paths = sorted({line.split()[-1] for line in maps.read_text().splitlines()
                        if '/' in line and any(name in line for name in
                                               ('libcuda.so', 'libcudart.so', 'libnvinfer.so'))})
        for path in paths:
            print('loaded library:', path, flush=True)


def main():
    print('host:', socket.gethostname(), 'uid:', os.getuid(), flush=True)
    print('python:', sys.executable, flush=True)
    print('container marker:', Path('/.dockerenv').exists() or Path('/run/.containerenv').exists(), flush=True)
    print('supplementary groups:', os.getgroups(), flush=True)
    for name in ('/dev/nvmap', '/dev/nvidia0', '/dev/nvidiactl', '/dev/nvidia-uvm', '/proc/device-tree'):
        path = Path(name)
        print(name, 'exists:', path.exists(), 'readable:', os.access(path, os.R_OK), flush=True)
    for name in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'CUDA_VISIBLE_DEVICES',
                 'NVIDIA_VISIBLE_DEVICES', 'NVIDIA_DRIVER_CAPABILITIES'):
        print(name + ':', os.environ.get(name, '<unset>'), flush=True)
    try:
        cuda = ctypes.CDLL('libcuda.so.1')
        code = cuda.cuInit(0)
        label = ctypes.c_char_p()
        cuda.cuGetErrorName(code, ctypes.byref(label))
        print('cuInit:', code, label.value.decode() if label.value else '', flush=True)
        loaded_cuda_libraries()
        if code:
            print('FAIL: CUDA driver initialization failed; check the container driver libraries/runtime.', flush=True)
            return 1
        import torch
        print('torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda, flush=True)
        torch.ones(1, device='cuda')
        torch.cuda.synchronize()
        print('PyTorch CUDA allocation: OK; GPU:', torch.cuda.get_device_name(0), flush=True)
        import tensorrt as trt
        print('TensorRT:', trt.__version__, trt.__file__, flush=True)
        logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(logger, '')
        builder = trt.Builder(logger)
        if builder is None:
            raise RuntimeError('TensorRT Builder returned None')
        print('TensorRT Builder: OK', flush=True)
        loaded_cuda_libraries()
        return 0
    except Exception as exc:
        print('FAIL:', type(exc).__name__, str(exc), flush=True)
        loaded_cuda_libraries()
        return 1


if __name__ == '__main__':
    sys.exit(main())
