"""SAM3 W8A8 TensorRT 10 helpers adapted from DART."""
import numpy as np, tensorrt as trt, ctypes, os, json, time
from .cuda_alloc import DevBuf, Stream, memcpy_async, D2D  # thin cudart wrapper (see cuda_alloc.py)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
NP_DTYPE = {trt.float32: np.float32, trt.float16: np.float16, trt.int8: np.int8, trt.int32: np.int32, trt.bool: np.bool_}

def load_engine(path, plugin=None):
    if plugin: ctypes.CDLL(plugin, mode=ctypes.RTLD_GLOBAL)
    rt = trt.Runtime(TRT_LOGGER)
    with open(path, "rb") as f: eng = rt.deserialize_cuda_engine(f.read())
    assert eng is not None, path
    return eng

class DeviceTensor:
    """Borrow a persistent Runner output until that Runner is executed again."""
    def __init__(self, buffer, shape, dtype):
        self.buffer, self.shape, self.dtype = buffer, shape, np.dtype(dtype)

    @property
    def __cuda_array_interface__(self):
        return dict(shape=self.shape, typestr=self.dtype.str,
                    data=(self.buffer.ptr, False), version=3, strides=None)


class Runner:
    """Synchronous single-stream executor. Feeds numpy inputs, returns numpy outputs."""
    def __init__(self, engine, outputs=None):
        self.engine = engine; self.ctx = engine.create_execution_context(); self.stream = Stream()
        self.names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self.inputs = [n for n in self.names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in self.names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        if outputs is not None: self.outputs = [n for n in self.outputs if n in outputs]
        # aliases: 'images' -> first input; fpn_0/1/2 -> the three largest 4-D outputs by width (desc)
        self.alias = {}
        if self.inputs and "images" not in self.inputs: self.alias["images"] = self.inputs[0]
        outs4 = [n for n in self.outputs if len(engine.get_tensor_shape(n)) == 4]
        outs4.sort(key=lambda n: -engine.get_tensor_shape(n)[-1])
        for i, n in enumerate(outs4[:3]):
            if f"fpn_{i}" not in self.outputs: self.alias[f"fpn_{i}"] = n
        self.device_casts = {}
        self.bufs = {}
        for n in self.names:
            shape = tuple(engine.get_tensor_shape(n)); dt = NP_DTYPE[engine.get_tensor_dtype(n)]
            nbytes = int(np.prod(shape)) * np.dtype(dt).itemsize
            self.bufs[n] = (DevBuf(nbytes), shape, dt)
            self.ctx.set_tensor_address(n, self.bufs[n][0].ptr)
    def _res(self, n): return self.alias.get(n, n)
    def __call__(self, feeds, want=None, device_outputs=()):
        feeds = {self._res(k): v for k, v in feeds.items()}
        temporaries = []  # Keep GPU dtype conversions alive until this stream completes.
        for n in self.inputs:
            buf, shape, dt = self.bufs[n]
            value = feeds[n]
            if isinstance(value, DeviceTensor):
                if value.shape[1:] != shape[1:] or value.shape[0] != 1:
                    raise ValueError(f"Incompatible device tensor for {n}: {value.shape} -> {shape}")
                # K=1 or K=4 head: repeat the single-image feature on the GPU.
                source_ptr = value.buffer.ptr
                if value.dtype != np.dtype(dt):
                    import torch
                    if n not in self.device_casts:
                        self.device_casts[n] = torch.empty(
                            value.shape, device="cuda",
                            dtype=torch.from_numpy(np.empty((), dtype=dt)).dtype)
                    converted = self.device_casts[n]
                    converted.copy_(torch.as_tensor(value, device="cuda"))
                    torch.cuda.current_stream().synchronize()
                    temporaries.append(converted)
                    source_ptr = converted.data_ptr()
                count = int(np.prod(value.shape)) * np.dtype(dt).itemsize
                if count * shape[0] > buf.nbytes:
                    raise ValueError("Device tensor exceeds head input buffer")
                for index in range(shape[0]):
                    memcpy_async(buf.ptr + index * count, source_ptr, count, D2D, self.stream)
            else:
                x = np.ascontiguousarray(value, dtype=dt)
                if x.shape != shape:
                    raise ValueError(f"Input shape mismatch for {n}: {x.shape} != {shape}")
                buf.upload(x)
        ok = self.ctx.execute_async_v3(self.stream.ptr); assert ok
        self.stream.sync()
        out = {}
        for n in (self.outputs if want is None else want):
            rn = self._res(n); buf, shape, dt = self.bufs[rn]
            out[n] = DeviceTensor(buf, shape, dt) if n in device_outputs else buf.download(shape, dt)
            if rn != n: out[rn] = out[n]
        return out
    def time(self, feeds, warmup=10, iters=50):
        feeds = {self._res(k): v for k, v in feeds.items()}
        for n in self.inputs:
            buf, shape, dt = self.bufs[n]; buf.upload(np.ascontiguousarray(feeds[n], dtype=dt))
        ptrs=[self.bufs[n][0].ptr for n in self.names]
        for _ in range(warmup): self.ctx.execute_async_v3(self.stream.ptr)
        DevBuf.sync(); t=time.perf_counter()
        for _ in range(iters): self.ctx.execute_async_v3(self.stream.ptr)
        DevBuf.sync(); return (time.perf_counter()-t)*1000/iters
