"""One in-flight RGB frame in shared memory; metadata over a private socketpair."""

import json
import os
from pathlib import Path
from multiprocessing import shared_memory, resource_tracker
import socket
import struct
import subprocess
import threading
import time

import numpy as np

MAX_MESSAGE = 1 << 20
MAX_IMAGE_BYTES = 128 << 20


def _receive(sock):
    def exact(size):
        chunks = bytearray()
        while len(chunks) < size:
            part = sock.recv(size - len(chunks))
            if not part:
                raise ConnectionError("Grounding worker disconnected")
            chunks.extend(part)
        return bytes(chunks)

    size = struct.unpack("!I", exact(4))[0]
    if size > MAX_MESSAGE:
        raise ValueError("Grounding IPC message exceeds limit")
    return json.loads(exact(size))


def _send(sock, value):
    raw = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    if len(raw) > MAX_MESSAGE:
        raise ValueError("Grounding IPC message exceeds limit")
    sock.sendall(struct.pack("!I", len(raw)) + raw)


class SharedMemoryGroundingDetector:
    """Own a legacy-environment subprocess and its reusable shared RGB buffer."""

    def __init__(
        self,
        engine_path,
        device,
        *,
        python,
        timeout=60.0,
        worker_module="dinov3_cases.grounding_ipc",
    ):
        self.lock = threading.RLock()
        self.shm = None
        self.closed = False
        self.sequence = 0
        self.last_metrics = {}
        self.device = device
        self.timeout = float(timeout)
        if not np.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Grounding IPC timeout must be positive")
        self.sock, child = socket.socketpair()
        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        venv = Path(python).absolute().parent.parent
        env["VIRTUAL_ENV"] = str(venv)
        env["PATH"] = f"{venv}/bin:/usr/local/cuda/bin:/usr/bin:/bin"
        # Load TensorRT libraries from the worker environment.
        env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu"
        env["PYTHONPATH"] = os.pathsep.join(
            p
            for p in env.get("PYTHONPATH", "").split(os.pathsep)
            if p and not any(v in p for v in ("dinov3_trt11_venv", "trtmc_venv"))
        )
        env["OMP_NUM_THREADS"] = env["OPENBLAS_NUM_THREADS"] = "8"
        try:
            self.process = subprocess.Popen(
                [str(python), "-u", "-m", worker_module, "--fd", str(child.fileno())],
                pass_fds=(child.fileno(),),
                env=env,
            )
        except BaseException:
            self.sock.close()
            raise
        finally:
            child.close()
        try:
            self.sock.settimeout(max(120.0, self.timeout))
            _send(self.sock, dict(device=device, engine_path=str(engine_path)))
            ready = _receive(self.sock)
            if not ready.get("ok"):
                raise RuntimeError(ready.get("error", "Grounding worker startup failed"))
            self.worker_info = ready["worker"]
            self.offline_prompts = ready.get("offline_prompts", [])
            self.precision = ready.get("precision")
            self.sock.settimeout(self.timeout)
        except BaseException:
            self.close()
            raise

    def _request(self, command, **fields):
        if self.closed:
            raise RuntimeError("Grounding worker is closed; restart the pipeline")
        self.sequence += 1
        request = dict(command=command, frame_id=self.sequence, **fields)
        sent = time.monotonic_ns()
        try:
            _send(self.sock, request)
            reply = _receive(self.sock)
            completed = time.monotonic_ns()
            if reply.get("frame_id") != self.sequence:
                raise RuntimeError("Grounding IPC frame ID mismatch")
        except (OSError, ConnectionError, ValueError, RuntimeError) as error:
            self.close()
            raise RuntimeError(f"Grounding IPC failed: {error}") from error
        if not reply.get("ok"):
            if reply.get("error_type") == "ValueError":
                raise ValueError(reply["error"])
            raise RuntimeError(reply.get("error", "Grounding worker failed"))
        if command == "detect":
            worker_ms = reply["detect_ms"]
            self.last_metrics = dict(
                frame_id=self.sequence,
                worker_detect_ms=worker_ms,
                worker_image_read_ms=reply["image_read_ms"],
                ipc_roundtrip_ms=(completed - sent) / 1e6,
                ipc_overhead_ms=max(0.0, (completed - sent) / 1e6 - worker_ms),
                send_to_worker_ms=max(0.0, (reply["received_ns"] - sent) / 1e6),
            )
        return reply

    def validate_prompt(self, prompt):
        with self.lock:
            self._request("validate_prompt", prompt=prompt)

    def detect(self, image, prompt, box_threshold, nms_iou, limit,
               *, min_width=0, min_height=0, min_area=0):
        with self.lock:
            prepare_start = time.monotonic_ns()
            pixels = np.asarray(
                image if image.mode == "RGB" else image.convert("RGB"), dtype=np.uint8
            )
            prepare_ms = (time.monotonic_ns() - prepare_start) / 1e6
            size = pixels.nbytes
            if size > MAX_IMAGE_BYTES:
                raise ValueError("RGB frame exceeds 128 MiB shared-memory limit")
            if self.closed:
                raise RuntimeError("Grounding worker is closed; restart the pipeline")
            if self.shm is None or self.shm.size < size:
                previous = self.shm
                self.shm = shared_memory.SharedMemory(create=True, size=size)
                if previous is not None:
                    previous.close()
                    previous.unlink()
            start = time.monotonic_ns()
            np.ndarray(pixels.shape, dtype=np.uint8, buffer=self.shm.buf)[:] = pixels
            copy_ms = (time.monotonic_ns() - start) / 1e6
            reply = self._request(
                "detect",
                shm=self.shm.name,
                shape=list(pixels.shape),
                prompt=prompt,
                box_threshold=box_threshold,
                nms_iou=nms_iou,
                limit=limit,
                min_width=min_width,
                min_height=min_height,
                min_area=min_area,
            )
            self.last_metrics["shared_copy_ms"] = copy_ms
            self.last_metrics["image_prepare_ms"] = prepare_ms
            self.last_metrics["image_bytes"] = size
            self.last_metrics["transport_total_ms"] = copy_ms + self.last_metrics["ipc_overhead_ms"]
            self.last_metrics["input_and_transport_ms"] = (
                prepare_ms + self.last_metrics["transport_total_ms"]
            )
            return reply["proposals"]

    def close(self):
        # Do not acquire the RPC lock: shutdown must interrupt a pending receive.
        self.closed = True
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        with self.lock:
            if self.shm is not None:
                self.shm.close()
                try:
                    self.shm.unlink()
                except FileNotFoundError:
                    pass
                self.shm = None


def worker_main(fd, detector_factory=None, version=None):
    from PIL import Image

    sock = socket.socket(fileno=fd)
    attached = None
    try:
        config = _receive(sock)
        try:
            if detector_factory is None:
                import tensorrt as trt
                import torch
                from .grounded import OmDetDetector

                if trt.__version__ != "11.1.0.106":
                    raise RuntimeError(
                        f"Expected TensorRT 11.1.0.106, got {trt.__version__}"
                    )
                detector_factory = OmDetDetector
                version = dict(tensorrt=trt.__version__, torch=torch.__version__, pid=os.getpid())
            detector = detector_factory(**config)
            _send(
                sock, dict(
                    ok=True, worker=version,
                    offline_prompts=getattr(detector, "offline_prompts", []),
                    precision=getattr(detector, "precision", None),
                )
            )
        except Exception as error:
            _send(sock, dict(ok=False, error=f"{type(error).__name__}: {error}"))
            return
        while True:
            request = _receive(sock)
            received = time.monotonic_ns()
            try:
                if request["command"] == "validate_prompt":
                    detector.validate_prompt(request["prompt"])
                    result = {}
                elif request["command"] == "detect":
                    shape = tuple(request["shape"])
                    if (
                        len(shape) != 3
                        or shape[2] != 3
                        or any(type(n) is not int or n <= 0 for n in shape)
                        or int(np.prod(shape)) > MAX_IMAGE_BYTES
                    ):
                        raise ValueError("Invalid shared RGB image shape")
                    if attached is None or attached.name != request["shm"]:
                        if attached is not None:
                            attached.close()
                        attached = shared_memory.SharedMemory(name=request["shm"])
                        # This independent interpreter must not unlink parent-owned SHM at exit.
                        resource_tracker.unregister(attached._name, "shared_memory")
                    if int(np.prod(shape)) > attached.size:
                        raise ValueError("Shared RGB shape exceeds allocated memory")
                    pixels = np.ndarray(shape, dtype=np.uint8, buffer=attached.buf)
                    # Parent holds its request lock until our response: the buffer
                    # cannot be overwritten while this image is being consumed.
                    image = Image.fromarray(pixels)
                    del pixels
                    read_at = time.monotonic_ns()
                    proposals = detector.detect(
                        image,
                        request["prompt"],
                        request["box_threshold"],
                        request["nms_iou"],
                        request["limit"],
                        min_width=request.get("min_width", 0),
                        min_height=request.get("min_height", 0),
                        min_area=request.get("min_area", 0),
                    )
                    del image
                    result = dict(
                        proposals=proposals,
                        received_ns=received,
                        image_read_ms=(read_at - received) / 1e6,
                        detect_ms=(time.monotonic_ns() - read_at) / 1e6,
                    )
                else:
                    raise ValueError("Unknown Grounding IPC command")
                _send(sock, dict(ok=True, frame_id=request["frame_id"], **result))
            except Exception as error:
                _send(
                    sock,
                    dict(
                        ok=False,
                        frame_id=request.get("frame_id"),
                        error_type=type(error).__name__,
                        error=str(error),
                    ),
                )
    except (ConnectionError, BrokenPipeError):
        pass
    finally:
        if attached is not None:
            attached.close()
        sock.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", type=int, required=True)
    worker_main(parser.parse_args().fd)
