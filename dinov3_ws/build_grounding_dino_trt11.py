"""Build an explicitly typed Grounding DINO ONNX using TensorRT 11.1."""

import argparse
import hashlib
import json
from pathlib import Path
import os
import tensorrt as trt


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--onnx", type=Path, required=True)
    args.add_argument("--output", type=Path, required=True)
    args.add_argument("--precision", default="Mixed FP16/FP32 (NVIDIA AutoCast)")
    options = args.parse_args()
    if trt.__version__ != "11.1.0.106":
        raise RuntimeError(f"Expected TensorRT 11.1.0.106, found {trt.__version__}")
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(options.onnx.resolve())):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.builder_optimization_level = 3
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        if any(d <= 0 for d in tensor.shape):
            raise ValueError(f"Expected fixed input: {tensor.name}: {tensor.shape}")
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")
    tmp = options.output.with_suffix(".engine.tmp")
    tmp.write_bytes(bytes(plan))
    os.replace(tmp, options.output)
    root = Path(__file__).resolve().parent
    metadata = json.loads(
        (root / "artifacts/tensorrt/grounding_dino_wingdzero_fp16.json").read_text()
    )
    metadata.update(
        tensorrt=trt.__version__,
        precision=options.precision,
        source_onnx=str(options.onnx.resolve()),
        engine_sha256=hashlib.sha256(bytes(plan)).hexdigest(),
    )
    options.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Built", options.output, flush=True)


if __name__ == "__main__":
    main()
