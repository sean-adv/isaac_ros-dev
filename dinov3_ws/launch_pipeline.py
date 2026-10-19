"""Run text-guided object detection and DINOv3 gallery recognition."""

import argparse
import os
from pathlib import Path
import sys

from dinov3_cases.runtime import GROUNDING_ENGINE, DINO_ENGINE, GROUNDING_PYTHON, CASES_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Run official SAM3 and Cat2Real object recognition."
    )
    if any(arg == "--mode" or arg.startswith("--mode=") for arg in sys.argv[1:]):
        parser.error("--mode has been removed; SAM3 and Cat2Real use the official Torch-TensorRT runtime")
    _, extra_args = parser.parse_known_args()
    workspace = Path(__file__).resolve().parent
    config = workspace / "config" / "dinov3_cases.yaml"
    if extra_args[:1] == ["--ros-args"]:
        extra_args = extra_args[1:]
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-u",
            "-m",
            "dinov3_cases.node",
            "--ros-args",
            "--params-file",
            str(config),
            "-p",
            f"cases_dir:={CASES_DIR}",
            "-p",
            f"engine_path:={DINO_ENGINE}",
            "-p",
            f"grounding_engine_path:={GROUNDING_ENGINE}",
            "-p",
            f"grounding_worker_python:={GROUNDING_PYTHON}",
            *extra_args,
        ],
    )


if __name__ == "__main__":
    main()
