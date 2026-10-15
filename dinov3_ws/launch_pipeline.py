"""Run text-guided object detection and DINOv3 gallery recognition."""

import argparse
import os
from pathlib import Path
import sys

from dinov3_cases.runtime import MODES, DINO_ENGINE, GROUNDING_PYTHON


def main():
    parser = argparse.ArgumentParser(
        description="Run SAM3 W8A8 and DINOv3 object recognition."
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="split",
        help="split: SAM3 TRT10 worker in DART/.venv + DINOv3 TRT11 node in dinov3_trt11_venv.",
    )
    options, extra_args = parser.parse_known_args()
    workspace = Path(__file__).resolve().parent
    config = workspace / "config" / "dinov3_cases.yaml"
    if extra_args[:1] == ["--ros-args"]:
        extra_args = extra_args[1:]
    execution, grounding_engine = MODES[options.mode]
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
            f'cases_dir:={workspace.parent / "dinov3_cases_data"}',
            "-p",
            f"engine_path:={DINO_ENGINE}",
            "-p",
            f"grounding_engine_path:={grounding_engine}",
            "-p",
            f"grounding_execution:={execution}",
            "-p",
            f"grounding_worker_python:={GROUNDING_PYTHON}",
            *extra_args,
        ],
    )


if __name__ == "__main__":
    main()
