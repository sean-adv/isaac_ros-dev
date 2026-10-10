"""Run text-guided object detection and DINOv3 gallery recognition."""

import argparse
import os
from pathlib import Path
import sys

from dinov3_cases.runtime import MODES, DINO_ENGINE, LEGACY_PYTHON


def main():
    parser = argparse.ArgumentParser(
        description="Run Grounding DINO and DINOv3 object recognition."
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="split",
        help="split: separate TRT 10/11 processes; single: both TRT 11 in one node.",
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
            f"grounding_worker_python:={LEGACY_PYTHON}",
            *extra_args,
        ],
    )


if __name__ == "__main__":
    main()
