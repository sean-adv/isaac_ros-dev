"""Run text-guided object detection and DINOv3 gallery recognition."""
import argparse
import os
from pathlib import Path
import sys

from dinov3_cases.trt_backend import DEFAULT_VARIANT, MODEL_VARIANTS


def main():
    parser = argparse.ArgumentParser(description='Run Grounding DINO and DINOv3 object recognition.')
    parser.add_argument('--model', choices=sorted(MODEL_VARIANTS), default=DEFAULT_VARIANT,
                        help='DINOv3 TensorRT model variant (default: %(default)s).')
    options, extra_args = parser.parse_known_args()
    workspace = Path(__file__).resolve().parent
    config = workspace / 'config' / 'dinov3_cases.yaml'
    if extra_args[:1] == ['--ros-args']:
        extra_args = extra_args[1:]
    model = MODEL_VARIANTS[options.model]
    os.execv(sys.executable, [sys.executable, '-u', '-m', 'dinov3_cases.node',
                             '--ros-args', '--params-file', str(config), '-p',
                             f'cases_dir:={workspace.parent / "dinov3_cases_data"}',
                             '-p', f'model_id:={model["model_id"]}',
                             '-p', f'engine_path:={model["engine_path"]}',
                             *extra_args])


if __name__ == '__main__':
    main()
