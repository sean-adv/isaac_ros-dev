"""Run DINOv3 matching and aligned-depth surface pose estimation."""
import os
from pathlib import Path
import sys


def main():
    workspace = Path(__file__).resolve().parent
    config = workspace / 'config' / 'dinov3_cases.yaml'
    extra_args = sys.argv[1:]
    if extra_args[:1] == ['--ros-args']:
        extra_args = extra_args[1:]
    os.execv(sys.executable, [sys.executable, '-u', '-m', 'dinov3_cases.node',
                             '--ros-args', '--params-file', str(config), '-p',
                             f'cases_dir:={workspace.parent / "dinov3_cases_data"}',
                             *extra_args])


if __name__ == '__main__':
    main()
