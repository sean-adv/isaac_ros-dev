"""Run DINOv3 matching and aligned-depth surface pose estimation."""
import os
from pathlib import Path
import sys


def main():
    workspace = Path(__file__).resolve().parent
    os.execv(sys.executable, [sys.executable, '-u', '-m', 'dinov3_cases.node',
                             '--ros-args', '-p',
                             f'cases_dir:={workspace.parent / "dinov3_cases_data"}',
                             *sys.argv[1:]])


if __name__ == '__main__':
    main()
