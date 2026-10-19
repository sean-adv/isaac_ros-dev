"""Build Cat2Real using official Torch-TensorRT (see build_models.py --help)."""
import sys
from build_models import main

if __name__ == '__main__':
    sys.argv[1:1] = ['--target', 'cat2real']
    main()
