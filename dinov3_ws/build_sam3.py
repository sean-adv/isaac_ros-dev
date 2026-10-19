"""Build official SAM3 vision with Torch-TensorRT; bbox head remains PyTorch."""
import sys
from build_models import main

if __name__ == '__main__':
    sys.argv[1:1] = ['--target', 'sam3']
    main()
