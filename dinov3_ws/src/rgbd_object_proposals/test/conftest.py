import sys
from pathlib import Path

package = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(package), str(package / 'examples'), str(package.parent / 'dinov3_cases')]
