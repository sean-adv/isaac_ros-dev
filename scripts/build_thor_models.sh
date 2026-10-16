#!/usr/bin/env bash
# Thor / CUDA 13: separate DARTF TRT10 and DINOv3 TRT11 environments.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DART="$ROOT/DART"
DINO="$ROOT/dinov3_tensorrt"
DINO_VENV=${DINO_VENV:-$ROOT/dinov3_trt11_venv}
PYTHON=${PYTHON:-/usr/bin/python3.12}
TARGET= ACTION= OUTPUT= DRY_RUN=0
usage() {
    cat <<'HELP'
用法：bash scripts/build_thor_models.sh [選項]
無參數：互動選單。
  --target dartf|dinov3|all   選擇模型
  --action env|convert|all   建置環境／轉模型／依序執行兩者
  --output DIR              本次模型輸出根目錄（可續跑；預設建立唯一目錄）
  --dry-run                 只列出命令，不安裝、不下載、不轉換
  -h, --help                顯示說明
環境：DART/.venv (TRT10)、dinov3_trt11_venv (TRT11)。
可設定 PYTHON、DINO_VENV、DINOV3_MODEL、ARCH、NVCC、CUTLASS_DIR，
以及 DARTF 的 DARTF_HF_MODEL、CKPT、CALIB_IMAGES、CALIB_IDS、CHECK_IMG。
HELP
}
die() { echo "錯誤：$*" >&2; exit 2; }
while (($#)); do
    case "$1" in
        --target|--action|--output)
            (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || die "$1 缺少值"
            case "$1" in
                --target) TARGET=$2;;
                --action) ACTION=$2;;
                --output) OUTPUT=$2;;
            esac
            shift 2;;
        --dry-run) DRY_RUN=1; shift;;
        -h|--help) usage; exit 0;;
        *) die "未知選項：$1";;
    esac
done
if [[ -z "$TARGET" || -z "$ACTION" ]]; then
    [[ -t 0 ]] || die '非互動模式必須指定 --target 與 --action；請參閱 --help'
    if [[ -z "$TARGET" ]]; then
        echo '模型：1) DARTF W8A8  2) DINOv3 ViT-B  3) 兩者'
        read -r -p '請選擇 [1-3]：' answer
        case "$answer" in 1) TARGET=dartf;; 2) TARGET=dinov3;; 3) TARGET=all;; *) die '無效選擇';; esac
    fi
    if [[ -z "$ACTION" ]]; then
        echo '操作：1) 建置環境  2) 轉模型  3) 建置環境後轉模型'
        read -r -p '請選擇 [1-3]：' answer
        case "$answer" in 1) ACTION=env;; 2) ACTION=convert;; 3) ACTION=all;; *) die '無效選擇';; esac
    fi
fi
case "$TARGET" in dartf|dinov3|all) ;; *) die 'target 必須為 dartf、dinov3 或 all';; esac
case "$ACTION" in env|convert|all) ;; *) die 'action 必須為 env、convert 或 all';; esac
# Resolve relative paths against the caller's directory before changing directory.
[[ "$DINO_VENV" = /* ]] || DINO_VENV="$PWD/$DINO_VENV"
[[ "$DINO_VENV" != "$DART/.venv" && ! "$DINO_VENV" -ef "$DART/.venv" ]] || die 'DINO_VENV 不可使用 DART/.venv'
if [[ -n "$OUTPUT" && "$OUTPUT" != /* ]]; then OUTPUT="$PWD/$OUTPUT"; fi
OUTPUT=${OUTPUT:-$ROOT/model_builds/$(date -u +%Y%m%dT%H%M%S)-$$}
# Do not inherit ROS/user Python packages or a previously activated venv.
unset PYTHONHOME
export PYTHONPATH= PYTHONNOUSERSITE=1
run() {
    printf '+ '; printf '%q ' "$@"; printf '\n'
    if (( ! DRY_RUN )); then "$@"; fi
}
check_env() {
    local py=$1 kind=$2
    run "$py" - "$kind" <<'PY'
import sys
import torch, tensorrt, numpy
assert sys.version_info[:2] == (3, 12), 'Requires Python 3.12'
assert torch.cuda.is_available(), 'CUDA unavailable'
assert torch.__version__.split('+')[0] == '2.14.1', torch.__version__
assert numpy.__version__ == '1.26.4', numpy.__version__
if sys.argv[1] == 'dartf':
    assert tensorrt.__version__ == '10.13.3.9', tensorrt.__version__
else:
    import torch_tensorrt
    assert torch_tensorrt.__version__.split('+')[0] == '2.14.0', torch_tensorrt.__version__
    assert tensorrt.__version__ == '11.1.0.106', tensorrt.__version__
print('Ready:', sys.executable, torch.__version__, tensorrt.__version__, torch.cuda.get_device_name())
PY
}
setup_dartf() {
    # --resume preserves the existing venv; the original default moves it away.
    run bash "$DART/dartf/scripts/setup_thor_venv.sh" --resume
}
setup_dino() {
    local py="$DINO_VENV/bin/python"
    run "$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Requires Python 3.12"'
    if [[ ! -e "$DINO_VENV" ]]; then
        run "$PYTHON" -m venv "$DINO_VENV"
    elif [[ ! -x "$py" ]]; then
        die "已有目錄不是可用 venv：$DINO_VENV"
    fi
    run "$py" -m pip install pip==26.2.1 setuptools==78.1.0 wheel==0.48.0
    run "$py" -m pip install torch==2.14.1 torchvision==0.29.1 numpy==1.26.4 \
        --index-url https://download.pytorch.org/whl/cu130
    # Do not install upstream pyproject pins (Torch/TRT 2.12) on this Thor stack.
    run "$py" -m pip install torch==2.14.1 torchvision==0.29.1 numpy==1.26.4 \
        torch-tensorrt==2.14.0 tensorrt==11.1.0.106 tensorrt-cu13==11.1.0.106 \
        transformers==5.8.0 huggingface-hub==1.33.0 safetensors==0.8.0 pillow==12.3.0
    check_env "$py" dinov3
    run "$py" - <<'PY'
import subprocess, sys
result = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
message = (result.stdout + result.stderr).strip()
print(message)
known = 'nvidia-cusparselt-cu13 0.8.1 is not supported on this platform'
if result.returncode and message != known:
    raise SystemExit(result.returncode)
if result.returncode:
    print('Known SBSA wheel metadata issue; GPU conversion/validation is still required.')
PY
}
convert_dartf() {
    check_env "$DART/.venv/bin/python" dartf
    # Validate weights/calibration before spending time compiling plugins.
    run "$DART/.venv/bin/python" - "$DART" <<'PY'
import json, os, sys
from pathlib import Path
root = Path(sys.argv[1]); dartf = root / 'dartf'
model = Path(os.environ.get('DARTF_HF_MODEL', root / 'models/hf-sam3'))
ids = Path(os.environ.get('CALIB_IDS', dartf / 'calib/calib_ids.json'))
images = Path(os.environ.get('CALIB_IMAGES', dartf / 'data/train2017'))
paths = [Path(os.environ.get('CKPT', root / 'models/sam3.pt')), model / 'config.json', ids,
         Path(os.environ.get('CHECK_IMG', images / '000000230971.jpg'))]
for path in paths:
    if not path.is_file(): raise SystemExit(f'Missing {path}; see MODEL_BUILD.zh-TW.md')
if not list(model.glob('*.safetensors')):
    raise SystemExit(f'Missing safetensors in {model}')
for iid in json.loads(ids.read_text()):
    path = images / (iid + '.jpg')
    if not path.is_file(): raise SystemExit(f'Missing calibration image: {path}')
PY
    local cutlass=${CUTLASS_DIR:-$DART/dartf/third_party/cutlass}
    [[ "$cutlass" = /* ]] || cutlass="$PWD/$cutlass"
    if [[ ! -d "$cutlass" ]]; then
        run git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git "$cutlass"
    fi
    if (( ! DRY_RUN )); then
        [[ $(git -C "$cutlass" describe --tags --exact-match HEAD) == v3.5.1 ]] || die 'CUTLASS 必須為 v3.5.1'
    fi
    run env PATH="$DART/.venv/bin:$PATH" CUTLASS_DIR="$cutlass" ARCH="${ARCH:-110}" \
        NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}" bash "$DART/dartf/plugins/build.sh"
    run env OUT="$OUTPUT/dartf/out_bbox" ASSETS="$OUTPUT/dartf/assets_bbox" \
        bash "$DART/dartf/scripts/build_bbox.sh"
}
convert_dino() {
    local py="$DINO_VENV/bin/python" model=${DINOV3_MODEL:-$DART/models/dinov3-vitb16}
    [[ "$model" = /* ]] || model="$PWD/$model"
    check_env "$py" dinov3
    if (( ! DRY_RUN )); then
        [[ -f "$model/config.json" ]] || die "缺少本機 DINOv3 模型：$model，請參閱 MODEL_BUILD.zh-TW.md"
    fi
    run mkdir -p "$OUTPUT/dinov3"
    run env PYTHONPATH="$DINO/src" "$py" -m dinov3_trt.cli \
        --model-id "$model" --local-files-only --dtype fp16 --height 224 --width 224 \
        --require-full-compilation --warmup 20 --iterations 100 \
        --output "$OUTPUT/dinov3/vitb16-fp16-trt.ts" --report "$OUTPUT/dinov3/benchmark.json"
    run env PYTHONPATH="$DINO/src" "$py" "$DINO/scripts/validate_thor_vitb.py" \
        --model "$model" --artifact "$OUTPUT/dinov3/vitb16-fp16-trt.ts" \
        --report "$OUTPUT/dinov3/accuracy.json"
    # The existing validator writes all_pass but does not set a failure exit code.
    run "$py" - "$OUTPUT/dinov3/accuracy.json" <<'PY'
import json, sys
assert json.load(open(sys.argv[1]))['all_pass'], 'DINOv3 accuracy validation failed'
PY
}
for target in dartf dinov3; do
    [[ "$TARGET" == all || "$TARGET" == "$target" ]] || continue
    if [[ "$target" == dartf ]]; then
        [[ -f "$DART/dartf/scripts/build_bbox.sh" ]] || die "缺少 DARTF 原始碼：$DART"
        [[ "$ACTION" == convert ]] || setup_dartf
        [[ "$ACTION" == env ]] || convert_dartf
    else
        [[ -f "$DINO/src/dinov3_trt/cli.py" ]] || die "缺少 DINOv3 原始碼：$DINO"
        [[ "$ACTION" == convert ]] || setup_dino
        [[ "$ACTION" == env ]] || convert_dino
    fi
done
if [[ "$ACTION" != env ]]; then echo "模型輸出：$OUTPUT"; fi
echo '完成。'
