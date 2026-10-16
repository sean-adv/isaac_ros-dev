# Thor 模型環境與轉換簡短教學

入口：`scripts/build_thor_models.sh`，請使用 **Bash**。此腳本依目前工作區的 Thor 成功建置紀錄整理；不會安裝主機 driver、CUDA 或系統 apt 套件。需要目前含 Thor 相容性調整的 `DART/dartf` 與 `dinov3_tensorrt` 原始碼。

| 選項 | Python 環境 | 主要版本 | 模型輸出 |
| --- | --- | --- | --- |
| `dartf` | `DART/.venv` | Python 3.12、Torch 2.14.1/cu130、TRT 10.13.3.9、NumPy 1.26.4 | W8A8 backbone `.plan`、c1/c4 FP16 bbox heads、text ONNX |
| `dinov3` | `dinov3_trt11_venv` | Python 3.12、Torch 2.14.1/cu130、Torch-TRT 2.14.0、TRT 11.1.0.106、NumPy 1.26.4 | ViT-B FP16 TorchScript `.ts`，內含 TRT，輸入 `[1,3,224,224]` |

「WA8A」依現有專案名稱採 **W8A8**：backbone 權重與 activation 都為 INT8。DARTF 此流程為 bbox only。

## 1. 系統前提

在目前的 Isaac ROS／Ubuntu 24.04 ARM64、Thor SM110、CUDA 13 開發容器內執行。先確認：

```bash
nvidia-smi
/usr/local/cuda/bin/nvcc --version
/usr/bin/python3.12 --version
```

DARTF 另外需要系統 TensorRT **10.13.3.9** 的 Python bindings、headers、libraries，以及 C++ 編譯工具。系統套件安裝方法見 [DARTF 重建教學](DART/dartf/README_THOR_REBUILD.zh-TW.md) 第 1 節。TRT 11 只安裝在 DINOv3 venv。

## 2. 選擇操作

從工作區根目錄執行，無參數會依序詢問「哪個模型」與「建環境／轉模型／兩者」：

```bash
bash scripts/build_thor_models.sh
```

也可直接指定，適合自動化：

```bash
# 只建置兩套環境，不轉模型
bash scripts/build_thor_models.sh --target all --action env

# 檢視預計命令，完全不執行
bash scripts/build_thor_models.sh --target all --action all --dry-run

# 已備妥環境與權重，只轉換指定模型
bash scripts/build_thor_models.sh --target dartf --action convert
bash scripts/build_thor_models.sh --target dinov3 --action convert

# 建環境後接著轉換（權重與校正影像需先備妥）
bash scripts/build_thor_models.sh --target all --action all
```

環境安裝需要網路。已有 venv 時會在原處接續安裝固定版本；DARTF 呼叫原有 setup 的 `--resume`，不搬移現有 venv。DINOv3 使用本機 Thor 紀錄的版本，不套用上游 `pyproject.toml` 的 Torch 2.12 相依設定；腳本以 `PYTHONPATH` 載入其原始碼，不需要 `uv sync`。可用 `DINO_VENV=/絕對路徑/venv` 指定另一個獨立環境。

## 3. 首次準備模型與校正資料

已存在下列檔案即可略過本節：

- DARTF：`DART/models/sam3.pt`、`DART/models/hf-sam3/` 的設定與權重，以及 `DART/dartf/calib/calib_ids.json` 指定的 COCO 校正圖。
- DINOv3：`DART/models/dinov3-vitb16/` 的設定、影像處理設定與 safetensors 權重。

先取得 Hugging Face 的 `facebook/sam3` 與 `facebook/dinov3-vitb16-pretrain-lvd1689m` 存取權。建好環境後，在終端登入（不要把 token 寫進腳本）：

```bash
DART/.venv/bin/hf auth login
DART/.venv/bin/python - <<'PY'
import json
import urllib.request
from pathlib import Path
from huggingface_hub import snapshot_download

snapshot_download('facebook/sam3', local_dir='DART/models/hf-sam3',
                  allow_patterns=['*.json', '*.safetensors', 'sam3.pt'], token=True)
checkpoint = Path('DART/models/sam3.pt')
if not checkpoint.exists() and not checkpoint.is_symlink():
    checkpoint.symlink_to('hf-sam3/sam3.pt')
assert checkpoint.is_file(), 'Missing official sam3.pt'

out = Path('DART/dartf/data/train2017')
out.mkdir(parents=True, exist_ok=True)
for iid in json.loads(Path('DART/dartf/calib/calib_ids.json').read_text()):
    path = out / (iid + '.jpg')
    if not path.is_file():
        urllib.request.urlretrieve('https://images.cocodataset.org/train2017/' + path.name, path)
PY

dinov3_trt11_venv/bin/hf auth login
dinov3_trt11_venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('facebook/dinov3-vitb16-pretrain-lvd1689m',
                  local_dir='DART/models/dinov3-vitb16',
                  allow_patterns=['*.json', '*.safetensors'], token=True)
PY
```

若只使用 DINOv3，只需建置 `dinov3` 環境並執行該段下載。

## 4. 輸出與續跑

預設每次轉換使用新的 `model_builds/時間-PID/`，保留原本 engines；也可指定固定目錄續跑：

```bash
bash scripts/build_thor_models.sh --target all --action convert --output model_builds/my_build
```

- `dartf/out_bbox/vision_int8.plan`：W8A8 backbone；`dartf/assets_bbox/`：bbox 推論所需 assets。原有 pipeline 會執行 backbone FP32 參考驗證；以完成標記跳過已完成階段。更換權重或設定時請換新輸出目錄。
- `dinov3/vitb16-fp16-trt.ts`、`benchmark.json`、`accuracy.json`：匯出模型、效能與精度報告。會重新載入 `.ts` 與 FP32 比較，`all_pass=false` 時腳本失敗；DINOv3 每次都會重建並覆寫指定目錄的同名檔。

DARTF 每次轉換會編譯共用 `DART/dartf/plugins/lq_plugins.so`，CUTLASS 不存在時下載 v3.5.1，並寫入原有 `dartf/logs/`；不要同時執行多個 DARTF 建置。預設 `ARCH=110`、`NVCC=/usr/local/cuda/bin/nvcc`。自訂 DARTF 模型／校正路徑請用絕對路徑；DINOv3 可用 `DINOV3_MODEL=/path/to/model`。

DARTF 額外的 heads/text 與實際推論驗證可依 [既有教學](DART/dartf/README_THOR_REBUILD.zh-TW.md) 第 5、6 節執行，並將 assets/ONNX 路徑替換成本次輸出。TensorRT engine 應在預定使用的 GPU 與版本上建置。
