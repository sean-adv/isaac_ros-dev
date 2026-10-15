# 目前部署：SAM3 DARTF W8A8 + DINOv3

偵測器使用本機已完成轉換並驗證的 `DART/dartf/assets_bbox/vision_int8.plan`，搭配同目錄的 FP16 grounding heads、text ONNX、tokenizer 與位置編碼，以及 `DART/dartf/plugins/lq_plugins.so`。

```bash
./dinov3_ws/run.sh --mode split
```

SAM3 worker 使用 `DART/.venv/bin/python`（TensorRT 10.13.3.9）；DINOv3 主程序使用 `dinov3_trt11_venv/bin/python`（TensorRT 11）。目前僅支援 `split`，兩種版本的引擎分別在各自環境載入。SAM3 使用套件內的 `sam3_runtime`。

SAM3 輸入為 RGB 1008×1008、bilinear resize、`pixel / 127.5 - 1`。Prompt 為 1–4 個逗號分隔的類別名稱，例如 `medicine bottle, medicine box.`，不需 OmDet 的離線 prompt 建置；主頁可直接修改文字。相機擷取頁提供目前 prompt 與四類 preset。單類使用 K=1 head，2–4 類使用 K=4 head，空類別以 empty prompt 補齊。

預設 confidence threshold 為 0.3，分數為 sigmoid(score) × sigmoid(presence)。NMS IoU 設為 1.0（停用），最小 bbox 寬高為 0；每幀最多保留 20 個框供後續 DINOv3 辨識。可在 `config/dinov3_cases.yaml` 調整。DINOv3 辨識與深度設定沿用現有配置。`tracking_enabled: false` 預設停用 ROI tracking，當幀 bbox 立即顯示，不附追蹤 ID；不載入 mask 或 SAM3 tracker engines。

TRT11 建置腳本僅供另外轉建使用，目前推論直接使用原始 TRT10 引擎。驗證 SAM3 TRT10 direct／IPC 結果、DART bbox smoke 偵測基準與 SAM3 bbox crop → DINOv3 特徵：

```bash
LD_LIBRARY_PATH="$PWD/dinov3_trt11_venv/lib/python3.12/site-packages/tensorrt_libs:/usr/local/cuda/lib64" \
PYTHONPATH="$PWD/dinov3_ws/src/dinov3_cases" \
dinov3_trt11_venv/bin/python dinov3_ws/tests/check_sam3_trt_pipeline.py
```

驗證結果寫入 `DART/dartf/assets_bbox/pipeline_validation.json`。

本機整合驗證：106 個單元／IPC／rendering 測試通過；SAM3 單類與四類 head 的 direct／shared-memory IPC 結果一致，bbox crop 可輸出 DINOv3 `[196, 384]` 有限特徵。實際 `run.sh` 的 HTTP 上傳、辨識及無追蹤 bbox overlay 已驗證，報告與範例圖位於 `DART/dartf/assets_bbox/app_integration.json`、`app_bbox_smoke.jpg`。

以下為原始建置說明，其中 OmDet 段落僅供舊模型重建參考，已非預設部署。

---

# OmDet + DINOv3 建置與更換 Prompt

本文件整理模型建置、ROS 2／Orbbec 相機設定、啟動方式，以及日常更換 OmDet prompt 的流程。
目前專案使用 Thor／aarch64、Python 3.12、ROS 2 Jazzy 與 TensorRT 11.1.0.106。
開始前需準備平台適用的 CUDA、PyTorch 與 torchvision；依賴清單不會替你選擇平台專用的 GPU wheel。

除相機編譯步驟外，以下指令都在工作區根目錄執行：

```bash
cd /workspaces/isaac_ros-dev
```

## 1. 下載原始碼

以下 clone 指令只在對應目錄尚不存在時執行。`OmDet` 大小寫需與建置腳本的預設路徑一致。

```bash
git clone https://github.com/om-ai-lab/OmDet.git OmDet
git clone https://github.com/facebookresearch/detectron2.git detectron2
git clone https://github.com/NVIDIA/TensorRT-Model-Connect.git TensorRT-Model-Connect
```

若需重現既有建置，請保留各原始碼 repo 的 commit，並核對產物 metadata 中的 `source_commit`。
更換 prompt 時，OmDet 原始碼 commit 必須與原 engine 建置版本一致。

## 2. 建立 Python 環境並安裝依賴

```bash
python3 -m venv dinov3_trt11_venv
```

先將平台適用的 PyTorch／torchvision 安裝到此環境，再安裝建置依賴：

```bash
dinov3_trt11_venv/bin/python -m pip install -r dinov3_ws/requirements-build-fp16.txt
# DINOv3 建置與驗證需要 CUDA Python bindings：
dinov3_trt11_venv/bin/python -m pip install cuda-python==13.0.3
```

編譯並安裝 detectron2。以下環境變數只套用到這次安裝指令；`11.0` 為本工作區 Thor 的 CUDA 架構設定。

```bash
PATH="$PWD/dinov3_trt11_venv/bin:$PATH" \
MAX_JOBS=8 \
TORCH_CUDA_ARCH_LIST=11.0 \
dinov3_trt11_venv/bin/python -m pip install --no-build-isolation --no-deps -e ./detectron2
```

## 3. 登入 Hugging Face 與下載 DINOv3 權重

請先取得 DINOv3 模型的 Hugging Face 存取權，再在終端登入：

```bash
dinov3_trt11_venv/bin/hf auth login
```

預先下載指定版本到專案快取：

```bash
dinov3_trt11_venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='facebook/dinov3-vits16-pretrain-lvd1689m',
    revision='114c1379950215c8b35dfcd4e90a5c251dde0d32',
    cache_dir='dinov3_ws/artifacts/huggingface',
)
PY
```

此步驟可用來提前確認下載權限；下一節的 `build_dinov3.sh --download` 也能下載所需權重，並使用既有快取。

## 4. 建置 OmDet 與 DINOv3

使用 Docker 時，先依 [Host 容器設定](../DOCKER_HOST.md) 啟動容器。
NVIDIA runtime 負責注入 host driver；Compose 啟動命令設定 `admin` 的 GPU 裝置群組。
模型建置不需額外的 GPU environment／群組 shell 檔案。
建置前，在容器內以 `admin` 確認 GPU：

```bash
dinov3_trt11_venv/bin/python dinov3_ws/tests/check_gpu_environment.py
```

CUDA、PyTorch allocation 與 TensorRT Builder 檢查需全部成功。
以診斷實際載入的 driver 與測試結果判斷可用性；不需因路徑含 `nvgpu` 就強制切換 OpenRM。

### OmDet

首次建置前，可先編輯 [config/omdet_prompts.json](config/omdet_prompts.json) 設定要使用的 prompt 與 labels。

```bash
./dinov3_ws/build_omdet.sh --download --max-labels 4 --max-prompt-tokens 77
```

此指令下載官方權重、產生離線文字特徵、匯出 ONNX 並建置預設 FP16 engine。
`--max-labels 4` 表示每個 prompt profile 最多 4 個類別；`--max-prompt-tokens 77` 包含 CLIP 起訖標記，單位是 token，不是字元。
產物位於 `dinov3_ws/artifacts/omdet/`。

### DINOv3

以下兩種寫法擇一執行即可。使用預設 ViT-S/16、224 × 224 輸入與 FP16：

```bash
./dinov3_ws/build_dinov3.sh --download
```

或明確指定模型、輸入尺寸與精度：

```bash
./dinov3_ws/build_dinov3.sh --download \
  --model facebook/dinov3-vits16-pretrain-lvd1689m \
  --image-size 224 --precision fp16
```

產物位於 `dinov3_ws/artifacts/dinov3/`。目前腳本的輸入尺寸選項只支援 224。

### 在現有工作區驗證建置

權重已快取時可離線建置，並指定獨立的新輸出目錄，避免覆蓋現有部署產物：

```bash
./dinov3_ws/build_dinov3.sh --offline \
  --model facebook/dinov3-vits16-pretrain-lvd1689m \
  --image-size 224 --precision fp16 \
  --output dinov3_ws/artifacts/my-build-check/dinov3

./dinov3_ws/build_omdet.sh --max-labels 4 --max-prompt-tokens 77 \
  --output dinov3_ws/artifacts/my-build-check/omdet
```

## 5. 安裝 Orbbec SDK ROS

以下假設 ROS 2 Jazzy 已安裝：

```bash
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths dinov3_ws/src --ignore-src -r -y --rosdistro jazzy

sudo apt update
sudo apt install libgflags-dev nlohmann-json3-dev \
  ros-${ROS_DISTRO}-image-transport \
  ros-${ROS_DISTRO}-image-transport-plugins \
  ros-${ROS_DISTRO}-compressed-image-transport \
  ros-${ROS_DISTRO}-image-publisher \
  ros-${ROS_DISTRO}-camera-info-manager \
  ros-${ROS_DISTRO}-diagnostic-updater \
  ros-${ROS_DISTRO}-diagnostic-msgs \
  ros-${ROS_DISTRO}-statistics-msgs \
  ros-${ROS_DISTRO}-xacro \
  ros-${ROS_DISTRO}-backward-ros \
  libdw-dev libssl-dev mesa-utils libgl1 libgoogle-glog-dev
```

## 6. 編譯 Orbbec 相機驅動

需要使用相機時，先準備好 `orbbec_sdk/src` 內的驅動原始碼，再執行：

```bash
cd /workspaces/isaac_ros-dev/orbbec_sdk
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y --rosdistro jazzy
colcon build --event-handlers console_direct+ --cmake-args -DCMAKE_BUILD_TYPE=Release
cd /workspaces/isaac_ros-dev
```

請在 `orbbec_sdk` 建置，因為 `run_camera.sh` 會載入 `orbbec_sdk/install/setup.bash`。
目前相機啟動腳本使用 `gemini_330_series.launch.py`；僅上傳圖片時可略過相機建置與啟動。

## 7. 啟動

需要相機時，在第一個終端執行：

```bash
cd /workspaces/isaac_ros-dev
./dinov3_ws/run_camera.sh
```

在另一個終端啟動辨識服務：

```bash
cd /workspaces/isaac_ros-dev
./dinov3_ws/run.sh
```

預設為 `split` 模式。瀏覽器開啟 <http://127.0.0.1:8090>，可使用相機畫面或上傳圖片。

若要使用 `split` 模式，改執行以下指令，不需同時啟動兩種模式：

```bash
./dinov3_ws/run.sh --mode split
```

## 8. 更換 SAM3 Prompt

SAM3 接受 1–4 個以逗號分隔的類別名稱，例如 `medicine bottle, medicine box.`。
可在網頁直接更新 prompt；文字特徵會在首次使用時編碼並快取，不需重建引擎。

下次啟動的預設 prompt 設定於 `config/dinov3_cases.yaml` 的 `grounding_prompt`。
SAM3 不使用 OmDet 的 `omdet_prompts.json` 或離線 prompt bank。

更換 `DART/dartf/assets_bbox/` 的模型後，重新啟動服務即可。此目錄須包含原始 TRT10 backbone、兩個 grounding heads、文字模型、tokenizer 與位置編碼；啟動時會檢查完整性。SAM3 固定由 `DART/.venv` 搭配 `DART/dartf/plugins/lq_plugins.so` 載入，不能混用 TRT11 引擎或 plugin。
