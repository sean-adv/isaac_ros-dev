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

預設為 `single` 模式。瀏覽器開啟 <http://127.0.0.1:8090>，可使用相機畫面或上傳圖片。

若要使用 `split` 模式，改執行以下指令，不需同時啟動兩種模式：

```bash
./dinov3_ws/run.sh --mode split
```

## 8. 更換 OmDet Prompt

OmDet 使用離線 prompt bank。新增或修改文字時，需要重新編碼文字特徵；在既有容量內不需重建 engine。

### 切換已存在的 Prompt

在網頁的 prompt 選單選擇已載入的 profile 即可，不需重新建置。
若要更改下次啟動的預設選項，修改 [config/dinov3_cases.yaml](config/dinov3_cases.yaml) 的 `grounding_prompt`，再重新啟動服務。

### 新增或修改 Prompt／Labels

1. 在執行 `run.sh` 的終端按 `Ctrl+C`，停止推論程序。
2. 編輯 [config/omdet_prompts.json](config/omdet_prompts.json)。每個 profile 只填 `prompt` 與 `labels`，例如：

   ```json
   {
     "profiles": [
       {
         "prompt": "Detect bottle, box.",
         "labels": ["bottle", "box"]
       }
     ]
   }
   ```

   `prompt` 同時是實際任務句與網頁／ROS 選擇鍵；`labels` 是有序的輸出類別，必須明確填寫，程式不會從 prompt 自動猜測類別。
   若要保留原有選項，請保留原 profile，再於 `profiles` 陣列新增一筆。

3. 在工作區根目錄更新離線文字特徵：

   ```bash
   cd /workspaces/isaac_ros-dev
   ./dinov3_ws/build_omdet.sh --stage-prompt
   ```

   `--stage prompts` 是同一模式的別名。此步驟使用已有的官方 CLIP／checkpoint，更新 `prompts.npz`、`prompts.json` 與相同容量 engine 的 JSON metadata，不匯出 ONNX、不重建 engine。

4. 若修改了預設 profile 的 prompt，同步修改 [config/dinov3_cases.yaml](config/dinov3_cases.yaml) 中既有的參數：

   ```yaml
   grounding_prompt: 'Detect bottle, box.'
   ```

   此值須對應 prompt bank 裡的 `prompt`；保留 YAML 原本的節點結構與縮排。

5. 重新啟動服務，讓文字特徵快取與網頁選單重新載入：

   ```bash
   ./dinov3_ws/run.sh
   ```

6. 使用相關實際圖片確認偵測框、類別與分數。

### 容量限制與重建

以本文件的首次建置指令為例，每個 profile 最多 4 個 labels，任務句最多 77 tokens（含起訖標記）。
多個 profile 可有不同類別數；限制套用於每個 profile，不是所有 profile 的類別總數。
實際容量記錄於 `dinov3_ws/artifacts/omdet/prompts.json`。

超限時 `--stage-prompt` 會報錯，不會截斷文字，也不會替換既有提示產物。
若需要 8 個類別，停止推論後重新完整建置：

```bash
./dinov3_ws/build_omdet.sh --max-labels 8 --max-prompt-tokens 77
```

已有權重時不必再加 `--download`。CLIP 的任務句 token 容量上限仍為 77；每個 label 名稱也受 CLIP 的 77-token 上限限制。
重建完成後同步確認 YAML 的 `grounding_prompt`，再啟動服務。
