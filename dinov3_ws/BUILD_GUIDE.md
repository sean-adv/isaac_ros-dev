# SAM3 + DINOv3 使用與建置

所有指令從工作區根目錄執行。容器設定見 [DOCKER_HOST.md](../DOCKER_HOST.md)。

## 啟動

```bash
./dinov3_ws/run_camera.sh  # 使用相機時，在另一個終端執行
./dinov3_ws/run.sh
```

瀏覽器開啟 <http://127.0.0.1:8090>，可使用相機或上傳圖片。啟動參數可透過 ROS 覆寫，例如：

```bash
./dinov3_ws/run.sh --ros-args -p grounding_box_threshold:=0.5
```

預設設定集中在 [config/dinov3_cases.yaml](config/dinov3_cases.yaml)。SAM3 接受 1–4 個以逗號分隔的類別名稱，例如 `medicine bottle, medicine box.`；網頁可直接更新，文字特徵按 prompt 快取，不需重建引擎。目前 confidence threshold 為 0.7、NMS 停用、每幀最多 20 個框；ROI tracking 預設停用。

SAM3 偵測間隔可在網頁的 **SAM3 detection interval** 即時調整，或啟動時指定 `-p grounding_detection_interval:=5`。設定檔預設 5：第 1、6、11… 個處理影格執行 SAM3，其間根據上次偵測的 bbox 為每個物件建立 OpenCV KCF 追蹤器；設為 1 則每個處理影格都執行 SAM3。計數以實際送入推論的不同影格為準，相機在忙碌時跳過的影格不計入；相同靜態圖片因圖庫重新整理而重跑辨識也不增加計數。

任一 KCF 追蹤失敗或框無效時，當幀提前執行 SAM3，並重新開始計算間隔。每次 SAM3 都會更新所有追蹤器；空偵測會清除舊框。來源、解析度或設定變更時重新偵測。DINOv3 持續使用當幀的追蹤框辨識，深度／3D 框與發布也使用該框。`bbox_source` 表示 `sam3` 或 `kcf`；追蹤期間的 `detection_score` 保留最近一次 SAM3 分數。既有 `tracking_enabled` 仍控制 ID 配對與 bbox 顯示門檻，與 KCF 偵測間隔獨立。

`config/dinov3_cases.yaml` 的 `bbox_smoothing_frames` 控制 bbox 移動平均，預設為 3（設為 1 停用），修改後重新啟動。此設定不顯示在網頁。每個物件取最近 N 個不同處理影格的原始 SAM3／KCF 框平均；不足 N 幀時使用已有樣本。SAM3 更新時依位置配對歷史，物件消失或來源／解析度／設定變更時清除歷史。同一幀重跑不增加樣本。平滑框供辨識裁切、顯示、3D 框與發布使用，KCF 本身仍以原始框運作，與 `tracking_enabled` 無關。

KCF 需要主程序環境安裝 `opencv-contrib-python`（包含 [OpenCV KCF](https://docs.opencv.org/4.x/d2/da2/classcv_1_1TrackerKCF.html)）。若原先使用一般 OpenCV wheel，先移除同名 `cv2` 套件再安裝，避免重疊檔案；不需修改 SAM3 的 DART 環境：

```bash
dinov3_trt11_venv/bin/python -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python-headless
dinov3_trt11_venv/bin/python -m pip install 'opencv-contrib-python>=4.8,<4.12'
```

## 執行環境與模型

兩個程序固定使用各自的 TensorRT 環境；路徑預設定義於 `src/dinov3_cases/dinov3_cases/runtime.py`。

| 用途 | Python | 模型／引擎 |
| --- | --- | --- |
| SAM3 偵測 | `DART/.venv/bin/python`，TensorRT 10 | `DART/dartf/assets_bbox/vision_int8.plan` |
| DINOv3 辨識 | `dinov3_trt11_venv/bin/python`，TensorRT 11 | `dinov3_tensorrt/artifacts/thor-vitb/vitb16-fp16-trt.ts` |
| DINOv3 前處理 | 主程序環境 | `DART/models/dinov3-vitb16` |

SAM3 bundle 還需 `ground_c1m_fp16.plan`、`ground_c4m_fp16.plan`、`text_c16.onnx`（含外部權重）、`bpe_simple_vocab_16e6.txt.gz`、`img_pos_c1.npy`、`img_pos_c4.npy`，以及 `DART/dartf/plugins/lq_plugins.so`。缺少時依 DART 的 `dartf/scripts/build_bbox.sh` 準備；更換模型後重新啟動服務。

SAM3 使用套件內的 `sam3_runtime`，輸入為 RGB 1008×1008、bilinear resize、`pixel / 127.5 - 1`；單類使用 K=1 head，2–4 類使用 K=4 head。DINOv3 將 bbox 與參考圖等比例縮放、補邊至 224×224，再比對 patch 特徵。參考圖庫存於 `dinov3_cases_data/`。

主程序依賴見 [requirements.txt](requirements.txt)；先安裝平台相容的 CUDA、PyTorch／torchvision，再安裝一般依賴。ROS 2 Jazzy、cv_bridge、vision_msgs、isaac_ros_apriltag_interfaces 與相機驅動需另外準備。SAM3 worker 沿用 DART 的獨立環境，不能套用 TRT11 的依賴限制。

## 程式與工具入口

| 檔案 | 用途 |
| --- | --- |
| `run.sh`、`launch_pipeline.py` | 設定主程序環境、模型路徑與 ROS 參數 |
| `node.py` | ROS 輸入、推論排程及結果發布 |
| `grounding_ipc.py`、`frame_pool.py` | SAM3 子程序與三槽共享 RGB 影格池 |
| `sam3_trt.py`、`encoder.py`、`trt_backend.py` | SAM3 偵測與 DINOv3 特徵後端 |
| `grounded.py`、`tracking.py` | 參考圖比對、診斷視圖及可選追蹤 |
| `depth_sync.py`、`depth_bbox.py`、`apriltag.py` | 深度同步、3D 框及 ROS 訊息 |
| `web.py`、`image_manager.py`、`static/`、`templates/` | HTTP API、圖庫與網頁 |

上述 Python 模組均位於 `src/dinov3_cases/dinov3_cases/`。

共享池的 slot 在推論與繪圖使用完畢前保持唯讀；socket 僅傳控制與結果。bbox 發布與 JPEG 繪圖各有獨立執行緒。`processing_ms`／`callback_to_result_ms` 截止於辨識與 3D 框完成，`render_ms`／`result_to_overlay_ms` 另計。

## 可選模型建置

目前服務直接載入上表的既有產物。以下工具供另建引擎及驗證使用，產物不會自動取代目前部署。

- `build_dinov3.sh`：透過 TensorRT-Model-Connect 建置原生 TensorRT bundle，預設為 ViT-S/16、224×224、FP16；後端仍支援此格式。
- `build_sam3_plugins.sh`、`build_sam3.sh`：實驗性 SAM3 TRT11 轉建；目前 SAM3 worker 固定使用 TRT10，不能直接載入這些產物。

建置依賴見 [requirements-build.txt](requirements-build.txt)。DINOv3 需準備 `TensorRT-Model-Connect` 原始碼及其依賴，並取得 Hugging Face 模型存取權：

```bash
dinov3_trt11_venv/bin/hf auth login
./dinov3_ws/build_dinov3.sh --download
```

已快取權重時使用 `--offline`。可用 `--model /path/to/model`、`--output /path/to/new-output` 指定本機模型與新產物目錄；其餘參數見 `./dinov3_ws/build_dinov3.sh --help`。原生引擎建置需以 `admin` 執行，避免權重與產物權限問題。

SAM3 TRT11 轉建需來源 `artifacts/sam3/bbox_w8a8/`、DART CUTLASS、CUDA 13 與匹配的 TensorRT 11 headers；`TRT_INC` 可指定 headers 位置：

```bash
./dinov3_ws/build_sam3_plugins.sh
./dinov3_ws/build_sam3.sh
```

## 測試與診斷

回歸測試、GPU 整合驗證及效能量測指令見 [tests/README.md](tests/README.md)。歷史精度與效能分析保留於 [OMDET_DINOV3_THOR_ANALYSIS.md](OMDET_DINOV3_THOR_ANALYSIS.md)，僅供查閱過去量測；舊 OmDet 程式與離線 prompt 設定已移除。

## 相機依賴

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

## 編譯相機驅動

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
