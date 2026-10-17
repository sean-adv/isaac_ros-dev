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

預設設定集中在 [config/dinov3_cases.yaml](config/dinov3_cases.yaml)。SAM3 接受 1–4 個以逗號分隔的類別名稱，例如 `medicine bottle, medicine box.`；網頁可直接更新，文字特徵按 prompt 快取，不需重建引擎。目前 confidence threshold 為 0.5、NMS 停用、每幀最多 20 個框；ROI tracking 預設停用。

SAM3 偵測間隔可在網頁的 **SAM3 detection interval** 即時調整，或啟動時指定 `-p grounding_detection_interval:=5`。設定檔預設 10：每隔 10 個不同處理影格嘗試提交一次背景 SAM3，最多一個進行中的任務，不排隊。等待回覆時，主流程持續用 KCF 更新當幀 bbox 並執行 DINOv3 辨識。首次初始化／切換來源或設定時，只提交一張背景偵測影像，等待期間相機回呼繼續保留最新輸入，但不累積 KCF 歷史，也不重複複製影像或等待深度。首次偵測完成後，KCF 先在偵測原圖初始化，再實際追蹤當前最新影格；沒有將舊框直接初始化於新影格。若位移太大導致首次追蹤失敗，清除失效框並從最新影像重新偵測；設為 1 則停用 KCF，維持每幀同步 SAM3。計數以實際送入推論的不同影格為準，相機跳過的影格不計入；相同靜態圖片重跑辨識也不增加計數。

穩態背景任務在 SAM3 原始輸入影格上建立一組新 KCF，再依序沿緩衝的處理影格追到最新位置，最後交給主流程；不直接將舊 bbox 套到新影格。緩衝最多 32 幀、約 64 MiB（至少保留一張影格；執行中批次另持有參照）。缺少必要影格、追趕途中追蹤失敗、或來源／解析度／設定已變更時丟棄舊結果，下一輪重新提交。共享輸入影格保留租約至 SAM3 回覆，避免讀取已被覆寫的資料。

穩態 replay 每次工作最多 `kcf_replay_max_frames: 8` 幀，時間預算 `kcf_replay_budget_ms: 150.0` ms，不含 SAM3 與 KCF 初始化。時間在每次 KCF 更新前檢查，單次不可中斷的更新可能超過預算，但不再啟動下一幀。超限時丟棄候選追蹤器，保留前景追蹤器與當前有效框，依正常偵測間隔再提交；不因超限立即強制重試。缺歷史、追蹤失敗或 context 變更仍走原有丟棄／補偵測路徑。每個 replay 更新前都檢查取消與 context，避免已切換來源的工作繼續消耗整批 replay。`performance.grounding_schedule` 提供最近工作的 `replay_frames`、`replay_ms`、`discard_reason`，以及目前 `busy`、`initializing`。參數修改後重新啟動。

任一主流程 KCF 追蹤失敗或框無效時會清除失效框並提前要求背景偵測；當下已有任務就等它完成，不同步阻塞補偵測。新 SAM3 結果取代整組追蹤器，空偵測會清除舊框。DINOv3、深度／3D 框與發布使用當幀框。`bbox_source=sam3` 表示套用了新偵測（可能已經過追趕），其餘為 `kcf`；追蹤期間的 `detection_score` 保留最近一次 SAM3 分數。`grounding_ipc.worker_detect_ms` 為背景偵測耗時，主流程的 `detect_ms` 則是取得當幀提案的耗時。既有 `tracking_enabled` 仍控制 ID 配對與 bbox 顯示門檻，與背景排程獨立。

`config/dinov3_cases.yaml` 的 `bbox_smoothing_frames` 控制 bbox 移動平均，預設為 3（設為 1 停用），修改後重新啟動。此設定不顯示在網頁。每個物件取最近 N 個不同處理影格的原始 SAM3／KCF 框平均；不足 N 幀時使用已有樣本。SAM3 更新時依位置配對歷史，物件消失或來源／解析度／設定變更時清除歷史。同一幀重跑不增加樣本。平滑框供辨識裁切、顯示、3D 框與發布使用，KCF 本身仍以原始框運作，與 `tracking_enabled` 無關。

KCF 前景與背景 replay 都使用固定 ROI；`kcf_roi_padding: 0.5` 表示偵測框四邊各擴張 50%，ROI 超出原圖的部分會裁切到圖像邊界。只將當前影格的 ROI 轉為 BGR，不縮小物件解析度。ROI 保持固定到下次 SAM3 交接，KCF 的局部框會映射回原圖座標。追蹤框距 ROI 人工邊界小於等於當前框寬／高的 10%（至少 1 pixel）時，保留有效框並提前要求背景偵測；跨出人工邊界時視為追蹤失敗。真實圖像邊界正常裁切，不因此反覆補偵測。參數修改後需重啟辨識服務。

相機 RGB、深度與深度 CameraInfo 訂閱在 `node.py` 固定使用 `ReliabilityPolicy.RELIABLE`，搭配 `KEEP_LAST`、depth=1，不提供 YAML 切換參數。此環境的 Orbbec 發布端為 RELIABLE；以獨立訂閱者交替測試，BEST_EFFORT 約 13 FPS、RELIABLE 約 29.7 FPS。這改善 ROS 接收缺幀，不保證辨識模型也能完成 30 FPS。更新程式後需重啟辨識服務，相機不用重啟。

主程序的 `opencv_num_threads` 預設設定為 `1`，限制 OpenCV 內部執行緒池以控制 CPU 負載；前景、背景 replay、發布及繪圖仍並行。設定為 `0` 保留 OpenCV 函式庫預設值。此設定只作用於主程序，不更改 SAM3 子程序或 PyTorch／BLAS 執行緒數，修改後重新啟動。

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

目前服務使用 ViT-B/16 FP16、224×224 輸入，對應 14×14 patch grid。依 `dinov3_tensorrt` 的 Torch-TensorRT CLI 重建並驗證：

```bash
IMAGE_SIZE=224 OUTPUT_DIR="$PWD/dinov3_tensorrt/artifacts/thor-vitb" ./dinov3_tensorrt/scripts/run_thor_vitb.sh
```

目前使用先前已建置的 224×224 engine，產物與精度報告位於 `dinov3_tensorrt/artifacts/thor-vitb/`。160×160 engine 保留於 `dinov3_tensorrt/artifacts/thor-vitb-160/`。

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

網頁 FPS 為辨識結果完成率，以最近約 3 秒的完成間隔總數除以總耗時，包含 SAM3／KCF 交替、辨識與排程等待；不是相機輸入或單一模型 FPS。切換輸入／設定時重新計算，首個結果尚無 FPS。API 的 `performance.result_fps` 為平滑值，`result_fps_instant` 保留相鄰結果的瞬時值。

## 框線繪製模式

在 `config/dinov3_cases.yaml` 設定 `bbox_visualization_mode: "3d"`（目前預設），或改成 `"2d"`，重新啟動辨識服務後生效。瀏覽器先將 RGB 原圖縮為 960×540，再轉色與繪製指定模式；2D 座標與 3D 投影內參依寬高比例換算，公尺座標及畸變係數保持原值。只有 ROS debug image 有訂閱者時才另外繪製全解析度疊圖；網頁不提供切換，舊網址的 `bbox` 查詢參數也不會覆寫設定。

3D 模式需要 `pose_enabled: true` 與有效同步深度；沒有有效 3D 資料的物件不繪框，也不退回 2D。2D bbox 仍用於偵測、追蹤、辨識裁切與 3D 計算；此設定只決定繪製方式，不會停用這些必要運算或改變 3D 結果發布。

## DINO 分類更新頻率

DINO 配合 `grounding_detection_interval`：新的 SAM3 結果完成 KCF replay 並交接後，才對該組當前框執行一次 DINO；背景偵測執行中與一般 KCF 影格沿用分類、相似度及 margin。分類依附追蹤器組實例與固定的物件順序，不新增數字 ID，也不將分類跨組配對。KCF 遺失／空偵測會清除分類，新組一律重新辨識（包括 unknown）。來源、解析度或設定變更會重建追蹤；`grounding_detection_interval: 1` 則每張同步 SAM3 影格都執行 DINO。

框座標、3D 深度與 ROS 結果仍依當前處理影格更新。Diagnostic images 保留最近一次辨識的原始裁切與特徵，KCF 影格不重建診斷快照。圖庫更新時用已保存的 descriptor 重新分類與更新診斷快照，不重新前向推論目標影像（新參考圖仍須建立特徵）。`performance.recognition_source` 為 `dino`／`cached`／`reclassified`／`none`，`recognition_frame_id` 指出目標影像最近一次 DINO 前向推論的影格。需重新啟動辨識服務生效。
