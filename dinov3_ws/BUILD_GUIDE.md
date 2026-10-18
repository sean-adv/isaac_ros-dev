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

預設設定集中在 [config/dinov3_cases.yaml](config/dinov3_cases.yaml)。SAM3 接受 1–4 個以逗號分隔的類別名稱，例如 `medicine bottle, medicine box.`；網頁可直接更新，文字特徵按 prompt 快取，不需重建引擎。目前 confidence threshold 為 0.5、NMS 停用、每幀最多 20 個框。

SAM3 偵測間隔可在網頁的 **SAM3 detection interval** 即時調整，或啟動時指定 `-p grounding_detection_interval:=5`。每隔 N 個不同處理影格嘗試提交一次背景 SAM3＋DINO，最多一個進行中的任務，不排隊；設為 1 則停用 KCF，維持每幀同步 SAM3。等待 SAM3／DINO 時，主流程持續用原本 KCF 更新 bbox，分類沿用快取。首次初始化／切換來源或設定時持續接收最新 RGB，不發布暫時的空結果；相同輸入未改變時不重複複製。計數以實際處理的不同影格為準，相機跳過的影格不計入。

SAM3 完成後，同一背景工作先在該次偵測的原始 RGB 影格與 bbox 上執行 DINOv3／Lab，並建立物件診斷快照，再以該偵測影格初始化候選 KCF。每次補追都直接取當下最新的已處理影格，跳過中間影格；若更新完成時又有新影格，就再次取最新影格更新，追上後才一起交接框、分類與診斷快照。偵測影像的獨立副本保留至初始化完成，共享偵測輸入租約在 SAM3 回覆後釋放。排程器只保存一張最新影格，不保存中間影格歷史，因此不再因歷史缺幀而丟棄候選。

補追呼叫次數與時間上限由 `kcf_replay_max_frames`（預設最多 8 次更新，並非跳過影格數或歷史容量）與 `kcf_replay_budget_ms` 控制，不含 SAM3、DINO 與 KCF 初始化耗時。時間在每次更新前檢查，單次不可中斷的更新可能超過預算；若已追上即可交接，否則超限後不再啟動下一次更新。超限時丟棄候選 tracker，保留前景 tracker 與當前有效框，依正常偵測間隔再提交。追蹤失敗或 context 變更仍會丟棄候選；變更來源後清空最新影格，不套用舊結果。跳幀間隔過大仍可能使 KCF 追蹤失敗。`grounding_schedule` 提供 `seed_frame_key`（原始偵測影格）、`seed_skipped_frames`（固定為 0，始終在偵測影格初始化）、`catchup_skipped_frames`（補追跳過的已處理影格總數）、`init_ms`、`replay_frames`（實際補追呼叫次數）、`replay_ms`、`discard_reason`，以及目前 `busy`、`initializing`。策略修改後需重新啟動服務。

網頁狀態列的 **result FPS** 是完整辨識主迴圈最近 3 秒的結果頻率，包含影像準備、深度等待、候選框更新與每幀 3D 計算（非同步模式的 DINO 已在背景執行）；不是 KCF 單獨 FPS 或 MJPEG 顯示 FPS。相機區塊另外顯示 **input FPS**。`/api/status` 的 `performance` 提供 `image_prepare_ms`（包含共享複製）、`shared_copy_ms`（前者的子項）、`depth_wait_ms`、`proposal_update_ms`（包含非同步排程、最新影格複製與前景 KCF）、`recognize_ms`（交接時記錄背景辨識耗時，其餘影格為 0）、`bbox3d_ms`、`render_ms`。首次交接的深度等待發生在辨識後，會同時包含在 `bbox3d_ms` 中。平常的 `detect_ms` 為影像準備＋深度等待＋候選框更新的合計，不能當成純 KCF 耗時。

bbox ROS 發布由獨立 worker 發送 Detection2D／Detection3D／AprilTag 資料，不包含 RGB。主頁只開一條 `/view.mjpg`，有可顯示 bbox 且疊圖未超過 3 秒時顯示 960×540 辨識疊圖；沒有框、3D 模式缺少有效深度或結果過期時，顯示原始相機預覽。兩種 JPEG 由各自 worker 按需求編碼，主頁正常顯示 bbox 時會停用原始預覽編碼。上傳圖片維持顯示上傳結果，不受 3 秒時限影響；影像管理頁或其他直接訂閱 `/camera.mjpg`／`/overlay.mjpg` 的客戶端仍可要求各自串流。有 ROS debug 訂閱者才產生全解析度疊圖影像。JPEG 或 DDS 不會直接阻塞 bbox worker，但繪圖、編碼與背景 KCF 會共用 CPU／記憶體資源。共享影格池若被 SAM3、正在繪圖與待繪圖工作占滿，主迴圈會先捨棄尚未開始的舊繪圖工作，回收該影格租約再取新輸入，避免舊疊圖阻擋推論。DINOv3／Lab 與圖庫特徵更新已移到 SAM3 的背景工作；深度同步與 3D 計算仍在主迴圈。

任一主流程 KCF 追蹤失敗或框無效時會清除失效框並提前要求背景偵測；當下已有任務就等它完成，不同步阻塞補偵測。新 SAM3 結果取代整組追蹤器，空偵測會清除舊框。DINOv3、深度／3D 框與發布使用當幀框。`bbox_source=sam3` 表示套用了新偵測（可能已經過追趕），其餘為 `kcf`；追蹤期間的 `detection_score` 保留最近一次 SAM3 分數。`grounding_ipc.worker_detect_ms` 為背景偵測耗時，主流程的 `detect_ms` 則是取得當幀提案的耗時。bbox 直接顯示當幀結果，不再提供額外的 ROITracker ID 配對或命中門檻。

`config/dinov3_cases.yaml` 的 `bbox_smoothing_frames` 控制 bbox 移動平均，預設為 3（設為 1 停用），修改後重新啟動。此設定不顯示在網頁。每個物件取最近 N 個不同處理影格的原始 SAM3／KCF 框平均；不足 N 幀時使用已有樣本。SAM3 更新時依位置配對歷史，物件消失或來源／解析度／設定變更時清除歷史。同一幀重跑不增加樣本。平滑框供辨識裁切、顯示、3D 框與發布使用，KCF 本身仍以原始框運作。

KCF 前景與背景 replay 都使用固定 ROI；`kcf_roi_padding: 0.5` 表示偵測框四邊各擴張 50%，ROI 超出原圖的部分會裁切到圖像邊界。只將當前影格的 ROI 轉為 BGR，不縮小物件解析度。ROI 保持固定到下次 SAM3 交接，KCF 的局部框會映射回原圖座標。追蹤框距 ROI 人工邊界小於等於當前框寬／高的 10%（至少 1 pixel）時，保留有效框並提前要求背景偵測；跨出人工邊界時視為追蹤失敗。真實圖像邊界正常裁切，不因此反覆補偵測。參數修改後需重啟辨識服務。

相機 RGB、深度與深度 CameraInfo 訂閱在 `node.py` 固定使用 `ReliabilityPolicy.RELIABLE`，搭配 `KEEP_LAST`、depth=1，不提供 YAML 切換參數。此環境的 Orbbec 發布端為 RELIABLE；以獨立訂閱者交替測試，BEST_EFFORT 約 13 FPS、RELIABLE 約 29.7 FPS。這改善 ROS 接收缺幀，不保證辨識模型也能完成 30 FPS。更新程式後需重啟辨識服務，相機不用重啟。

### 相機接收 FPS 與執行使用者

相機、辨識服務與 ROS 量測工具應使用同一個 Linux 使用者；本專案的開發容器預設為 `admin`。在兩個 admin 終端分別執行 `./dinov3_ws/run_camera.sh` 與 `./dinov3_ws/run.sh`，避免其中一端經由 `sudo` 變成 root。可先執行 `id -un` 確認；從 host 進入既有容器時使用 `docker compose exec --user admin isaac-ros bash`。

2026-10-18 以同一個相機程序、1920×1080 RGB、30 FPS、RELIABLE／KEEP_LAST depth=1，依序使用 admin、root、admin 接收，每輪暖機 4 秒再量測 12 秒，影像接收率為 **29.89、22.24、29.74 FPS**。root 接收端也無法探索到相機節點名稱；三輪均沒有新增 UDP `RcvbufErrors`。這定位到跨使用者通訊問題，與 [Fast DDS SHM 的使用者權限限制](https://fast-dds.docs.eprosima.com/en/2.x/fastdds/transport/shared_memory/shared_memory.html) 一致。先前以 root 執行的低 FPS 量測不能代表全部以 admin 執行的效能。

優先統一程序使用者並重新啟動相關節點，再量測；不需為此降低相機解析度、放寬 `/dev/shm` 權限或先修改系統 UDP 緩衝。若同使用者仍掉幀，另查發布端時間戳、ROS 訂閱 QoS 與系統負載。CameraInfo 約 30 FPS 本身不等於完整影像已成功傳送 30 FPS。

對照資料與可重現量測腳本位於 `artifacts/camera_delivery_user_comparison_20261018/`。此實驗只驗證相機接收，不保證完整辨識結果也達到 30 FPS。

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

## 3D 深度計算的 ROI 與射線快取

3D worker 保留原始 `16UC1`／`32FC1` 深度，裁切 bbox 後才轉 `float64` 公尺，避免每幀轉換整張深度圖。像素投影方向由 worker 專用的 `ProjectionRayCache` 保存；解析度、K、畸變係數或畸變模型改變時重建，只保留一份校正。分割、全部前景點、分位數、平均 Z 與診斷欄位算法保持一致。

1920×1080 的 float64 雙通道射線表占 33,177,600 bytes（約 33 MB），2026-10-18 實測首次計算含建表約 190–198 ms；之後每幀查表。這筆首次／校正變更成本需與穩態耗時分開看。

以 12 張真實對齊深度、固定測試 ROI、OpenCV 單執行緒、三輪交替順序量測，各方案每情境 72 次，平均耗時如下（ms）：

| 測試框 | 修改前 | 只做 ROI 轉換 | ROI 加射線快取 |
| --- | ---: | ---: | ---: |
| 1 個 300×300 | 17.32 | 14.09 | 9.02 |
| 1 個 600×400 | 21.42 | 19.04 | 13.50 |
| 3 個 300×300 | 36.41 | 33.78 | 23.35 |

逐欄比較的輸出差異為 0。此測試回放相同實際深度與固定 ROI，相機驅動持續運作，未執行 SAM3／DINO 或 JPEG；耗時包含深度轉換與 3D 框計算，不代表整體辨識 FPS，也不能直接套用前次不同場景的 15.4 ms。原版、深度輸入與報告保存在 `artifacts/depth_bbox_optimization_20261018/`，重跑方式見測試 README。

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

DINO 與 SAM3 共用 `grounding_detection_interval`：每次背景工作先由 SAM3 找框，再對同一偵測影格的框執行 DINO／Lab，完成 KCF replay 後一起交接。兩個模型依序執行，因為 DINO 需要 SAM3 的候選框；等待期間前景 KCF 持續更新位置並沿用原分類。分類欄位隨各自的 proposal 保留，不新增數字 ID，也不將分類跨組配對。來源、解析度或設定變更會重建追蹤；`grounding_detection_interval: 1` 則停用 KCF，由推論 worker 每個新影格依序執行兩個模型。

框座標、3D 深度與 ROS 結果依當前處理影格更新。Diagnostic images 保留 SAM3 偵測影格上的辨識裁切與特徵，KCF 影格不重建診斷快照。圖庫更新於下一次背景工作套用；網頁修改圖庫或設定會使舊結果失效，靜態上傳圖片則每秒可提交刷新工作。`performance.recognition_source` 為 `dino`／`cached`／`none`，`recognition_frame_id` 指出背景辨識使用的偵測影格，可能早於當前 KCF 影格。需重新啟動辨識服務生效。

## DINO + Lab 相似度

最終辨識分數為 `score = 0.8 * S_DINO + 0.2 * S_Lab`。DINO 保留原本的 descriptor cosine similarity；Lab 使用 bbox 原始裁切與每張參考圖的聯合 L/a/b 直方圖（8×16×16 bins，L1 正規化），以 Bhattacharyya coefficient 計算 0～1 相似度。直方圖不含模型輸入 padding，參考圖完全透明的像素不計入；全透明參考圖會拒絕載入。

先對每張參考圖融合兩種分數，再取各類別最高分；最佳類別、參考圖、threshold、margin、網頁與 ROS 的 score 都使用融合結果。結果另保留 `dino_score`、`lab_score` 便於檢查。診斷畫面的 comparison score 使用融合結果，patch heatmap 維持 DINO patch similarity。

Lab 跟隨 DINO 在同一背景工作更新，KCF 影格沿用；圖庫或辨識門檻改變於下一次背景辨識套用。重新啟動辨識服務後生效。

## 追蹤遺失與分類快取恢復

非同步 KCF 遺失部分物件時，保留其餘當幀有效框，立即嘗試提交背景 SAM3（不等待 detection interval，仍限制單一任務）。有效框保留各自原有分類，前景不因框數改變重跑 DINO／Lab，也不依列表位置重新配對分類；下一次背景 SAM3＋DINO 完成後再一起交接新結果。全部追蹤遺失時清除失效框與分類，持續輸出無框影像並等待新 SAM3 結果。

同一影格收到 SAM3 交接結果時，bbox 平滑快取會檢查原始框的數量、順序與座標，避免截掉新增物件；同幀修正取代原樣本，不重複累積平滑權重。真正的模型／影像處理例外仍沿用既有錯誤處理；上述恢復針對追蹤狀態與分類快取變化。
