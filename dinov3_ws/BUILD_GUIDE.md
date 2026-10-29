# 官方 SAM3 + Cat2Real 使用與建置

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

預設設定集中在 [config/dinov3_cases.yaml](config/dinov3_cases.yaml)。SAM3 接受 1–4 個以逗號分隔的類別名稱；文字特徵依 prompt 快取。`grounding_containment_threshold` 在最終追蹤框上，以交集／小框面積移除被包含的框。DINO 分類依 SAM3 原始影格裁切執行。

追蹤已改成 **DeepStream 9.1 NvDCF**，使用 GPU 整批追蹤物件。原生 C++ 子程序與模型 `.venv` 分離；子程序使用系統 CUDA／TensorRT，不繼承 `run.sh` 優先載入的 pip TensorRT 11。Python 不需要 `pyds` 或 PyGObject。容器內初次安裝與編譯：

```bash
sudo bash dinov3_ws/native/install_nvdcf.sh
# SDK 已安裝、只有 C++ 原始碼更新時：
bash dinov3_ws/native/build_nvdcf.sh
```

安裝器使用 NVIDIA 官方 DeepStream 9.1.0 ARM64 SDK。Compose 已改用 `native/Dockerfile` 衍生映像，從 host 執行 `docker compose build isaac-ros` 可將依賴與橋接程式寫入映像。現有容器已安裝，可直接重啟辨識服務；SDK 位於 `/opt/nvidia/deepstream/deepstream-9.1`，橋接執行檔位於 `artifacts/nvdcf/nvdcf_bridge`。模型與 `.venv` 不需重建。

SAM3 偵測間隔可從網頁或 `grounding_detection_interval` 調整。每 N 個不同處理影格提交一次背景 SAM3＋DINO；設為 1 則停用 NvDCF，使用同步偵測。背景工作先在偵測原始影格初始化候選 NvDCF，再補追一張最新已處理影格；主流程交接前會將候選更新至當前影格。等待期間舊 NvDCF 持續追蹤。只保留最新影像，不保存整段歷史，因此快速運動與大幅跳幀仍需實測。

每次成功 SAM3 結果逐物件銜接：匹配目標改由新 NvDCF context 接手，未匹配且仍追蹤有效的舊目標保留原 context；連續 `nvdcf_max_missed_detections` 次已提交 SAM3 結果漏偵測才移除（預設 3）。空偵測只增加一次漏偵測，背景等待及一般追蹤影格不計次；不再擁有有效目標的 context 會關閉。兩組追蹤器到達相同影格後，以位置關聯公開 `track_id`；這不是跨長期遮擋的 ReID。原生 `nvdcf_id` 僅在各自的 context 內有意義。辨識標籤、分數與診斷快照隨物件交接。

`nvdcf_max_missed_frames` 控制 Python 層的失追寬限（預設 30，設 0 停用，重啟生效）：容忍 N 個連續缺失的已處理影格，第 N+1 次缺失才移除，重跑同影格不計次。期間保留 context、分類與 ID，隱藏過期框並暫停該物件的 SAM3 miss 計數。部分或全部物件剛失追皆提前偵測；仍有隱藏物件時，以 `min(grounding_detection_interval, max(1, nvdcf_max_missed_frames // 3))` 幀重試 SAM3（預設 10），增加遮擋解除後的回配機會。新偵測與隱藏物件配對後移除舊 owner，避免同 ID 由兩組 context 重複輸出。

`nvdcf_width`／`nvdcf_height` 預設 640×384，須為 32 的倍數。只有追蹤影像縮放；輸出框還原至原始 RGB 座標，SAM3、DINO 裁切與深度／3D 計算使用原圖。NvDCF 設定在 [config/nvdcf.yaml](config/nvdcf.yaml)，可透過 ROS `nvdcf_config` 指定另一份檔案。預設使用 HOG＋ColorNames，不啟用 ReID 模型。`maxShadowTrackingAge` 控制偵測間隔內暫時遺失的原生追蹤存活期，與 Python 寬限分開；原生失追後可能需要新的偵測才能恢復輸出。

KCF 的 ROI、Lab 外觀檢查、平行 worker、replay 次數及 SAM3 miss 參數已移出啟用中的設定。舊 KCF 模組與歷史測試保留供回歸對照，執行管線不再使用它們。`opencv_num_threads` 只控制影像前處理的 OpenCV 執行緒池。

`bbox_source=sam3` 表示新偵測交接，其餘追蹤結果為 `nvdcf`。`detection_score` 保留 SAM3 分數；`tracker_confidence` 是 NvDCF 追蹤分數。`bbox_smoothing_frames` 控制原始追蹤框的移動平均，設為 1 停用。平滑結果不回饋追蹤器。

`performance.grounding_schedule` 提供 `backend`、`busy`、`initializing`、`seed_frame_key`、`ready_frame_id`、`catchup_skipped_frames`、`replay_frames`、`job_ms` 與 `handoff`。result FPS 包含完整結果迴圈；`proposal_update_ms` 包含追蹤與交接，不能當成純 SAM3 耗時。既有 3D bbox、ROS Detection2D／Detection3D／AprilTag、960×540 H.264／WebRTC 輸出繼續使用目前影格的框。

相機 RGB、深度與深度 CameraInfo 訂閱在 `node.py` 固定使用 `ReliabilityPolicy.RELIABLE`，搭配 `KEEP_LAST`、depth=1，不提供 YAML 切換參數。此環境的 Orbbec 發布端為 RELIABLE；以獨立訂閱者交替測試，BEST_EFFORT 約 13 FPS、RELIABLE 約 29.7 FPS。這改善 ROS 接收缺幀，不保證辨識模型也能完成 30 FPS。更新程式後需重啟辨識服務，相機不用重啟。

### 相機接收 FPS 與執行使用者

相機、辨識服務與 ROS 量測工具應使用同一個 Linux 使用者；本專案的開發容器預設為 `admin`。在兩個 admin 終端分別執行 `./dinov3_ws/run_camera.sh` 與 `./dinov3_ws/run.sh`，避免其中一端經由 `sudo` 變成 root。可先執行 `id -un` 確認；從 host 進入既有容器時使用 `docker compose exec --user admin isaac-ros bash`。

2026-10-18 以同一個相機程序、1920×1080 RGB、30 FPS、RELIABLE／KEEP_LAST depth=1，依序使用 admin、root、admin 接收，每輪暖機 4 秒再量測 12 秒，影像接收率為 **29.89、22.24、29.74 FPS**。root 接收端也無法探索到相機節點名稱；三輪均沒有新增 UDP `RcvbufErrors`。這定位到跨使用者通訊問題，與 [Fast DDS SHM 的使用者權限限制](https://fast-dds.docs.eprosima.com/en/2.x/fastdds/transport/shared_memory/shared_memory.html) 一致。先前以 root 執行的低 FPS 量測不能代表全部以 admin 執行的效能。

優先統一程序使用者並重新啟動相關節點，再量測；不需為此降低相機解析度、放寬 `/dev/shm` 權限或先修改系統 UDP 緩衝。若同使用者仍掉幀，另查發布端時間戳、ROS 訂閱 QoS 與系統負載。CameraInfo 約 30 FPS 本身不等於完整影像已成功傳送 30 FPS。

對照資料與可重現量測腳本位於 `artifacts/camera_delivery_user_comparison_20261018/`。此實驗只驗證相機接收，不保證完整辨識結果也達到 30 FPS。

主程序的 `opencv_num_threads` 預設設定為 `1`，限制 OpenCV 內部執行緒池以控制 CPU 負載；前景、背景 replay、發布及繪圖仍並行。設定為 `0` 保留 OpenCV 函式庫預設值。此設定只作用於主程序，不更改 SAM3 子程序或 PyTorch／BLAS 執行緒數，修改後重新啟動。

## 執行環境與模型

模型執行、下載與編譯均在 `dinov3_ws` 內完成，使用官方 Hugging Face、PyTorch、Torch-TensorRT 與 TensorRT 套件。預設不需要相鄰的 DART、dinov3_tensorrt 或 TensorRT-Model-Connect 原始碼、venv、plugin 或模型路徑。

| 用途 | 執行方式 | 本專案內模型 |
| --- | --- | --- |
| SAM3 vision | 官方 Torch-TensorRT FP16，1008×1008 | `artifacts/sam3/vision-fp16.ts` 與同名 `.json` |
| SAM3 bbox／文字 | 官方 Transformers PyTorch FP16，文字依 prompt 快取 | `artifacts/sam3/source/` |
| Cat2Real | 官方 Torch-TensorRT FP32，384×384 | `artifacts/cat2real-dinov3-384/cat2real-fp32-trt.ts` 與同名 `.json` |
| Cat2Real 前處理 | Transformers processor | `artifacts/cat2real-dinov3-384/source/` |
| 商品圖庫 | 首次載入／修改後重新編碼 | `data/gallery/` |

兩個程序都使用 `dinov3_ws/.venv/bin/python` 與 TensorRT 11，保留 IPC 以配合背景偵測和影格租約。SAM3 head 回傳 bbox、logits 與 presence，不執行 mask／SAM3 tracking，也不載入自訂 plugin 或 ONNX Runtime。支援 1–4 類別，直接使用實際類別數，沒有舊 K=1／K=4 engine bucket。

SAM3 前處理為 RGB bilinear 1008×1008、`pixel / 127.5 - 1`。Cat2Real 置中補白成正方形後以 OpenCV bilinear 縮放至 384×384，做一次 rescale 與 ImageNet normalize。辨識使用 L2-normalized CLS/pooler_output（1024 維），24×24 patch 特徵供熱圖。沿用 DINO 閾值篩選再以 Lab 選參考圖的分類流程。

環境清單在 [requirements.txt](requirements.txt)、[requirements-build.txt](requirements-build.txt)；Thor 的已驗證版本在 [constraints-thor.txt](constraints-thor.txt)。目前環境已放在 `.venv/` 並通過 pip check；從零安裝可執行：

```bash
bash dinov3_ws/setup_env.sh
```

這個入口先從官方 CUDA 13.0 wheel index 安裝 Torch／torchvision，再安裝受 constraints 限制的套件。預設 Python 3.12；可用 `PYTHON_BIN` 與 `DINOV3_VENV` 指定位置。CUDA driver/runtime、ROS 2 Jazzy、cv_bridge、vision_msgs、isaac_ros_apriltag_interfaces 與相機驅動是系統前提，不由 pip 安裝。`run_camera.sh` 使用 ROS 已能找到的 `orbbec_camera`，或透過 `ORBBEC_SETUP=/path/to/install/setup.bash` 載入已安裝的相機套件。

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

## 模型重建與驗證

以下指令不載入其他專案的 Python 程式：

```bash
# 用現有本機權重重建兩個模型
dinov3_ws/.venv/bin/python dinov3_ws/build_models.py

# 分別建置
dinov3_ws/.venv/bin/python dinov3_ws/build_models.py --target sam3
dinov3_ws/.venv/bin/python dinov3_ws/build_models.py --target cat2real

# 需要下載時使用固定 revision；SAM3 需先取得官方存取權並登入
dinov3_ws/.venv/bin/hf auth login
dinov3_ws/.venv/bin/python dinov3_ws/build_models.py --download

# 不重新編譯，只重載產物與原始 FP32 比較
dinov3_ws/.venv/bin/python dinov3_ws/build_models.py --validate-only

# 單／四類別、圖庫與 IPC 的實際 GPU 回歸
dinov3_ws/.venv/bin/python dinov3_ws/tests/check_official_pipeline.py
```

Cat2Real 固定 revision `a763964f2041722ffcc5bc4c72e3cb241530727e`；SAM3 固定 `3c879f39826c281e95690f02c7821c4de09afae7`。兩者都要求 vision 圖完整編譯，儲存 TorchScript 後重新載入，與重新載入的原始 FP32 權重比較，通過才寫入部署中繼資料。`validation.json` 保存模型 SHA256、套件版本、樣本結果與測速。Cat2Real FP16 在此 Thor 上會產生 NaN，因此固定 FP32。

SAM3 整個 bbox 圖目前無法直接完整編譯：decoder 有 scalar extraction 與 TensorRT reshape 限制。因此只將 vision 編譯成 TensorRT，bbox／文字使用原始官方 PyTorch 模組。`model_modules.py` 只包裝官方模組呼叫，省略不需要的分割遮罩。

`build_cat2real.py`、`build_sam3.py` 與 shell 建置入口轉交 `build_models.py`。舊 ONNX／TRTMC／Qwen 實驗程式不屬於目前部署，不需要安裝它們的依賴或執行它們。從零重建 engine 需要相容 GPU 與 TensorRT 版本。
