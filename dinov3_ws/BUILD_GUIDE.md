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

預設設定集中在 [config/dinov3_cases.yaml](config/dinov3_cases.yaml)。SAM3 接受 1–4 個以逗號分隔的類別名稱，例如 `medicine bottle, medicine box.`；網頁可直接更新，文字特徵按 prompt 快取，不需重建引擎。目前 confidence threshold 為 0.5、IoU NMS 停用、每幀最多 20 個框。SAM3 框裁切至原圖並通過尺寸過濾後，先以 `grounding_containment_threshold`（預設 0.95，範圍 0–1；0 停用，修改後重啟）作為「交集面積 ÷ 小框面積」門檻移除被大框包含的小框（跨 prompt 類別，優先保留大框；同面積時保留高信心框），再套用 IoU NMS 與框數上限，最後進行 DINO 分類及 KCF 初始化。

SAM3 偵測間隔可在網頁的 **SAM3 detection interval** 即時調整，或啟動時指定 `-p grounding_detection_interval:=5`。每隔 N 個不同處理影格嘗試提交一次背景 SAM3＋DINO，最多一個進行中的任務，不排隊；設為 1 則停用 KCF，維持每幀同步 SAM3。等待 SAM3／DINO 時，主流程持續用原本 KCF 更新 bbox，分類沿用快取。首次初始化／切換來源或設定時持續接收最新 RGB，不發布暫時的空結果；相同輸入未改變時不重複複製。計數以實際處理的不同影格為準，相機跳過的影格不計入。

SAM3 完成後，同一背景工作先在該次偵測的原始 RGB 影格與 bbox 上執行 DINOv3／Lab，並建立物件診斷快照，再以該偵測影格初始化候選 KCF。每次補追都直接取當下最新的已處理影格，跳過中間影格；若更新完成時又有新影格，就再次取最新影格更新，追上後才一起交接框、分類與診斷快照。偵測影像的獨立副本保留至初始化完成，共享偵測輸入租約在 SAM3 回覆後釋放。排程器只保存一張最新影格，不保存中間影格歷史，因此不再因歷史缺幀而丟棄候選。

補追呼叫次數由 `kcf_replay_max_frames`（預設最多 8 次更新，並非跳過影格數或歷史容量）控制，沒有經過時間上限；`replay_ms` 仍記錄補追耗時。超限時丟棄候選 tracker，保留前景 tracker 與當前有效框，依正常偵測間隔再提交。追蹤失敗或 context 變更仍會丟棄候選；變更來源後清空最新影格，不套用舊結果。跳幀間隔過大仍可能使 KCF 追蹤失敗。`grounding_schedule` 提供 `seed_frame_key`（原始偵測影格）、`seed_skipped_frames`（固定為 0，始終在偵測影格初始化）、`catchup_skipped_frames`（補追跳過的已處理影格總數）、`init_ms`、`replay_frames`（實際補追呼叫次數）、`replay_ms`、`discard_reason`，以及目前 `busy`、`initializing`。策略修改後需重新啟動服務。

網頁狀態列的 **result FPS** 是完整辨識主迴圈最近 3 秒的結果頻率，包含影像準備、深度等待、候選框更新與每幀 3D 計算（非同步模式的 DINO 已在背景執行）；不是 KCF 單獨 FPS 或 MJPEG 顯示 FPS。相機區塊另外顯示 **input FPS**。`/api/status` 的 `performance` 提供 `image_prepare_ms`（包含共享複製）、`shared_copy_ms`（前者的子項）、`depth_wait_ms`、`proposal_update_ms`（包含非同步排程、最新影格複製與前景 KCF）、`recognize_ms`（交接時記錄背景辨識耗時，其餘影格為 0）、`bbox3d_ms`、`render_ms`。首次交接的深度等待發生在辨識後，會同時包含在 `bbox3d_ms` 中。平常的 `detect_ms` 為影像準備＋深度等待＋候選框更新的合計，不能當成純 KCF 耗時。

bbox ROS 發布由獨立 worker 發送 Detection2D／Detection3D／AprilTag 資料。即時畫面統一縮放及畫框成 960×540，由單一 H.264 編碼器產生 Annex-B access unit，再分別交給網頁 WebRTC 與 ROS `/medicine_recognition/image_annotated/h264`（`sensor_msgs/CompressedImage`，format=h264）；兩端共用相同編碼資料，不重新編碼。預設 auto 優先 NVENC、不可用則 libx264。無觀看者與 ROS 訂閱者時停止繪圖／編碼。相機模式有有效框且結果未超過 3 秒時顯示疊圖，否則選最新相機預覽；上傳結果不過期。圖庫頁也使用同一條影像，原始相機擷取仍獨立取 RGB。DDS 影像發布與 bbox 發布各有 worker；慢影像訂閱者不阻塞編碼或 bbox。接收端佇列溢位時等待新 IDR，不任意保留缺參考影格的 P-frame。MJPEG 路由與舊原始影像輸出已移除。參數、ROS 解碼方式及 host-only ICE 限制見 `src/dinov3_cases/README.md` 的「共用 H.264 即時影像」。共享影格池滿時仍先回收尚未開始的舊繪圖工作，讓推論繼續。

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

KCF 需要主程序環境安裝 `opencv-contrib-python`（包含 [OpenCV KCF](https://docs.opencv.org/4.x/d2/da2/classcv_1_1TrackerKCF.html)）。若原先使用一般 OpenCV wheel，先移除同名 `cv2` 套件再安裝，避免重疊檔案；主程序與 SAM3 worker 共用同一份 `.venv` 套件：

```bash
dinov3_ws/.venv/bin/python -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python-headless
dinov3_ws/.venv/bin/python -m pip install 'opencv-contrib-python>=4.8,<4.12'
```

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
