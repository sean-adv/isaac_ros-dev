# 目前部署測試

一般回歸使用 `dinov3_ws/.venv`。`test_qwen_encoder.py` 為舊 Qwen 實驗，其缺少的設定常數早於本次切換；目前部署不執行該測試：

```bash
source /opt/ros/jazzy/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=dinov3_ws/src/dinov3_cases:$PYTHONPATH \
  dinov3_ws/.venv/bin/pytest dinov3_ws/tests --ignore=dinov3_ws/tests/test_qwen_encoder.py
```

實際 GPU 回歸：

```bash
dinov3_ws/.venv/bin/python dinov3_ws/tests/check_official_pipeline.py
```

測試影像預設為 `artifacts/validation/person.jpg`。報告輸出 `artifacts/standalone-pipeline-validation.json`，檢查本機模型路徑、FP32 bbox IoU、1–4 類別、Cat2Real 圖庫、IPC parity，以及沒有載入其他模型專案的模組或自訂 SAM3 plugin。數值模型驗證另執行 `build_models.py --validate-only`。下方保留其他功能回歸與歷史效能診斷入口；舊模型實驗腳本不屬於目前部署。

# 測試與診斷

此目錄只保留目前 SAM3 + DINOv3 流程的回歸測試，以及部署驗證與效能診斷工具。舊 OmDet 的離線 prompt、FP16/FP32 精度比較、IPC GPU 比較與舊版整合腳本已移除；原始量測報告仍保留。

## 日常回歸測試

KCF 非同步模式（`grounding_detection_interval > 1`）採逐物品交接：背景 SAM3
結果在偵測影格初始化、補追到目前影格後，以框重疊、中心距離及尺寸配對。
配對物品沿用固定 `track_id` 並更新自己的 KCF／ROI；SAM3 漏偵測或回傳空集合時，
舊 KCF 可在有限時間內繼續追蹤。新舊框統一可短暫 coasting，但連續漏掉三次
SAM3 結果或距最後 SAM3 支持達五秒即淘汰，即使 KCF 仍回報成功也不續命。
這些限制分別由 `kcf_max_missed_detections` 與
`kcf_max_unconfirmed_seconds` 設定，修改後重啟。漏偵測次數僅計算完成交接的
SAM3 結果；相機／KCF 更新不增加次數。偵測過期時間從 SAM3 輸入影格時間起算，
不從延遲結果完成時重新起算。被淘汰的身分也會從恢復快取移除。網頁物品清單
顯示 track 狀態、SAM3 age 及 miss 次數；`unknown` 類別本身不會觸發淘汰。
失效的 KCF 不顯示舊框，但保留最後身分一秒供鄰近位置
重新配對；逾時、相機或設定 context 改變後不沿用舊身分。外觀近似物品交叉或
移動過遠仍可能換 ID，這版配對尚未加入外觀特徵。同步模式（間隔為 1）仍逐幀
使用 SAM3，沒有 KCF 可以在漏偵測時保留。

在工作區根目錄執行，不需啟動相機或載入部署引擎：

```bash
source /opt/ros/jazzy/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="$PWD/dinov3_ws/src/dinov3_cases:$PYTHONPATH" \
dinov3_ws/.venv/bin/python -m pytest -q dinov3_ws/tests
```

使用專案 TRT11 Python 環境，並先載入 ROS 2 Jazzy；若 `isaac_ros_apriltag_interfaces` 安裝於額外的工作區，還需載入該工作區環境。保留既有 `PYTHONPATH`，讓 ROS 訊息可匯入。停用自動載入 pytest 外掛，是為了避免未使用的 ROS launch-testing 外掛引入額外依賴；本套測試不依賴外掛。

| 測試 | 保留原因 |
| --- | --- |
| `test_sam3.py` | Prompt 驗證、分數、原圖座標與 head 輸入 |
| `test_tracked_recognition.py` | DINO 隨 KCF 組生命週期快取、新組重辨識、遺失清除、圖庫重分類與同步模式 |
| `test_grounded.py` | 物品辨識、unknown、框過濾、圖庫更新與熱圖 |
| `test_patches.py`、`test_reference_gallery.py` | DINOv3 前處理、patch 順序與參考圖對齊 |
| `test_trt_backend.py`、`test_build_dinov3.py` | 引擎輸入限制、建置產物與驗證證據 |
| `test_frame_pool.py`、`test_grounding_ipc.py` | 共享影格生命週期、子程序失敗、逾時與清理 |
| `test_realtime.py`、`test_h264_stream.py` | 最新影格排程、發布與繪圖互不阻塞、共用 H.264 編碼、IDR 恢復、ROS 封包與雙 WebRTC 接收端實際解碼（不重新編碼） |
| `test_cloud_filter.py`、`test_depth_bbox.py`、`test_depth_sync.py` | 深度同步、有效座標、3D 框與 ROS 訊息 |
| `test_apriltag.py` | Isaac ROS 訊息序列化、座標及空結果發布 |
| `test_bbox_visualization.py` | 2D/3D 框投影、config 指定單一模式與首頁顯示 |
| `test_image_manager.py`、`test_camera_capture.py` | 圖庫、去背、相機擷取與 bbox 預覽 |
| `test_tracking.py` | bbox 移動平均、跨偵測配對、同幀修正與歷史重設 |
| `test_async_kcf.py` | 背景 SAM3 與 DINO 共用排程且不卡住 KCF、偵測影格初始化／補追及框分類原子交接、過期結果與呼叫／時間預算、租約及錯誤處理 |
| `test_kcf.py` | SAM3 間隔、KCF 框更新、失敗補偵測、重設與實際 OpenCV 追蹤 |

`ipc_fake_worker.py` 是 IPC 測試必要的子程序 fixture，不能當作一般診斷腳本刪除。

前端互動測試另需 Node.js：

```bash
node --test dinov3_ws/tests/test_images_ui.cjs
```

這組測試檢查手動 prompt 輸入、輪詢保留文字、擷取驗證、重複點擊與 bbox 預覽。

## 部署整合驗證

以下工具需要實際 GPU、模型與 ROS 環境，修改引擎或部署流程時才執行，不納入日常 pytest：

- `check_gpu_environment.py`：檢查 CUDA driver、PyTorch allocation 與 TensorRT Builder。
- `check_sam3_trt_pipeline.py`：轉交 `check_official_pipeline.py`，驗證官方 SAM3 單／四類別、Cat2Real 與同環境 TRT11 IPC。
- `check_async_kcf_pipeline.py`：真實 SAM3／DINOv3 流程，先以 30 FPS 持續輸入並延遲首次 SAM3，確認以最新 RGB 初始化並略過偵測期間影格；再刻意阻塞穩態背景 SAM3，確認前景 KCF 持續完成、SAM3 偵測影格上完成 DINOv3 後，框、分類與診斷圖片一起交接。
- `check_grounded_trt_pipeline.py`：實際SAM3／Cat2Real HTTP／ROS 流程，以及刻意阻塞 H.264 編碼時的 bbox 發布。

從工作區根目錄執行：

```bash
LD_LIBRARY_PATH="$PWD/dinov3_ws/.venv/lib/python3.12/site-packages/tensorrt_libs:/usr/local/cuda/lib64" \
PYTHONPATH="$PWD/dinov3_ws/src/dinov3_cases:$PYTHONPATH" \
dinov3_ws/.venv/bin/python dinov3_ws/tests/check_sam3_trt_pipeline.py
```

其他部署工具使用相同環境，替換腳本名稱即可。工具會載入實際模型並寫出驗證報告，執行前可查看腳本中的輸出位置。

## 效能診斷

效能量測入口，按問題選用：

- `profile_camera_input.py`：只看 ROS 相機輸入的 FPS 與影格年齡。
- `profile_grounding_stages.py`：以儲存影像量測 SAM3 worker 與共享記憶體傳輸。
- `profile_camera_pipeline.py`：量測真實相機的整體辨識流程。

使用部署整合驗證的 Python 與環境，附上 `--help` 可查看參數。將結果輸出至 `dinov3_ws/artifacts/`，不放進測試程式目錄。


### KCF 延遲與 SAM3 間隔比較

```bash
source /opt/ros/jazzy/setup.bash
LD_LIBRARY_PATH="$PWD/dinov3_ws/.venv/lib/python3.12/site-packages/tensorrt_libs:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" \
PYTHONPATH="$PWD/dinov3_ws/src/dinov3_cases:${PYTHONPATH:-}" \
dinov3_ws/.venv/bin/python dinov3_ws/tests/profile_kcf_intervals.py \
  --intervals 30 15 10 5 30 --seconds 20 --warmup-seconds 5 \
  --output dinov3_ws/artifacts/kcf_interval_profile.json
```

使用目前 YAML 設定與真實相機、SAM3、DINOv3，直接計時主流程 KCF 與背景 replay，保存每幀資料及 median／p95。初始化與切換間隔期間持續接收相機輸入，使用正式流程的初始化等待與 replay 預算；每組仍在首次結果後暖機再量測穩態。冷啟動連續輸入與首次交接另由 `check_async_kcf_pipeline.py` 驗證。測試程序使用隨機 HTTP port，不修改正式參數檔。`kcf_update_ms` 包含 RGB→BGR、所有物件 KCF 更新與框整理；`kcf_detect_stage_ms` 另外包含主流程的影像準備、深度等待及非同步排程。間隔以處理影格計算，應搭配 `processed_frame_fps` 解讀。


`profile_kcf_threads.py` 比較 OpenCV 的 `1 / 2 / 4 / 14` 個內部執行緒。以固定 ROI 追蹤相同靜態相機影像中的手動標註框，在實際相機驅動繼續運作時，分別量測前景 KCF、前景＋持續背景 replay；加上 `--preview` 則改測 replay＋30 FPS 的 960×540 縮放、色彩轉換與共用 H.264 編碼。各組先暖機，三輪交替正反順序，每組預設 3 秒，執行緒池只在所有測試 worker 停止時調整。程序 CPU 的 100% 代表一個 CPU 核心；system CPU 是全機平均。這是靜態輸入 CPU 壓力測試，未執行 SAM3／DINOv3，不能視為模型推論 FPS。

```bash
PYTHONPATH="$PWD/dinov3_ws/src/dinov3_cases:${PYTHONPATH:-}" \
dinov3_ws/.venv/bin/python dinov3_ws/tests/profile_kcf_threads.py \
  --image dinov3_ws/artifacts/kcf_roi_camera_sample.png \
  --boxes dinov3_ws/artifacts/kcf_roi_manual_boxes.json \
  --preview --output dinov3_ws/artifacts/kcf_threads_preview.json
```


相機輸入量測需使用與待測服務一致的 QoS；目前 Orbbec 部署選用 RELIABLE、KEEP_LAST depth=1。沿用上方 ROS/Python 環境執行：

```bash
dinov3_ws/.venv/bin/python dinov3_ws/tests/profile_camera_input.py \
  --reliability reliable --queue-depth 1 --seconds 10 \
  --output dinov3_ws/artifacts/camera_reliable_input.json
```

工具仍預設 best_effort、depth=5，以保留舊量測口徑。測到 best_effort 掉幀不能直接推論感測器本身幀率不足。`profile_kcf_intervals.py` 會額外輸出 `camera_callback_fps` 與實際 `image_reliability`，將輸入接收率和模型結果 FPS 分開。

比較相同偵測間隔下的繪圖與深度成本，可執行 `profile_kcf_intervals.py --intervals 30 --seconds 10 --warmup-seconds 3 --output <report.json>`，依序加上 `--browser-streams`（要求兩路瀏覽器影像編碼）、`--no-pose`（停用深度同步／3D 框）。使用不同報告路徑；工具使用隨機 HTTP port。每份報告包含前景 KCF 計時與主迴圈各段耗時。此比較需要相機正在發布影像；只看到 launch 程序不代表有實際輸入。


### 3D 框 ROI 轉換與射線快取比較

使用已保存的同一批真實深度及修改前實作，分別量測整張轉換、ROI 轉換、ROI 加射線快取。工具逐欄比對所有 3D 與診斷輸出，並獨立記錄首次建表成本；固定 ROI 不代表 SAM3 實際偵測框，結果是此階段耗時而非完整流程 FPS。

```bash
PYTHONPATH="$PWD/dinov3_ws/src/dinov3_cases:${PYTHONPATH:-}" \
dinov3_ws/.venv/bin/python dinov3_ws/tests/profile_depth_bbox.py \
  --baseline dinov3_ws/artifacts/depth_bbox_optimization_20261018/baseline_depth_bbox.py \
  --samples dinov3_ws/artifacts/depth_bbox_optimization_20261018 \
  --output dinov3_ws/artifacts/depth_bbox_recheck.json
```

比較時保持相同背景負載；2026-10-18 的量測只有原相機驅動持續運作，無推論與 JPEG 負載。`--rounds` 和 `--repeats` 預設為 3 和 2，使用 12 張輸入時每種方案、每個情境有 72 個計時樣本。
