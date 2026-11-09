# 執行流程、排程與影像輸出

[回程式導覽](../README.md) · [3D 深度框](depth-bbox.md)

## 模型耗時 topic 與網頁

網頁 Live view 顯示各階段最近一次成功完成的毫秒數。
`/api/status` 的 `model_timings` 與 ROS topic `/medicine_recognition/model_timings`
共用量測資料。topic 使用 `std_msgs/msg/String` JSON，5 Hz、Reliable、KeepLast(1)、Volatile。

```bash
ros2 topic echo /medicine_recognition/model_timings std_msgs/msg/String
```

topic 的 JSON 只有三個數值欄位，單位固定為 ms：

```json
{"nvdcf_ms": 23.3, "dinov3_ms": 205.7, "sam3_ms": 219.8}
```

`/api/status` 的 `model_timings` 使用相同格式。尚無成功樣本時數值為 `null`，
網頁顯示 `—`；快取結果與執行中保留最近成功值。
來源或設定改變時清空，忽略舊工作的遲到結果。

計時均為 wall time：NvDCF 是目前 live contexts 整體追蹤更新（含前處理與 IPC，
不含背景 candidate 初始化／補追）；DINOv3 是整批圖庫刷新、辨識及診斷資料建立；
SAM3 是 `worker_detect_ms`（不含主程序 IPC）。這些不是純 GPU kernel 時間，
非同步階段不可直接相加作為總延遲。間隔 1 不執行 NvDCF，該階段顯示 `—`。

## 必須維持的生命週期

- SAM3 框屬於較舊的影格；須在偵測原圖初始化，再補追至當前影格，才可交接。大幅跳幀仍可能追錯。
- 相同影格重算不能增加追蹤命中次數或重複累加平滑樣本。
- 相機持續送影格時，SAM3／DINO／3D 的背景完成通知合併到下一張 RGB，避免同影格再複製、追蹤、發布及畫圖編碼。若超過 200 ms 沒有新輸入，仍可刷新該影格，讓首次 3D 與延後類別完成；靜態上傳可立即套用背景結果。這類刷新不增加 processed RGB FPS。
- 同步模式每次直接執行單次辨識；非同步模式由 NvDCF 沿用分類，新的辨識結果只依 seed ownership 更新仍存活物件。
- 診斷裁切屬於辨識當時的影格，不可套用後續 NvDCF 框冒充目前影像。
- 共享影格在所有租約釋放前不可覆寫；舊 generation 的背景結果不可發布。
- 診斷快照保留完整舊圖庫版本，避免更新圖庫時混用特徵與標籤。


網頁 `processed RGB FPS`（API `result_fps`）只計不同 `(generation, frame_id)` 的已完成 RGB，不把同影格的背景刷新算成新影格；相機的 `input FPS` 是輸入回呼頻率。主迴圈只保留可配對的深度快照並提交背景 3D 工作，不等待深度或量測完成；非同步模式的 DINOv3 在背景執行，`recognize_ms` 記錄交接結果的背景辨識耗時；細分計時和影像輸出方式見 [建置與執行說明](../../../BUILD_GUIDE.md)。

啟動或套用設定後，非同步模式的第一份框等待 SAM3 與 NvDCF 初始化完成便開始持續追蹤，DINO 類別稍後更新；間隔 1 的同步模式仍等待完整辨識；相機擷取與網頁預覽仍獨立運作。同步深度會在等待推論前保留。這可避免首次初始化等待期間持續補追而反覆超限；模型載入完成後，網頁會顯示準備首次偵測的狀態。`/api/status` 的 `grounding_schedule` 可查看目前初始化、背景工作及最近一次補追丟棄原因。

### 共用 ROS Image 即時影像

單一疊圖輸出為 `/medicine_recognition/image_annotated`（`sensor_msgs/msg/Image`，
`bgr8`、960×540、Reliable、KeepLast(1)、Volatile），保留來源 RGB 的 header。
`annotated_image_topic` 可覆寫 topic；`video_fps` 控制最大輸出頻率（1–30）。
疊圖與 ROS 發布各用獨立 worker，待處理工作只保留最新一份，不讓慢速 DDS 阻塞推論。
無網頁觀看者且無外部 Image 訂閱者時不繪圖。

首頁與圖庫頁都使用 `<img src="/image_annotated.mjpg">`。HTTP bridge 實際訂閱同一個
annotated Image topic，背景 worker 將最新影格轉為 JPEG（quality=90），以 MJPEG 提供給網頁。
所有觀看者共用同一份 JPEG；每個接收端只取最新完成影格，不建立逐幀佇列。
無網頁觀看者時不做 JPEG 轉換；靜態來源可重送快取 JPEG，無需重新編碼。
連線關閉或輸出錯誤會回收觀看者；圖片播放器會在網路錯誤後重連。
HTTP 經反向代理時應停用此路徑的 response buffering。

3D 模式等待有效深度量測及相機參數，不暫時畫 2D 框；DINO 尚未完成時可先顯示
已就緒 3D 框和 SAM3 名稱。相機辨識結果超過 3 秒或無可見框時顯示原始相機預覽；
上傳圖片不過期，沒有 ROS header 時使用繪圖當下節點時間。
圖庫手動擷取仍使用原始 RGB，ROI 預覽依 `<img>` 的 natural size 及顯示留白換算。

RViz2 使用 **Image** display，選擇上述 topic，Transport Hint=`raw`、Reliability=`Reliable`，
Queue Size 建議為 1。結構化 Detection2D／Detection3D、AprilTag topic、物件診斷及圖庫圖片保持原有用途。

網頁圖片串流已取代先前的影片傳輸；舊影片封包 topic、協商／peer API 及編碼設定已移除。
MJPEG 與 raw Image 的頻寬可能較高；實際 CPU、頻寬及端到端延遲需在目標場景量測。

## NvDCF 執行環境

NvDCF 每次更新後，Python 輸出端也會檢查 `nvdcf.yaml` 的 `TargetManagement.minTrackerConfidence`（啟動時讀取）。沒有原生輸出、信心低於門檻或信心非有限值時，該框立即隱藏，不等待 SAM3 結果；即使同次有偵測輸入也適用。隱藏軌跡沿用下述保留與恢復規則。若 NvDCF 對背景仍有高於門檻的信心，此規則無法判定物體實際已離開。

已解除所有權的 NvDCF context 由單一背景 worker 回收，交接不等待舊子程序退出；最多排隊 8 個，滿載時施加背壓，停機時等待全部回收。仍有可見或遮擋保留物件的 context 不會提前回收。候選框補追到共同影格後先交接，seed-frame 的分類完成後再更新；延後分類不能重新建立已移除的物件，也不能貼到新的偵測回合。`grounding_schedule` 另外提供 `candidate_create_ms`、`candidate_seed_ms`、`candidate_catchup_ms`，用來區分候選準備成本；`geometry_ready_ms` 是框就緒時間，`job_ms` 在 DINO 完成後更新為整件背景工作時間。DINO 失敗保留框，錯誤可從 `recognition_error` 查看。

短暫遮擋使 NvDCF 不再輸出物件時，`nvdcf_max_missed_frames` 保留其原 context、公開 ID 與分類 metadata（預設 30，設 0 停用）。以不同的已處理 RGB 影格計數，同影格重跑不累計；容忍 N 次連續缺失，第 N+1 次仍缺失才移除。失追期間不顯示過期框，也不提交舊框做 3D 量測，該物件的 SAM3 miss 計數暫停。原生輸出恢復時沿用原生及公開 ID；若由新 SAM3 結果重新配對，保留公開 ID，但可換用新的原生 context。任何物件剛失追時會提前要求 SAM3；仍有隱藏物件時，以 `min(grounding_detection_interval, max(1, nvdcf_max_missed_frames // 3))` 個處理影格重試，預設最多每 10 幀一次。原生 shadow 存活期另外受 `maxShadowTrackingAge` 限制，跨遮擋重新配對使用幾何位置而非 ReID，無法保證大幅位移後仍配對正確。包含框過濾與已知轉 unknown 的立即退場規則仍適用於可見物件。

原生追蹤器使用 DeepStream 9.1，執行於獨立 C++ 子程序，不載入模型 `.venv` 的 TensorRT 11。安裝與參數見 [BUILD_GUIDE.md](../../../BUILD_GUIDE.md)。GPU smoke test：

```bash
PYTHONPATH=dinov3_ws/src/dinov3_cases dinov3_ws/.venv/bin/python dinov3_ws/tests/check_nvdcf.py
PYTHONPATH=dinov3_ws/src/dinov3_cases dinov3_ws/.venv/bin/python dinov3_ws/tests/check_nvdcf_handoff.py
PYTHONPATH=dinov3_ws/src/dinov3_cases dinov3_ws/.venv/bin/python dinov3_ws/tests/check_nvdcf_occlusion.py
```
