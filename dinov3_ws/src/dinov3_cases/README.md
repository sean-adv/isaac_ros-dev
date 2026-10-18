# dinov3_cases 程式導覽

目前部署入口是 `node.py:main`，使用 SAM3 找候選物件，再以 DINOv3 特徵與 Lab 色彩直方圖比對圖庫。HTTP 網頁與 ROS 訂閱共用同一個節點狀態。

## 資料流程

1. `node.py` 接收最新 RGB 影格，先配對深度，再取得 `frame_pool.py` 的共享影像租約。
2. `async_kcf.py` 決定同步偵測或背景偵測；`grounding_ipc.py` 將影像交給獨立 TensorRT 10 子程序中的 `sam3_trt.py`。主程序的 DINOv3 使用 TensorRT 11。
3. 非同步模式每 `grounding_detection_interval` 個已處理影格提交一次背景工作：SAM3 找框後，DINOv3 在同一張偵測影像上辨識；圖庫更新也在此背景工作完成。前景 KCF 持續追蹤舊集合。背景以偵測影像初始化新 tracker，再每次跳到當下最新已處理影格補追，追上後一起交接框、分類及診斷快照；部分追蹤遺失只移除該物件，不在前景重跑 DINO。追蹤遺失或接近 ROI 邊界可提早觸發整組工作；靜態上傳圖片每秒可刷新。間隔為 1 時停用 KCF，由推論 worker 依序執行兩個模型。
4. `tracking.py` 平滑候選框；可選的 ROI 追蹤另外管理 ID、命中比例及顯示門檻。
5. `grounded.py` 統一縮放裁切、建立圖庫、分類及管理辨識快取。相似度為 `0.8 × DINO cosine + 0.2 × Lab Bhattacharyya`，再依相似度及類別分差決定是否為 unknown。
6. `depth_bbox.py` 計算相機座標下的 3D 框。ROS 發布和疊圖各用獨立 worker，待處理輸出只保留最新一份；共享影格池滿時，先釋放尚未開始的舊疊圖工作，讓新推論繼續。
7. `object_visualization.py` 在網頁要求時才產生物件診斷圖；`visualization.py` 負責 2D／3D 疊圖。

## 修改位置

| 要修改的功能 | 主要檔案 |
| --- | --- |
| ROS 參數、排程、發布 | `node.py` |
| 分類分數、unknown 規則、辨識快取 | `grounded.py` |
| 模型輸入與推論 | `encoder.py`、`patches.py`、`trt_backend.py`、`sam3_trt.py` |
| 偵測間隔、背景回放、KCF 遺失處理 | `async_kcf.py`、`kcf.py` |
| 圖庫路徑驗證、支援格式、檔案列舉、讀圖 | `gallery_files.py` |
| HTTP 狀態與辨識操作 | `web.py` |
| 圖庫上傳、相機裁切、HSV 去背 | `image_manager.py` |
| 診斷圖與比較參考圖 | `object_visualization.py` |
| 網頁互動與外觀 | `static/`、`templates/` |

`grounded.py` 保留原有診斷函式的匯入入口，既有整合工具仍可使用；新程式可直接從 `object_visualization.py` 匯入。

## 必須維持的生命週期

- SAM3 框屬於較舊的影格；在最新影格初始化是位置近似，偵測期間物體快速移動可能使新 tracker 追錯區域。
- 相同影格重算不能增加追蹤命中次數或重複累加平滑樣本。
- KCF 組更換或物件數改變時重新編碼；圖庫或分類門檻改變時可用既有 descriptor 重新分類。
- 診斷裁切屬於辨識當時的影格，不可套用後續 KCF 框冒充目前影像。
- 共享影格在所有租約釋放前不可覆寫；舊 generation 的背景結果不可發布。
- 診斷快照保留完整舊圖庫版本，避免更新圖庫時混用特徵與標籤。

測試命令見 `../../tests/README.md`。重構前已存在的限制：`qwen_encoder.py` 引用 `runtime.py` 中不存在的 `QWEN_*` 設定，導致 `test_qwen_encoder.py` 無法收集；目前主節點僅接入 DINOv3。Qwen 程式與既有修改予以保留，未在此次整理中刪除或補上另一套推論流程。

網頁 `result FPS` 是完整辨識主迴圈的結果頻率，相機的 `input FPS` 是輸入回呼頻率。主迴圈還包含深度同步與 3D 框；非同步模式的 DINOv3 在背景執行，`recognize_ms` 記錄交接結果的背景辨識耗時；細分計時和影像輸出方式見 `../../BUILD_GUIDE.md`。

啟動或套用設定後，第一份結果會等待同一張 RGB 影格的 SAM3、DINOv3 與 KCF 初始化完成，再恢復持續追蹤；相機擷取與網頁預覽仍獨立運作。同步深度會在等待推論前保留。這可避免首次初始化等待期間持續補追而反覆超限；模型載入完成後，網頁會顯示準備首次偵測的狀態。`/api/status` 的 `grounding_schedule` 可查看目前初始化、背景工作及最近一次補追丟棄原因。
