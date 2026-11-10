# dinov3_cases 程式導覽

目前部署入口是 `ros/node.py:main`，使用 SAM3 找候選物件，再以 DINOv3 特徵與 Lab 色彩直方圖比對圖庫。HTTP 網頁與 ROS 訂閱共用同一個節點狀態。

完整新結構、舊檔案對照與驗證結果見 [Python 套件重整紀錄](../../../docs/PYTHON_PACKAGE_REFACTOR.md)。

## 資料流程

1. `ros/node.py` 接收最新 RGB 影格，先配對深度，再取得 `frame_pool.py` 的共享影像租約。
2. `tracking/async_nvdcf.py` 決定同步偵測或背景偵測；`inference/grounding_ipc.py` 將影像交給同一份 `.venv` 的 TensorRT 11 子程序中的 `inference/sam3_trt.py`，使用官方 TensorRT vision 與 PyTorch bbox head。主程序使用 Cat2Real FP32 Torch-TensorRT。
3. 以 `grounding_detection_interval` 秒為最小啟動間隔提交背景 SAM3＋DINO 工作，0.0 表示空閒且有新影格時提交。前景 NvDCF 持續追蹤；背景 SAM3 出框後，先在偵測原圖建立候選 NvDCF，再補追最新快照，前景補追至共同影格後立即交接框；此時不等待 DINO。DINO 隨後辨識同一張 seed-frame 的裁切，只依不可變的 seed ownership key 更新仍存活物件的分類，不覆寫目前框。未識別物件一律只顯示 SAM3 名稱，包含等待 DINO 與完成後仍為 unknown 的情況，避免每次辨識時文字切換；已配對物件暫時沿用舊類別，不加上待辨識或更新中的字樣。交接逐物件配對：配對成功者由新 context 接手，漏偵測但追蹤有效者保留原 context 與標籤；SAM3 漏偵測不再淘汰原生追蹤仍有效的物件。靜態上傳每秒可刷新；所有間隔均維持 NvDCF，1.0 表示每秒一次偵測。
4. `tracking/identities.py` 平滑候選框並在共同影格關聯公開身分。已辨識物件變為 `unknown` 時移除該追蹤；初始 unknown 仍可顯示。NvDCF 原生 ID 用於 context 內的標籤綁定，公開 `track_id` 延續跨偵測回合的幾何關聯。
5. `inference/encoder.py` 將 Cat2Real 裁切圖置中補白至 384，從一次 forward 保留 pooler 描述向量與 patch 熱圖特徵；`recognition/grounded.py` 建立圖庫並執行單次辨識；分類結果由 `tracking/async_nvdcf.py` 綁定至追蹤物件並沿用。先以 `recognition_threshold` 篩選 DINO cosine 分數，再從候選參考圖中選 Lab Bhattacharyya 分數最高者的分類；無候選時為 unknown。`score` 為所選參考圖的 DINO 分數，不再計算加權分數。
6. `geometry/async_bbox3d.py` 獨立排程背景深度工作，透過 `geometry/depth_bbox.py` 計算中心、尺寸與旋轉；預設最小間隔約 0.167 秒；部署設定 0.0 表示每個新處理影格提交一次。ROS 發布和疊圖各用獨立 worker，待處理輸出只保留最新一份；共享影格池滿時，先釋放尚未開始的舊疊圖工作，讓新推論繼續。
7. `recognition/object_visualization.py` 在網頁要求時才產生物件診斷圖；`api/visualization.py` 負責 2D／3D 疊圖。

## 修改位置

### SAM3 區域過濾

`config/dinov3_cases.yaml` 的 `sam3_roi_width`、`sam3_roi_height`、
`sam3_roi_center_x`、`sam3_roi_center_y` 指定相機原圖像素範圍。
寬／高 `0` 代表該軸不限，兩者皆 `0` 停用過濾；中心 `-1` 跟隨畫面中點，
中心 `0` 為有效的左／上邊座標。超出畫面的區域取與原圖的交集。
SAM3 bbox 必須完整位於區域內才交給後續辨識；NvDCF 追蹤框移出區域後也不輸出。
完整影像、2D bbox 原圖座標、3D 深度與影像輸出皆維持原樣，上傳目標圖片不套用相機區域。

網頁 **Recognition settings → SAM3 region** 可即時調整四個參數，
按 **Apply region** 生效並清除舊結果；重啟後使用 YAML 初始值。
**Preview crop area** 只在目前瀏覽器的完整相機辨識畫面上疊半透明框，保留 bbox。
輸入時可先預覽尚未套用的值。預覽依原始相機尺寸、輸出影像尺寸和
`object-fit: contain` 留白換算位置，視窗改變大小時同步更新；不修改 ROS 影像或其他觀看者的畫面。

| 要修改的功能 | 主要檔案 |
| --- | --- |
| ROS 參數預設值與啟動驗證 | `config.py` |
| ROS 輸入、排程、發布 | `ros/node.py` |
| 網頁設定／來源切換與結果失效 | `recognition/recognition_state.py` |
| 分類分數、unknown 規則、單次辨識 | `recognition/grounded.py` |
| 模型輸入與推論 | `inference/encoder.py`、`inference/patches.py`、`inference/trt_backend.py`、`inference/sam3_trt.py` |
| 偵測間隔、背景補追與交接 | `tracking/async_nvdcf.py` |
| NvDCF 子程序、影像轉換與原生 API | `tracking/nvdcf.py`、`../../native/nvdcf_bridge.cpp` |
| 背景 3D 框間隔、快取、深度與旋轉 | `geometry/async_bbox3d.py`、`geometry/depth_bbox.py` |
| 圖庫路徑驗證、支援格式、檔案列舉、讀圖 | `recognition/gallery_files.py` |
| HTTP 狀態與辨識操作 | `api/web.py` |
| 圖庫上傳、相機裁切、HSV 去背 | `api/image_manager.py`、`recognition/images.py` |
| 診斷圖與比較參考圖 | `recognition/object_visualization.py` |
| 網頁互動與外觀 | `api/static/`、`api/templates/` |


測試命令見 [測試說明](../../tests/README.md)。

## 詳細說明

- [執行流程、生命週期、計時與 ROS Image／MJPEG](docs/runtime.md)
- [3D 深度框、點雲篩選、2D 邊界擬合與背景量測](docs/depth-bbox.md)
- [維護狀態與相容性](docs/maintenance.md)

`recognition/recognition_state.py` 集中網頁造成的辨識狀態更新；呼叫端必須持有
`node.lock`，並在更新前完成參數驗證或圖片解碼。更新設定／來源與
遞增 `generation` 必須在同一個鎖定區段內完成，避免發布舊工作的結果。


編碼介面統一為 `prepare_image()`、`encode_with_descriptor()` 與
`encode_regions()`。圖庫與單張效能診斷仍使用 `encode_object()`；
候選框辨識直接使用批次 `encode_regions()`。前處理使用 Cat2Real 置中補白與
OpenCV 雙線性縮放，分類描述向量使用 pooler 輸出；不再支援舊 `encode()`、
patch 平均向量後備路徑、`prepare_object()` 或 `run_dino_image()`。
