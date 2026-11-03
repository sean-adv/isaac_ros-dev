# 目前系統功能地圖

盤點日期：2026-11-03。基準為本次任務開始時的工作區（包含未提交的 encoder／測試修改），
不是歷史 audit 所列功能的聯集。依據是啟動入口、實際路由／模板／JavaScript、
節點排程、子程序與原生載入、部署設定，以及完整回歸測試。
目前環境的測試證據不等於真實相機與完整模型端到端驗證。

以下程式碼路徑除特別註明外，相對 repository 根目錄；應用模組位於
`dinov3_ws/src/dinov3_cases/dinov3_cases/`。

## 使用者功能與 HTTP 介面

系統只有兩份 HTML 模板，首頁與圖庫互相連結；沒有另外一套舊前端路由。
Flask 由 ROS 節點內的 threaded Werkzeug server 提供，預設 `127.0.0.1:8090`。

| 頁面／操作 | 資料流與介面 | 程式碼 |
| --- | --- | --- |
| `/` 即時辨識 | 顯示相機／上傳圖片、偵測分類與公開 track ID、3D 狀態、輸入／處理 FPS、模型耗時 | `templates/cases.html`、`static/cases.js`、`api/web.py` |
| 文字 prompt、門檻與偵測間隔 | `POST /api/grounding`；更新設定並使舊 generation 結果失效；間隔 >1 檢查 NvDCF | `api/web.py`、`recognition/recognition_state.py`、`recognition/grounded.py` |
| SAM3 區域套用／重設／本機預覽 | `GET/POST /api/sam3-roi`；完整原圖座標、只留下完全包含的框；預覽只影響瀏覽器 | `inference/sam3_roi.py`、`static/sam3_roi.js` |
| 切換 ROS 相機／上傳目標 | `POST /api/source` multipart；上傳解碼後才切換狀態 | `api/web.py`、`recognition/recognition_state.py` |
| 新增參考圖、選擇瀏覽圖庫 | `POST /api/cases`、`POST /api/select`；選擇僅影響圖庫瀏覽，分類仍使用整個 gallery | `api/web.py`、`grounded.ObjectGallery` |
| 參考圖 patch 網格 | `GET /api/references?case=…`、`GET /api/references/grid/<group>/<filename>` | `api/web.py`、`inference/patches.py`、`inference/cat2real.py` |
| 偵測列表與狀態輪詢 | `GET /api/status`、`GET /api/detections`；狀態含 worker／排程／錯誤／SAM3 事件 | `api/web.py`、`ros/node.py` |
| SAM3 推論間隔圖與清除顯示歷史 | 使用 status 的 `sam3_inference_events`；清除只作用於目前瀏覽器 | `static/sam3_intervals.js` |
| 按需物件診斷、切換檢視／參考分類 | `GET /api/object_visualizations?reference_case=…`、`GET /debug/objects/<revision>/<index>/<view>.jpg`；crop/grid/patches/similarity/heatmap/reference | `recognition/object_visualization.py`、`api/web.py` |
| `/images` 圖庫管理 | 分類建立／選擇、上傳、重新整理、圖片選取與刪除 | `templates/images.html`、`static/images.js`、`api/image_manager.py` |
| 相機 SAM3 框預覽與擷取 | `POST /api/images/camera-preview`、`POST /api/images/capture`；同一張原始 RGB 送 SAM3 並裁切儲存，套用相機 ROI | `web.capture_camera`、`api/image_manager.py` |
| HSV 點選去背 | `POST /api/images/preview` 預覽，`POST /api/images/cutout` 另存透明 PNG；支援色相／飽和／明度／羽化／連通區 | `image_manager.remove_background` |
| 共用即時影像 | `GET /image_annotated.mjpg`；兩個頁面共用 annotated ROS Image、背景 JPEG、斷線重連 | `api/image_stream.py`、`static/image_stream.js` |

圖庫其餘必要介面：`GET /api/images?case=…`、`POST /api/images/labels`、
`POST /api/images/upload`、`GET/DELETE /api/images/file/<group>/<filename>`。
`/static/*` 由 Flask 提供。所有現用 API 均保留原本路徑、欄位與行為。
圖片路徑驗證、禁止 symlink、EXIF 方向與格式篩選集中於 `recognition/gallery_files.py`。
圖片可寫資料是 `dinov3_ws/data/gallery/`，不可當作 dead code 清除。

## 主要運算流程

1. `node.CaseNode` 接收最新 RGB、registered depth 與 CameraInfo；
   `geometry/depth_sync.py` 按時間與尺寸配對深度，`frame_pool.py` 管理共享影格租約。
2. `grounding_ipc.SharedMemoryGroundingDetector` 以同一 Python 環境啟動
   `python -m dinov3_cases.inference.grounding_ipc` 子程序。影像經共享記憶體、指令經 socket 傳遞；
   `sam3_trt.Sam3Detector` 使用官方 SAM3 TensorRT FP16 vision 與 PyTorch bbox/text head。
   文字支援 1–4 類別；沒有 mask 或 SAM3 tracking 流程。
3. 間隔 1 為同步 SAM3／辨識；間隔 >1 由 `tracking/async_nvdcf.py` 單一背景工作執行偵測，
   在 seed frame 建立 NvDCF，再補追至共同影格並先交接框。前景 NvDCF 持續追蹤；
   DINO 分類稍後只更新仍存活的 seed owner，不覆寫最新框。
4. `inference/encoder.py` 使用 Cat2Real FP32 pooler descriptor 與 patch features。
   圖庫單張編碼使用 `encode_object`；候選框走 `encode_regions`，同影格微批次，
   不等待其他影格。前處理為 384×384 置中白色補邊；CUDA ROI 與 CPU 圖庫路徑皆必要。
5. `grounded.ObjectGallery` 依檔名／mtime／size 更新完整圖庫。
   分類先套用 DINO cosine 門檻，再選 Lab 相似度最高的參考圖；無候選輸出 unknown。
   回傳 `score` 是所選參考圖 DINO 分數，`margin` 是不同候選分類的 Lab gap。
6. `tracking/identities.py` 維持公開身分、幾何配對與平滑；NvDCF 保留漏偵測／失追寬限。
   失追期間隱藏舊框、避免提交舊框量測；重獲後可沿用 ID／分類。
7. `geometry/async_bbox3d.py` 獨立背景量測，`geometry/depth_bbox.py` 與 `geometry/cloud_filter.py`
   做深度連通區、單平面 RANSAC、平面內方向與 2D 投影擬合。
   `gpu_geometry.py` 的 CUDA 及 CPU Open3D 路徑都由現用設定選擇。
8. 結構化 ROS 發布、疊圖、Image 發布、JPEG 轉換各有背景工作；pending 只保留最新資料。
   未被觀看時不繪圖／轉 JPEG。`ObjectVisualizationSnapshot` 按需產生診斷圖，
   保留辨識當時完整圖庫版本，避免圖庫更新混用資料。

來源／設定／圖庫變動在 `node.lock` 內遞增 generation 並清除舊輸出。
背景工作不得發布舊 generation，租約釋放前不得覆寫共享影格。
相同影格的背景刷新不增加追蹤命中或 processed RGB FPS。

## ROS 2 介面

節點名為 `dinov3_cases`，可用 ROS arguments／YAML 覆寫參數。
沒有應用自訂 ROS Service／Action；HTTP 操作直接更新同一節點狀態。

| 方向 | 預設 topic | 型別／行為 |
| --- | --- | --- |
| 訂閱 | `/camera/color/image_raw` | `sensor_msgs/Image`，Reliable KeepLast(1) |
| 訂閱 | `/camera/depth/image_raw` | `sensor_msgs/Image`，pose 啟用時；16UC1 mm／32FC1 m |
| 訂閱 | `/camera/depth/camera_info` | `sensor_msgs/CameraInfo`，pose 啟用時 |
| 發布 | `/medicine_recognition/detections` | `vision_msgs/Detection2DArray` |
| 發布 | `/medicine_recognition/detections_3d` | `vision_msgs/Detection3DArray` |
| 發布 | `/tag_detections` | `isaac_ros_apriltag_interfaces/AprilTagDetectionArray`，由辨識框轉換，並非另跑 AprilTag detector |
| 發布 | `/medicine_recognition/model_timings` | `std_msgs/String` JSON：`nvdcf_ms`、`dinov3_ms`、`sam3_ms`；5 Hz |
| 發布並由網頁 bridge 訂閱 | `/medicine_recognition/image_annotated` | `sensor_msgs/Image`，bgr8、960×540、保留來源 header，Reliable KeepLast(1) |

## 入口、部署與動態依賴

| 位置 | 責任 |
| --- | --- |
| `start.sh`、`compose.yaml`、`.devcontainer/` | Jetson host 的 NVIDIA 開發容器；host network/IPC、USB、workspace volume；容器保持存活，需另啟動相機／辨識 |
| 根 `run.sh` | 容器內轉送應用；host 經 docker exec。預設既有 container/path 與 Compose 不同，外部使用情況不明，保留 |
| 根 `run_camera.sh` → `dinov3_ws/run_camera.sh` | Orbbec Gemini 330 外部 ROS launch；可用 `ORBBEC_SETUP`，否則搜尋已安裝套件／`orbbec_sdk/install` |
| `dinov3_ws/run.sh` → `launch_pipeline.py` → `node:main` | 載入 ROS、venv、LD_LIBRARY_PATH、DDS、thread defaults；載入 YAML、模型預設並轉送 ROS arguments |
| `src/dinov3_cases/setup.py`、`package.xml` | ament／console entry `case_recognition`；打包 Python、static、templates；wheel 單獨不包含 workspace config／模型 |
| `setup_all.sh`、`setup_env.sh` | Thor ARM64 系統依賴、Python 環境、native 建置；不自動下載／重建模型 |
| `build_models.py`、`inference/model_modules.py` | 取得固定 revision 權重、編譯與數值驗證；runtime 也使用 model wrappers |
| `native/build.sh`、`CMakeLists.txt` | 唯一原生建置流程，targets `all`／`gpu_geometry`／`nvdcf_bridge` |
| `native/install_deps.sh`、`Dockerfile` | DeepStream 9.1／編譯依賴與原生 runtime image |
| `config/` | ROS 部署覆寫、Fast DDS 大影像 localhost transport、NvDCF 原生設定 |

必要的非直接 import 依賴：SAM3 Python 子程序；NvDCF C++ 子程序（`tracking/nvdcf.py`）；
`ctypes.CDLL` CUDA helper（`gpu_geometry.py`）；DeepStream tracker `dlopen`（native bridge）；
Torch-TensorRT import 的反序列化 operator 註冊；本機 engine 與同名 JSON metadata；
processor/tokenizer source assets；ROS 訊息與外部 Orbbec launch。
`runtime.py` 依 metadata 選 batch8／native vision，缺少時用已支援的單筆引擎，並非可刪相容層。

Python runtime／模型建置共用 `requirements.txt` 與 `constraints-thor.txt`；
`requirements-dev.txt` 額外提供 pytest。Torch／TensorRT 的版本關係不可只看直接 import 刪除。
系統提供 ROS Jazzy、CUDA、DeepStream、相機與訊息套件。

主要環境變數：`DINOV3_VENV`、`PYTHON_BIN`、`ORBBEC_SETUP`、
`FASTDDS_DEFAULT_PROFILES_FILE`／`FASTRTPS_DEFAULT_PROFILES_FILE`、`ROS_DOMAIN_ID`、
`OMP_NUM_THREADS`／`OPENBLAS_NUM_THREADS`／`MKL_NUM_THREADS`；
建置使用 `CUDA_ARCH`、`DEEPSTREAM_DIR`、`NATIVE_BUILD_DIR`、`DEEPSTREAM_DEB`；
host 使用 `HOST_WORKSPACE`、`HOST_USER_UID/GID`、`DISPLAY`、`XAUTHORITY` 及根 wrapper 的
`DINOV3_CONTAINER*`。相機／辨識入口強制 localhost discovery 並清除遠端 peers/server。

## 實際部署設定與程式預設

`config.py` 是內建預設；`config/dinov3_cases.yaml` 是目前啟動覆寫，兩者用途不同。
目前 YAML：prompt `box, can.`、SAM3 threshold 0.8、containment 0.5、DINO threshold 0.75、
偵測間隔 3、NvDCF 640×384、miss detections 5／frames 15、平滑 1、
ROI 458×420 中心 (370,200)、3D interval 1／max age 1.0、depth 0.2–1.0 m、
quantile 0.10、CUDA plane backend、3D overlay、video_fps 30。
網頁設定只更新執行狀態，重啟後仍用 YAML。

`recognition_margin`／HTTP `margin` 目前仍出現在 UI／API 設定契約，雖不參與分類，
本次保留以符合現用 UI／介面不變要求；不與診斷輸出的 Lab gap 混淆。

## 測試與文件邊界

`tests/test_*.py` 覆蓋現用分類、圖庫、UI HTTP、共享影格、IPC、ROS Image、
非同步交接／失追、3D／CUDA 與排程；沒有因歷史命名刪除現用測試。
`test_images_ui.cjs`、`test_sam3_roi_ui.cjs` 是 Node.js 互動測試。
`ipc_fake_worker.py`、`encoder_stub.py` 是必要測試替身。
`check_*.py` 是 GPU／完整模型 smoke；`profile_*.py` 是相機、模型、深度與服務診斷，
即使無 runtime import 仍有明確開發用途。

本次清理前完整 pytest：470 passed（14.38 秒），無失敗／skip／xfail。
詳細分類、清理變更與驗證結果見 [CLEANUP_REPORT.md](CLEANUP_REPORT.md)。
操作／模型建置分別保留在根 README、`dinov3_ws/README.md`、`BUILD_GUIDE.md`、
`DOCKER_HOST.md`；runtime／depth 專題保留於套件 docs。根 `REPOSITORY_AUDIT.md`
是歷史記錄，不能用其中舊 API、失敗數或架構作目前需求。

## 工作區外圍資源

`orbbec_sdk/` 是現用相機 fallback。`DART/`、`detectron2/`、`dinov3_tensorrt/`、
`OmDet/`、`TensorRT-Model-Connect/`、`sam2_trt_inference/` 等相鄰 repository
沒有出現在本應用啟動依賴中，但是否仍用於其他工作流程不明。
外部 repository、舊 venv、`copy/`、`trash/`、模型、資料與 `.git` history 全部保留。
本次不將它們視為已確認淘汰，也不刪除大型資產來製造清理成效。
