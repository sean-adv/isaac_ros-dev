# Python package 重整紀錄

日期：2026-11-03。修改前完整回歸：472 passed（14.25 秒）。

## 範圍與原始責任

掃描整個 workspace 的目錄與受 Git 管理檔案；維護範圍是本 repository 的 78 個 Python 檔案。相鄰外部 repositories、venv、模型、build/install、copy/trash 與 Orbbec SDK 均為忽略的外部或產生資料，不改寫其結構。以下責任依來源內容、函式及匯入分析整理。

| 原始檔案 | 實際責任／測試範圍 | 新位置 |
| --- | --- | --- |
| `dinov3_ws/build_models.py` | Build/validate official Torch-TensorRT SAM3 and Cat2Real without sibling repos. | 不變 |
| `dinov3_ws/launch_pipeline.py` | Run text-guided object detection and DINOv3 gallery recognition. | 不變 |
| `dinov3_ws/src/dinov3_cases/dinov3_cases/__init__.py` | 套件標記 | 不變 |
| `dinov3_ws/src/dinov3_cases/dinov3_cases/apriltag.py` | Publish RGB-D objects using the Isaac ROS AprilTag contract. | `ros/messages.py` |
| `dinov3_ws/src/dinov3_cases/dinov3_cases/async_bbox3d.py` | Periodic full depth-box measurements with one active and one latest job. | `geometry/async_bbox3d.py` |
| `dinov3_ws/src/dinov3_cases/dinov3_cases/async_nvdcf.py` | Whole-frame NvDCF handoff, optionally publishing boxes before DINO labels. | `tracking/async_nvdcf.py` |
| `dinov3_ws/src/dinov3_cases/dinov3_cases/cat2real.py` | Cat2Real centered white-square preparation followed by OpenCV bilinear resize. | `inference/cat2real.py` |
| `dinov3_ws/src/dinov3_cases/dinov3_cases/cloud_filter.py` | Reject unreliable depth components before plane fitting. | `geometry/cloud_filter.py` |
| `dinov3_ws/src/dinov3_cases/dinov3_cases/config.py` | Startup parameter defaults and validation, before workers are started. | 不變 |
| `dinov3_ws/src/dinov3_cases/dinov3_cases/depth_bbox.py` | Plane-oriented depth bounds and final detector-constrained cuboids. | `geometry/depth_bbox.py`；ROS 訊息函式抽至 `ros/messages.py` |
| `dinov3_ws/src/dinov3_cases/dinov3_cases/depth_sync.py` | Match registered RGB-D frames and camera intrinsics by size, frame and time. | `geometry/depth_sync.py` |
| `dinov3_ws/src/dinov3_cases/dinov3_cases/encoder.py` | Cat2Real pooler descriptor and spatial features from one TRT forward. | `inference/encoder.py` |
| `dinov3_ws/src/dinov3_cases/dinov3_cases/frame_pool.py` | Bounded RGB frame storage shared by inference and asynchronous rendering. | 不變 |
| `dinov3_ws/src/dinov3_cases/dinov3_cases/gallery_files.py` | Shared gallery paths, image decoding and deterministic file selection. | `recognition/gallery_files.py` |
| `dinov3_ws/src/dinov3_cases/dinov3_cases/gpu_geometry.py` | CUDA geometry ABI, independent of the TensorRT and DeepStream processes. | 不變 |
| `dinov3_ws/src/dinov3_cases/dinov3_cases/grounded.py` | Text-guided proposals followed by per-object DINOv3 gallery retrieval. | `recognition/grounded.py`；ROS 訊息函式抽至 `ros/messages.py` |
| `dinov3_ws/src/dinov3_cases/dinov3_cases/grounding_ipc.py` | Leased RGB frames in shared memory; metadata over a private socketpair. | `inference/grounding_ipc.py` |
| `dinov3_ws/src/dinov3_cases/dinov3_cases/image_manager.py` | Local image gallery and HSV background removal, independent of ROS/model loading. | `api/image_manager.py`；HSV 處理抽至 `recognition/images.py` |
| `dinov3_ws/src/dinov3_cases/dinov3_cases/image_stream.py` | Latest-only ROS Image to shared MJPEG; JPEG encoding stays off the ROS executor. | `api/image_stream.py` |
| `dinov3_ws/src/dinov3_cases/dinov3_cases/model_modules.py` | Tensor-only wrappers around official Hugging Face model modules. | `inference/model_modules.py` |
| `dinov3_ws/src/dinov3_cases/dinov3_cases/node.py` | ROS input, inference scheduling and result publication with official Torch-TensorRT models. | `ros/node.py` |
| `dinov3_ws/src/dinov3_cases/dinov3_cases/nvdcf.py` | DeepStream 9.1 NvDCF in a native process isolated from Torch/TensorRT libraries. | `tracking/nvdcf.py` |
| `dinov3_ws/src/dinov3_cases/dinov3_cases/object_visualization.py` | On-demand object diagnostics and bounded immutable result snapshots. | `recognition/object_visualization.py` |
| `dinov3_ws/src/dinov3_cases/dinov3_cases/patches.py` | DINOv3 patch normalization, matching and reference-grid rendering. | `inference/patches.py` |
| `dinov3_ws/src/dinov3_cases/dinov3_cases/recognition_state.py` | Recognition state transitions shared by HTTP controls. | `recognition/recognition_state.py` |
| `dinov3_ws/src/dinov3_cases/dinov3_cases/runtime.py` | Self-contained model/data paths for the official Torch-TensorRT deployment. | 不變 |
| `dinov3_ws/src/dinov3_cases/dinov3_cases/sam3_roi.py` | Filter full-image boxes by a configurable camera region, without cropping pixels. | `inference/sam3_roi.py` |
| `dinov3_ws/src/dinov3_cases/dinov3_cases/sam3_trt.py` | Official Torch-TensorRT SAM3 vision plus Hugging Face PyTorch bbox head. | `inference/sam3_trt.py` |
| `dinov3_ws/src/dinov3_cases/dinov3_cases/timing.py` | Rolling throughput measured from completed inference results. | 不變 |
| `dinov3_ws/src/dinov3_cases/dinov3_cases/tracking.py` | Geometric history matching and moving averages for SAM3/NvDCF boxes. | `tracking/identities.py` |
| `dinov3_ws/src/dinov3_cases/dinov3_cases/trt_backend.py` | Load the official Torch-TensorRT Cat2Real artifact. | `inference/trt_backend.py` |
| `dinov3_ws/src/dinov3_cases/dinov3_cases/visualization.py` | 2D rectangles and projected 3D box wireframes on a matching RGB frame. | `api/visualization.py` |
| `dinov3_ws/src/dinov3_cases/dinov3_cases/web.py` | HTTP routes for the shared recognition node; inference stays in node.py. | `api/web.py` |
| `dinov3_ws/src/dinov3_cases/setup.py` | setuptools / ament 資源與 console script 安裝 | 不變 |
| `dinov3_ws/tests/check_async_nvdcf_pipeline.py` | Real models: geometry precedes DINO; tracking continues through either stall. | 不變 |
| `dinov3_ws/tests/check_gpu_environment.py` | Report CUDA driver loading and check PyTorch and TensorRT initialization. | 不變 |
| `dinov3_ws/tests/check_grounded_trt_pipeline.py` | Official-model ROS/HTTP smoke, including a deliberately stalled renderer. | 不變 |
| `dinov3_ws/tests/check_nvdcf.py` | Real GPU smoke test: tracking motion, metadata, IDs, and TensorRT isolation. | 不變 |
| `dinov3_ws/tests/check_nvdcf_handoff.py` | Real GPU check: partial refresh retains native state until three SAM3 misses. | 不變 |
| `dinov3_ws/tests/check_nvdcf_occlusion.py` | Real GPU: full occlusion preserves owners, SAM3 refresh restores public ID. | 不變 |
| `dinov3_ws/tests/check_official_pipeline.py` | GPU/IPC regression using only models and packages inside dinov3_ws. | 不變 |
| `dinov3_ws/tests/encoder_stub.py` | CPU test double for the current encoder protocol; no runtime fallback support. | 不變 |
| `dinov3_ws/tests/ipc_fake_worker.py` | CPU fixture for shared-memory frame identity and worker failure tests. | 不變 |
| `dinov3_ws/tests/profile_camera_input.py` | Measure camera delivery alone, with no inference or image copying. | 不變 |
| `dinov3_ws/tests/profile_camera_pipeline.py` | Profile the real camera pipeline without changing production inference code. | 不變 |
| `dinov3_ws/tests/profile_depth_bbox.py` | Compare saved pre-change 3D code, ROI conversion, and cached rays on identical depth frames. | 不變 |
| `dinov3_ws/tests/profile_gpu_pipeline.py` | Compare identical saved RGB-D inputs on legacy and CUDA/native paths. | 不變 |
| `dinov3_ws/tests/profile_grounding_stages.py` | Measure current SAM3 worker and shared-memory transport on a saved frame. | 不變 |
| `dinov3_ws/tests/profile_live_localhost.py` | Sample a running pipeline and its MJPEG output without loading models. | 不變 |
| `dinov3_ws/tests/test_apriltag.py` | Isaac ROS output must survive serialization with the RGB-D contract intact. | 不變 |
| `dinov3_ws/tests/test_async_bbox3d.py` | Independent frame cadence, bounded jobs and coherent cached 3D measurements. | 不變 |
| `dinov3_ws/tests/test_async_bbox3d_node.py` | The node's 3D cadence stays independent of SAM3 and never waits for depth work. | 不變 |
| `dinov3_ws/tests/test_async_nvdcf.py` | Timing and identity contracts without requiring DeepStream on a test host. | 不變 |
| `dinov3_ws/tests/test_bbox3d_fit.py` | Final cuboids match detector pixels while retaining orthogonal local axes. | 不變 |
| `dinov3_ws/tests/test_bbox_visualization.py` | Calibrated wireframe projection and configuration-selected web overlays. | 不變 |
| `dinov3_ws/tests/test_camera_capture.py` | Manual capture uses the chosen prompt and crops the exact raw camera frame. | 不變 |
| `dinov3_ws/tests/test_cat2real.py` | Cat2Real must use one pooler forward and map centered padding correctly. | 不變 |
| `dinov3_ws/tests/test_cloud_filter.py` | Coverage and temporal rejection must happen before any RANSAC call. | 不變 |
| `dinov3_ws/tests/test_config.py` | Startup configuration validation without constructing a ROS node or workers. | 不變 |
| `dinov3_ws/tests/test_depth_bbox.py` | Robust foreground bounds, calibration and the standard ROS detection contract. | 不變 |
| `dinov3_ws/tests/test_depth_sync.py` | RGB-D synchronization, depth units and inference snapshot lifetime. | 不變 |
| `dinov3_ws/tests/test_frame_pool.py` | 共享影格租約、不可覆寫與延遲釋放 | 不變 |
| `dinov3_ws/tests/test_gpu_acceleration.py` | Actual CUDA kernels and native IPC; skipped when the native helpers are absent. | 不變 |
| `dinov3_ws/tests/test_grounded.py` | Object identity, rejection, duplicate proposals, and gallery lifecycle contracts. | 不變 |
| `dinov3_ws/tests/test_grounding_ipc.py` | Exercise real subprocess IPC, SHM ownership, sequence checks and cancellation. | 不變 |
| `dinov3_ws/tests/test_image_manager.py` | HSV 去背、上傳、圖庫路徑與 EXIF 回歸 | 不變 |
| `dinov3_ws/tests/test_image_stream.py` | Shared ROS Image rendering, bounded JPEG delivery and HTTP streaming regressions. | 不變 |
| `dinov3_ws/tests/test_lab_similarity.py` | DINO thresholds gate references; Lab selects the winner without weighted scores. | 不變 |
| `dinov3_ws/tests/test_patches.py` | patch 次序、正規化、幾何與模型 parity | 不變 |
| `dinov3_ws/tests/test_realtime.py` | Latency regressions: latest-only streams and input-driven scheduling. | 不變 |
| `dinov3_ws/tests/test_reference_gallery.py` | Reference grids must follow the same resize and stride as inference. | 不變 |
| `dinov3_ws/tests/test_sam3.py` | Official SAM3 bbox coordinates, preprocessing, cache and prompt contract. | 不變 |
| `dinov3_ws/tests/test_sam3_roi.py` | Camera-region validation, full-box containment and live setting updates. | 不變 |
| `dinov3_ws/tests/test_timing.py` | 吞吐視窗與跨 generation 模型計時 | 不變 |
| `dinov3_ws/tests/test_tracked_recognition.py` | Single-shot seed recognition and NvDCF classification/diagnostic lifetime. | 不變 |
| `dinov3_ws/tests/test_tracking.py` | BBox smoothing, geometric history matching and frame-local results. | 不變 |
| `dinov3_ws/tests/test_trt_backend.py` | TensorRT 輸入形狀、metadata 與 batch profile | 不變 |
| `dinov3_ws/tests/test_web_workflow.py` | Exercise the current pages and gallery/source controls without models or a camera. | 不變 |

## 原始模組相依圖（含函式內延遲匯入）

以下列出每個應用模組的直接內部相依；tests／工具的匯入也在本次同步更新。

- `__init__.py` → 無內部匯入
- `apriltag.py` → `depth_bbox`
- `async_bbox3d.py` → `tracking`
- `async_nvdcf.py` → `nvdcf`, `tracking`
- `cat2real.py` → `runtime`
- `cloud_filter.py` → 無內部匯入
- `config.py` → `async_bbox3d`, `async_nvdcf`, `grounded`, `nvdcf`, `runtime`
- `depth_bbox.py` → 無內部匯入
- `depth_sync.py` → 無內部匯入
- `encoder.py` → `cat2real`, `gpu_geometry`, `patches`, `runtime`, `trt_backend`
- `frame_pool.py` → 無內部匯入
- `gallery_files.py` → 無內部匯入
- `gpu_geometry.py` → `runtime`
- `grounded.py` → `gallery_files`, `nvdcf`, `object_visualization`
- `grounding_ipc.py` → `frame_pool`, `sam3_trt`
- `image_manager.py` → `gallery_files`
- `image_stream.py` → 無內部匯入
- `model_modules.py` → 無內部匯入
- `node.py` → `apriltag`, `async_bbox3d`, `async_nvdcf`, `cloud_filter`, `config`, `depth_bbox`, `depth_sync`, `encoder`, `frame_pool`, `gpu_geometry`, `grounded`, `grounding_ipc`, `image_stream`, `object_visualization`, `patches`, `runtime`, `sam3_roi`, `timing`, `tracking`, `visualization`, `web`
- `nvdcf.py` → `runtime`
- `object_visualization.py` → `patches`
- `patches.py` → 無內部匯入
- `recognition_state.py` → 無內部匯入
- `runtime.py` → 無內部匯入
- `sam3_roi.py` → 無內部匯入
- `sam3_trt.py` → `grounded`, `model_modules`
- `timing.py` → 無內部匯入
- `tracking.py` → 無內部匯入
- `trt_backend.py` → `runtime`
- `visualization.py` → `depth_bbox`
- `web.py` → `(package import)`, `cat2real`, `gallery_files`, `grounded`, `image_manager`, `nvdcf`, `patches`, `runtime`, `sam3_roi`

## 新結構與責任邊界

```text
dinov3_ws/
├── launch_pipeline.py                 # 原 shell 操作入口所呼叫的參數／exec 啟動器
├── build_models.py                    # 原模型建置／驗證 CLI
├── config/                            # 原部署設定，位置與參數不變
├── native/                            # 原 CUDA／DeepStream helper 與 Dockerfile
├── tests/                             # 原回歸、整合、效能工具；更新匯入與 mock 路徑
└── src/dinov3_cases/
    ├── setup.py                       # find_packages + 新 console entrypoint / package_data
    └── dinov3_cases/
        ├── config.py                  # ROS 參數預設與驗證
        ├── runtime.py                 # 工作區、模型、圖庫路徑
        ├── frame_pool.py              # 推論／繪圖共用的影格租約
        ├── timing.py                  # worker／HTTP／ROS 共用計時
        ├── gpu_geometry.py            # 深度與推論共用 CUDA ABI 綁定
        ├── api/
        │   ├── web.py                 # HTTP 辨識與狀態 API
        │   ├── image_manager.py       # 圖庫 HTTP API、相機擷取及獨立 app factory
        │   ├── image_stream.py        # 共用 JPEG worker／MJPEG
        │   ├── visualization.py       # 顯示疊圖
        │   ├── templates/
        │   └── static/
        ├── inference/
        │   ├── encoder.py             # Cat2Real 特徵／批次 ROI 推論
        │   ├── cat2real.py            # Cat2Real 前處理
        │   ├── trt_backend.py         # TensorRT 載入及輸入／metadata 驗證
        │   ├── model_modules.py       # 官方模型的 tensor-only 編譯介面
        │   ├── sam3_trt.py            # SAM3 vision／文字／bbox head
        │   ├── sam3_roi.py            # 原圖座標區域驗證與過濾
        │   ├── patches.py             # patch 特徵、相似度及參考網格
        │   └── grounding_ipc.py       # SAM3 子程序與共享記憶體協定
        ├── recognition/
        │   ├── grounded.py            # 候選清理、圖庫索引與 DINO／Lab 分類
        │   ├── gallery_files.py       # 路徑安全、圖檔解碼及原子存檔
        │   ├── recognition_state.py   # 設定／來源／圖庫的 generation 更新
        │   ├── images.py              # 自 HTTP 模組抽出的原 HSV 去背函式
        │   └── object_visualization.py # 辨識結果的延遲診斷與快照
        ├── tracking/
        │   ├── identities.py          # 原 tracking.py：幾何配對、公開 ID、生命週期及平滑
        │   ├── async_nvdcf.py         # 背景候選、補追、交接與延後分類
        │   └── nvdcf.py               # DeepStream 原生子程序整合
        ├── geometry/
        │   ├── depth_bbox.py          # 校正／點雲／平面與 2D 約束 3D 擬合
        │   ├── async_bbox3d.py        # 有界背景量測與快取
        │   ├── depth_sync.py          # RGB-D 時間／frame／尺寸配對
        │   └── cloud_filter.py        # 點雲品質與時間一致性
        └── ros/
            ├── node.py                # ROS 節點與既有 worker 生命週期
            └── messages.py            # Detection2D／Detection3D／AprilTag 訊息轉換
```

所有套件都有明確的 `__init__.py`，不在其中預先載入模型、啟動 worker 或建立相容轉接。跨領域共用模組保留頂層，避免再建立缺乏責任邊界的 utils 目錄。Camera 硬體驅動實際由外部 `orbbec_camera` ROS 套件提供，本專案只處理 ROS 訊息和校正資料，因此不新增虛構的 camera package。

`ros.node` 組合 API、推論、追蹤、幾何與訊息轉換；`api` 使用 recognition 狀態／圖庫及既有推論驗證；`recognition` 使用推論特徵與原圖庫規則；`geometry` 的背景關聯重用 `tracking.identities`。ROS 訊息建構已從分類與深度模組抽出，核心計算不再提供 ROS 序列化函式。`images.py` 的去背計算亦不再依賴 Flask。

## 執行流程與背景工作

- 使用者仍執行根目錄或 `dinov3_ws/` 原 shell 入口。`run.sh` 設定 ROS、venv、動態函式庫及原 `src/dinov3_cases` PYTHONPATH，交由 `launch_pipeline.py` exec `python -m dinov3_cases.ros.node`。console script `case_recognition` 改指向 `dinov3_cases.ros.node:main`。
- ROS 節點保留 `dinov3_cases` 名稱、所有參數、RGB／depth／CameraInfo 訂閱，以及 Detection2D、Detection3D、AprilTag、annotated Image、model timings 的既有 topic、QoS、header 與欄位。原程式未提供自訂 ROS service，因此沒有新增 service 層。
- 最新 RGB → 配對深度 → 共享影格 → SAM3 → NvDCF → Cat2Real／Lab → 背景 3D 框 → 獨立發布／繪圖 → ROS Image → JPEG／MJPEG 的資料流程不變。
- 背景角色保留：HTTP server、主辨識 worker、SAM3 Python 子行程、NvDCF 原生 context／cleanup worker、延後辨識／補追工作、3D 深度 worker、ROS 結果發布、繪圖、影像發布、JPEG 編碼。未變更鎖、事件、queue 深度、執行緒數、模型精度、推論參數或 GPU 排程。

## 耦合與重複職責的處理

- 原 `node.py` 同時負責 ROS、HTTP server 生命週期與多種 worker 組合，耦合最高。移到 `ros/node.py` 作為 composition root；本輪不重寫其生命週期，避免改變排程與資源所有權。
- 原 `grounded.py`、`depth_bbox.py` 同時包含計算與 ROS 訊息轉換；只抽出訊息函式。原 `apriltag.py` 完整功能合併進同一訊息模組，沒有棄用或刪除功能。
- 原 `image_manager.py` 同時含 HTTP 與 HSV 計算；原去背函式移入 `recognition/images.py`。
- `grounded.clean_proposals` 的候選 containment 與追蹤層的 containment 分別服務 seed proposal 和合併後的公開 track，有不同資料與生命週期，保留兩者。未因名稱類似而合併不同演算法。
- patch 網格繪圖與數值處理、辨識診斷與分類之間仍有既有相依；本輪按同一特徵／結果領域集中管理，避免為了搬移而拆成大量微型模組。

## 路徑與部署檢查

| 項目 | 處理方式 |
| --- | --- |
| Python imports | 應用內部改為明確 absolute imports；tests、工具、mock／monkeypatch 字串同步更新 |
| 動態 worker 模組 | 預設 `worker_module` 改為 `dinov3_cases.inference.grounding_ipc`，保留可注入 fake worker 的原介面 |
| 模組物件匯入 | `from dinov3_cases import node/depth_bbox` 更新至實際領域套件 |
| Flask 資源 | templates/static 與兩個 app factory 同在 api；保留 URL 與內容；package_data 改屬 `dinov3_cases.api` |
| 套件安裝 | `find_packages` 收錄子套件；ament resource、package.xml、console script 名稱保留 |
| Dockerfile／Compose | Docker 只建 native helper，Compose 掛載工作區；沒有指向被移動 Python 檔案，不需要修改 |
| Shell／PYTHONPATH | package 根目錄未移動，原 shell 入口與 source root 不變；launch_pipeline 的 -m 字串已更新 |
| 模型／CUDA／NvDCF 路徑 | `runtime.py` 留在原位置，`PROJECT`、artifacts、圖庫、native 預設路徑計算不變 |
| ROS launch | 相機仍由外部 orbbec_camera launch；本 repository 的 Python 啟動器已更新，無本地 ROS launch.py 需要搬移 |
| 效能工具動態載入 | `spec_from_file_location` 指向使用者指定的 baseline 檔案，不依賴舊套件位置，保留原行為 |

原本 runtime 預設路徑依 source workspace 佈局推算，本輪維持此操作方式；wheel 驗證僅證明可安裝、可匯入及資源完整，不代表 wheel 已包含模型、圖庫、ROS 或 native dependencies。

## 驗證

- 修改前：完整 pytest **472 passed，14.25 秒**。
- 修改後：相同完整 pytest **472 passed，15.08 秒**，沒有新增 skip／xfail，也未修改測試斷言。
- 語法樹對照：30 個保留模組的非 import 邏輯一致（僅 worker 模組字串按新位置更新）；四個抽出函式的 AST 完全一致。
- 9 個 HTML／JS／CSS 資源與原檔案逐位元組一致。
- `compileall`、既有 shell scripts 的 `bash -n`、`git diff --check` 通過。
- wheel 在暫存副本建置並安裝，避免污染或替換目前 venv 的套件。
- 真實 `check_official_pipeline.py` 通過：官方 SAM3、Cat2Real、圖庫辨識、多類別驗證及新 SAM3 subprocess／共享記憶體 parity 均成功；TensorRT 11.1.0.106、Torch 2.14.1+cu130。未重新建置或修改模型權重。
- Node.js 不存在（`node: command not found`），所以兩個 `.cjs` 前端互動測試未能執行。已更新其靜態資源路徑；未以略過斷言來假裝通過。

測試命令與 GPU／相機工具仍見 `dinov3_ws/tests/README.md`。本輪不重啟外部相機或 Docker 容器。

補充完成結果：

- wheel 隔離安裝後，在 `/tmp` 執行並保留 ROS 環境：**37 個子模組全部可匯入**；`case_recognition` metadata 指向正確且可載入 callable；獨立圖庫首頁、`/images` 及七個 JS／CSS 資源皆回傳 HTTP 200；`python -m dinov3_cases.inference.grounding_ipc --help` 成功。
- 真實 `check_grounded_trt_pipeline.py` **通過**：實際 SAM3 worker、Cat2Real、HTTP 圖庫／來源切換、ROS detections、960×540 bgr8 影像發布；刻意阻塞 renderer 時仍先發布 bbox，釋放後能繼續影像輸出，最後共享影格釋放檢查成功。
- 驗證日誌：`/tmp/package-refactor-baseline.log`、`/tmp/package-refactor-tests.log`、`/tmp/package-refactor-wheel.log`、`/tmp/package-refactor-models.log`、`/tmp/package-refactor-ros-http.log`；暫存日誌不納入部署內容。
