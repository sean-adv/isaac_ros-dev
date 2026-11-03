# Repository audit 與整理記錄

> 本文件下方的初次 audit 記錄反映清理當時狀態。後續使用者已明確要求改用 ROS Image 並移除影片編碼；目前狀態以本節及更新後的應用／建置文件為準。

## 後續更新：單次辨識與同步清理（2026-11-03）

依使用者要求，移除 `TrackedObjectRecognizer`：部署每次都建立新實例，
跨呼叫快取與重新分類分支沒有實際重用。節點直接執行單次辨識，保留 NvDCF
分類沿用、背景交接與診斷快照。輸入 margin 僅留在 ROS／HTTP 設定邊界驗證，
內部分類介面不再接收；輸出的 Lab gap 保留。深度佇列改用原有 `node.lock`，
移除無等待者的 Condition；取消追蹤 4 個 Python 快取檔（保留本機檔案）。

完整 pytest：**467 通過、0 失敗**（14.36 秒），未新增 skip／xfail。
相較前次 474 項，移除舊快取專用的 9 項測試、加入 2 項單次辨識測試；
現用節點交接／失追回歸仍保留。74 個 Python 檔案語法檢查通過。
本輪未啟動真實模型／相機；沒有 Node.js，未執行前端測試。
詳見 [維護狀態](dinov3_ws/src/dinov3_cases/docs/maintenance.md)。

## 後續更新：移除未使用的多平面姿態估算（2026-11-03）

使用者確認目前只使用 RANSAC，移除未被主流程匯入的 `surface_pose.py`
與專屬 `test_surface_pose.py`（移除前為 3 通過、10 失敗）。保留
`depth_bbox.py` 的單平面 RANSAC、CPU／CUDA 實作與現用深度／姿態測試。
完整 pytest：**474 通過、0 失敗**（14.37 秒），未新增 skip／xfail。
下方多平面 API 與失敗數字為移除前的歷史記錄。

## 後續更新：移除資源監控（2026-11-03）

依使用者要求移除 `resources.py`、`static/resources.js` 及專屬的
`test_resources.py`、`test_resource_web.py`。此功能原先未接入部署節點與首頁；
模型耗時與效能診斷工具保留。下方資源監控與測試數字為移除前的歷史記錄。

## 後續更新：ROS Image／MJPEG（2026-11-02）

- 網頁實際訂閱 `/medicine_recognition/image_annotated`（可由 `annotated_image_topic` 覆寫），背景轉 JPEG，首頁／圖庫透過 `/image_annotated.mjpg` 的 `<img>` 顯示。保留 960×540、bgr8、header、Reliable／KeepLast(1) 與 `video_fps` 上限。
- 新增 `image_stream.py`：最新影格單一 pending slot、所有 HTTP 觀看者共享 JPEG；無觀看者不轉 JPEG。renderer 排除自己的 bridge subscription，無外部／網頁觀看者不繪圖；DDS 與 JPEG 各在背景工作，不阻塞推論。
- 移除 H.264／WebRTC Python 模組、協商／peer API、ROS 封包 topic、播放器、GPU video C++ bridge／build wrapper、專屬設定，以及 av／aiortc／FFmpeg build 依賴。保留 CUDA geometry 與 NvDCF；既有編譯 binary、模型及環境套件未擅自刪除／uninstall。
- raw Image、3D／慢讀者／影格節流測試保留並調整，移除舊 codec／peer 專用測試；保留的 profile／整合工具已改用圖片路徑。
- 完整 pytest：**476 passed、12 failed**（14.87 秒），失敗仍為原有 resource endpoint 2 項與 surface pose 10 項；真實 ROS Image→JPEG、多網頁共享及背景編碼測試通過。
- 真實 SAM3／Cat2Real 模型 ROS／HTTP smoke 通過：刻意阻塞 renderer 不阻塞結果，解除後收到 raw Image，4 個合成相機輸入皆收到 ROS 偵測結果；報告保存在 `/tmp/isaac-image-migration/memory_pipeline_validation.json`，未覆蓋使用者模型報告。
- CUDA／NvDCF configure／build、Python imports／AST、shell、wheel 內容及 diff 檢查通過；新 wheel 含 image bridge／player，無舊 codec 模組。未執行 Node.js／真實瀏覽器、Compose image build、真實相機端到端或前後效能 benchmark。
- 套用：重啟辨識程序並重新載入兩個網頁。若重建 native image，使用既有 `native/build.sh`／Dockerfile。已安裝的舊影片 binary 不再被應用呼叫，不需立即刪除。網頁改用 JPEG，與 raw Image 一樣可能增加頻寬；未聲稱效能與影片路徑相同。

---

日期：2026-11-02。目標是保留功能、數值精度、ROS／HTTP 介面及排程，以可追溯的小改動降低維護負擔。

## 範圍與證據

盤點 128 個 Git 追蹤檔案，讀取主要入口、套件設定、應用 Python 的 AST 依賴、原生程式載入方式、設定與測試引用，並盤點根目錄未追蹤／忽略目錄。未對外部 repository 的內部實作進行逐檔重構；它們有各自的 Git、依賴和資源生命週期。靜態無引用僅作線索，不能證明外部使用者未使用某個公開模組。

開始時已有以下修改，全部原樣保留：

- `dinov3_ws/README.md`、`BUILD_GUIDE.md`、`setup_all.sh`。
- `native/Dockerfile`、`build_gpu_geometry.sh`、`build_nvdcf.sh`、`build_video_gpu.sh`、`install_nvdcf.sh`。
- 未追蹤的 `native/CMakeLists.txt`、`build.sh`、`install_deps.sh`。

Git ownership 不一致，檢查使用單次 `git -c safe.directory=/workspaces/isaac_ros-dev`，未變更全域 Git 設定。一般 shell sandbox 因 `.aws` 掛載權限錯誤無法啟動，改用核准的 sandbox 外命令；未讀取憑證內容。

## Architecture overview

| 目錄／入口 | 職責與依賴 |
| --- | --- |
| `start.sh`、`compose.yaml`、`.devcontainer/` | Host 容器與開發環境；NVIDIA runtime、USB、host network／IPC |
| `run.sh` | 容器內轉送應用入口；host 上保留既有 `docker exec` 目標及環境覆寫 |
| `run_camera.sh` → `dinov3_ws/run_camera.sh` | 單一相機設定來源，啟動外部 Orbbec launch |
| `dinov3_ws/run.sh` → `launch_pipeline.py` → `dinov3_cases.node:main` | 載入 ROS、venv、DDS、執行緒預設；合併 YAML 與命令列 ROS 參數 |
| `src/dinov3_cases/setup.py`、`package.xml` | ament Python 套件與 `case_recognition` console entry point；封裝網頁 static／templates |
| `node.py`、`frame_pool.py`、`depth_sync.py` | ROS 輸入、共享影格生命週期、同步與背景輸出排程 |
| `grounding_ipc.py`、`sam3_trt.py`、`model_modules.py` | `python -m dinov3_cases.grounding_ipc` 子程序；SAM3 官方 TRT vision FP16 + PyTorch bbox head |
| `async_nvdcf.py`、`nvdcf.py`、`tracking.py` | 原生 tracker 子程序、候選補追與交接、公開 ID／框平滑 |
| `encoder.py`、`trt_backend.py`、`cat2real.py`、`patches.py`、`grounded.py` | Cat2Real FP32 特徵、微批次、圖庫與 Lab 比對／分類快取 |
| `async_bbox3d.py`、`depth_bbox.py`、`cloud_filter.py`、`gpu_geometry.py` | 非同步深度、連通點雲、單平面擬合及 3D 框；CPU／CUDA 分支 |
| `web.py`、`image_manager.py`、`object_visualization.py`、`static/`、`templates/` | HTTP、圖庫維護、診斷與前端；共用節點狀態 |
| `visualization.py`、`h264_stream.py`、`gpu_video.py` | 共用疊圖、CPU／GPU 編碼、WebRTC 與 ROS 發布 |
| `runtime.py`、`config/` | 相對專案的模型／資料預設路徑、ROS 參數、NvDCF 與 DDS 設定 |
| `build_models.py`、`native/` | 模型取得／編譯／驗證、CUDA／DeepStream／FFmpeg helper 建置 |
| `tests/` | 回歸、IPC fixture、GPU 整合、相機與效能診斷；均保留 |
| `artifacts/`、`data/gallery/`、`.venv/` | 外部模型與生成產物、可寫圖庫、機器專屬 Python 環境；不屬於原始碼發佈 |

資料流：RGB + registered depth → frame pool → SAM3 背景偵測 → NvDCF 補追／前景追蹤 → Cat2Real 延後更新類別；獨立 3D worker 使用深度；ROS 發布與疊圖 worker 讀取完成結果，H.264 封包共用於 WebRTC／ROS。此生命週期及執行順序未修改。

## A. Safe to Remove — 已執行

| 項目 | 證據及處理 |
| --- | --- |
| 4 個受追蹤的 `__pycache__/*.cpython-312.pyc` | 對應 `__init__.py`、`node.py`、`patches.py`、`trt_backend.py` 均存在；Python 可重新產生，入口與封裝使用原始碼。刪除 Git 內的快取，既有 ignore 已涵蓋 |
| 根目錄相機腳本重複實作 | 與 `dinov3_ws/run_camera.sh` 只有專案路徑寫法不同，DDS、ROS discovery、相機 launch 參數完全相同。根入口保留並委派至唯一實作，原始參數完整轉送 |
| 測試文件過時的狀態描述 | `test_surface_pose.py` 實際已追蹤，完整 baseline 有 12 項失敗；更新描述並列出第二組前端測試 |

未移除任何 runtime dependency、模型、engine、metadata、圖庫、測試資料或使用者產物。未發現可在排除副作用、外部 API 與動態載入後確認安全移除的應用 import／class／function。

## B. Recommended Refactoring — 低風險部分已執行

- 相機啟動邏輯合併；兩個既有腳本路徑皆保留，無新增 runtime 依賴。
- 新增 `requirements-dev.txt`，沿用既有 requirements，將已安裝的 pytest 8.4.2 明列為開發依賴；未改動 runtime setup 或已測試的 Thor pins。
- 新增 `native/.dockerignore`，Docker context 只允許 Dockerfile、CMake、shell、C++／CUDA build inputs，防止未來把 native 目錄的 cache、binary 或臨時資料帶入映像。
- 新增根 README 與本報告，記錄唯一推薦啟動流程、相容入口差異與部署邊界。

`node.py` 約 955 行、`async_nvdcf.py` 約 498 行，集中排程與生命週期；`web.py` 直接依賴節點狀態。這些位置有耦合，但拆分容易改變鎖、租約及背景工作順序，本次不重新分層。原生 CMake 合併已是使用者進行中的改動，未再次重寫。

## C. Requires Verification — 保留

- 多平面姿態估算：初次 audit 時保留待確認；2026-11-03 使用者確認只使用 RANSAC 後，已移除模組與專屬測試，見上方後續更新。
- 資源監控：初次 audit 時未接入 node／首頁；2026-11-03 已依使用者要求移除實作與專屬測試，見上方後續更新。
- 根 `run.sh` 的 host 預設 container／路徑與 Compose 不同：可能仍供既有部署使用，保留並文件化；後續確認使用者後再統一。
- `requirements-build.txt` 目前只有 include runtime requirements；模型 wrapper 也在 runtime 使用 Transformers／Torch-TensorRT，無證據可移走這些依賴。保留相容檔名及 setup 引用。
- AST 看似未使用的 `torch_tensorrt` import：實際會註冊反序列化 operator，保留。`model_modules.py` 同時由 builder 與 SAM3 runtime 使用；不得視為 build-only。
- `tests/ipc_fake_worker.py` 是真實子程序 fixture；`profile_depth_bbox.py`／`profile_gpu_pipeline.py` 會動態匯入外部 baseline 檔案；所有 check／profile 工具皆保留。
- 外部 repository：`DART/`、`detectron2/`、`dinov3_tensorrt/`、`OmDet/`、`TensorRT-Model-Connect/`、`sam2_trt_inference/` 等皆保留；應用未發現對它們的直接 import，但這不能證明其他工作流程未使用。
- `orbbec_sdk/` 是相機入口的 fallback install 來源，不能刪除。其他舊 venv、`copy/`（約 1.3 GB）、`trash/`（約 41 GB）、`dinov3_cases_data/`、`isaac_ros_assets/` 和歷史權重也不清除；先取得擁有者確認與備份／還原證據。

## Deployment readiness

目前是以完整工作區 volume 執行的 Thor 開發容器，尚非自包含 application image。

| 類別 | 現況與發佈處理 |
| --- | --- |
| 應用套件 | 納入 `src/dinov3_cases`、啟動器、`config/`、requirements／pins；套件的 static／templates 須一起發佈 |
| 模型資源 | 外部提供 `artifacts/cat2real-dinov3-384/`、`artifacts/sam3/` 中所需模型、source tokenizer／processor、`.ts` 與匹配 `.json` metadata；用 volume 或獨立模型 artifact |
| 模型選擇 | `runtime.py` 以 metadata 存在與否選 batch8／native vision，只有 `.ts` 不足。保留現行相對專案預設；入口以 ROS 參數覆寫 engine／gallery／worker |
| 圖庫與資料 | `data/gallery/` 為可寫且應持久化的資料；不以 image 重建覆蓋。`DINO_MODEL`／`SAM3_MODEL` 仍依專案 artifacts 路徑，任意搬移權重尚非完整支援 |
| 原生 runtime | 需 helpers、CUDA driver/runtime、DeepStream 9.1、FFmpeg；優先 workspace artifacts，否則 `/usr/local/bin`／`lib`。DeepStream 及 aarch64 library 路徑仍固定 |
| Build-time | CMake、nvcc、g++、pkg-config、FFmpeg headers；模型 compiler 與 runtime 使用同組 Torch-TensorRT，尚未採 multi-stage image 分離工具鏈 |
| Development | pytest 由新增 dev requirements 管理；前端測試需 Node.js。check／profile／測試不必放入最小 runtime 套件 |
| 不應封裝 | `.venv/`、build/install/log、cache、外部 repo、舊環境、copy/trash、非必要歷史 engines／benchmark outputs；不要複製整個 workspace |

仍有缺口：

1. `setup.py` 的 wheel 可帶入 Python／網頁資源，但未包含 workspace `config/`、模型與 `launch_pipeline.py`；`runtime.PROJECT` 依來源目錄層級定位。wheel 安裝成功不等於可單獨部署，需另定 relocation／share 資源策略後驗證。
2. Compose 目前維持開發容器存活，不自動啟動相機／辨識；host root wrapper 的舊目標尚未統一。正式服務的 readiness／restart 行為需後續設計。
3. `setup_env.sh` 固定 Thor CUDA 13、Python 3.12；ROS Jazzy、Orbbec 與自訂訊息由系統提供，沒有獨立鎖定驅動來源版本；目前沒有其他平台的可重現路徑。
4. Docker base image 使用 tag，DeepStream 下載沒有 checksum；Thor constraints 不是所有 transitive dependency 的完整 lock。未嘗試升版／降版或強制合併 backend。
5. Hugging Face 需要存取權；既有 metadata 與硬體／套件版本應隨 engine 發佈。`build_models.py --validate-only` 仍會重寫 validation／metadata，因此本次未對使用者模型執行此命令。
6. 先釐清 12 項既有回歸失敗的預期契約，再建立完整綠燈 CI；不能以忽略測試替代確認。

## Validation results

整理前完整 pytest：**489 passed、12 failed**（12.10 秒）。失敗分布為 `test_resource_web.py` 2 項及 `test_surface_pose.py` 10 項；完整測試保留，未增加 skip／xfail。

| 驗證 | 結果 |
| --- | --- |
| 完整 pytest（整理後） | **489 passed、12 failed**，11.67 秒；失敗名稱與 baseline 完全相同，未新增失敗。涵蓋真實 CUDA／native tracker／GPU video 測試，但不等同完整模型或相機驗證 |
| Python／入口 | 32 個應用模組實際匯入成功；相對 module imports 的 AST 引用、`node.main` callable、launch／build_models `--help` 均通過 |
| 設定與資源 | package／DDS XML、ROS YAML、OpenCV 格式 NvDCF YAML 解析成功；現有 engine／metadata、模型 source 目錄、圖庫、helpers 路徑存在；metadata 保持 Cat2Real FP32／SAM3 vision FP16 |
| 相機入口 | 3 組 DDS 環境的 mock launch 比對通過，包含帶空白參數、清除遠端 ROS discovery 與從其他目錄呼叫；不啟動硬體 |
| ROS launch 引用 | base ROS 無 Orbbec；依現有 fallback source `orbbec_sdk/install/setup.bash` 後，`ros2 pkg prefix` 與 `gemini_330_series.launch.py --show-args` 通過 |
| Shell | 13 支 tracked／既有未追蹤 shell scripts 的 `bash -n` 通過 |
| GPU 基礎環境 | CUDA driver `cuInit`、Thor 上 PyTorch CUDA allocation、TensorRT 11.1 Builder 成功；本次驗證以工具的 root 執行身分進行，不能代替 admin 權限驗證 |
| Native build／install | `/tmp` 獨立 CMake configure、三個 targets 編譯與 install 成功；已安裝三個 binary／library 的 `ldd` 無 missing library，未覆蓋現有 artifacts |
| Python wheel | `/tmp` 複本 bdist_wheel 成功；console entry point、所有 static／templates、ament marker／package.xml 均存在，無 bytecode；尚非獨立部署驗證 |
| 使用者改動／diff | 原先 11 個 dirty／untracked 檔案 SHA-256 完全一致；`git diff --check` 通過 |
| 前端 Node 測試 | **未驗證**：環境沒有 Node.js，保留兩個 `.cjs` 測試與操作指令 |
| Docker Compose／image | **未驗證**：Docker CLI 存在但未安裝 Compose plugin，未 build image 或重啟容器；native build 成功不代表 Docker build 已通過 |
| 真實相機、完整模型與效能 | **未驗證**：未啟動相機或主要服務；未載入 SAM3／Cat2Real 做完整 pipeline parity、精度／吞吐／延遲 benchmark，也未重寫模型 metadata／validation reports |

驗證初次曾用通用 PyYAML 讀取 NvDCF 的 `%YAML:1.0` 格式而失敗；改以實際適用的 OpenCV FileStorage 重驗通過，未更動設定格式。

## Remaining recommendations

資源監控與多平面 API 已依使用者要求移除；後續確認 root host wrapper 的服務目標。後續部署工作應先定義最小 source/config/model manifest、補齊 wheel relocation、鎖定 base image／DeepStream 校驗與 Orbbec 版本，再在實際 admin／相機環境建立模型 parity 和端到端效能基準。約 8.2 GB 的 `.git` 及大型外部目錄可另做容量盤點，但本次不改寫 Git history 或刪除外部資料。
