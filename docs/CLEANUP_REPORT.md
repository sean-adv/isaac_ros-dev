# 功能導向清理報告

日期：2026-11-03。功能基準見 [CURRENT_FUNCTIONALITY.md](CURRENT_FUNCTIONALITY.md)。
本輪起點已有 12 個 tracked 修改及 1 個 untracked encoder 測試替身；已在 `/tmp/isaac-cleanup-original`
保存原始內容，不將既有 encoder／辨識清理冒列為本輪變更。

## 分類與依據

| 分類 | 項目 | 完整路徑／判斷依據與處理 |
| --- | --- | --- |
| KEEP | 兩個頁面、全部現用 HTTP routes／static／templates | 首頁與圖庫互相連結，JS 呼叫現有 API；沒有可證實淘汰的 UI 功能鏈 |
| KEEP | SAM3 IPC、NvDCF、Cat2Real、追蹤、3D／診斷／影像輸出 | node 啟動／worker／HTTP 與 ROS 路徑均會到達；不是只依引用數保留 |
| KEEP | CPU plane backend、單筆 engine 與 batch engine | 有明確設定與 metadata 選擇路徑，硬體／部署差異仍需要 |
| KEEP | `model_modules.py`、`torch_tensorrt` import、native helpers | builder／runtime 共用，反序列化註冊、子程序與 CDLL/dlopen 不是一般 import 依賴 |
| KEEP | 全部現用 tests／check／profile、standalone image-manager factory | 測試驗證同一圖庫路由／離線去背與相機不可用行為；診斷工具具有獨立入口 |
| REMOVE | `native/build_gpu_geometry.sh`、`build_nvdcf.sh` | 只把固定 target 轉給 `build.sh`；Docker／setup_all 已直接使用 `build.sh`，沒有獨立建置邏輯 |
| REMOVE | `native/install_nvdcf.sh` | 已明確標為 compatibility installer；現行 setup／Docker 分開系統安裝與使用者建置。錯誤提示遷移至現用指令後移除整條舊入口 |
| REMOVE | `requirements-build.txt` | 內容只有 `-r requirements.txt`；無 build 專用依賴。setup／dev requirements 與文件直接使用唯一清單 |
| REMOVE | `web.decode_image` | 只設定 `load_image(..., mode="RGB")` 的本地包裝；呼叫處直接指定模式，底層 EXIF／格式／大小驗證保留 |
| REFACTOR | 兩個圖庫上傳／儲存入口 | PNG 暫存檔寫入與 rename 合併至既有 `gallery_files.py`；保留原本 RGB／RGBA、檔名規則、回應與狀態更新 |
| REFACTOR | 模型識別字串 | runtime 與 builder 改用既有 Cat2Real／SAM3 常數，維持模型 ID、revision 與 metadata 值 |
| REFACTOR | 過時文件與歷史測試數字 | 現況集中功能地圖／本報告；測試與維護文件保留操作、現用契約與最新驗證；歷史 audit 明確標示非現況 |
| UNCERTAIN | 根 host `run.sh` 舊容器預設、margin 設定欄位 | 外部部署與目前 UI／HTTP 契約仍可能使用；保留並記錄，避免改變現用介面 |
| UNCERTAIN | 非同步完整模型 smoke 的 pending 狀態／交接時序 | 清理前後都在再次交接階段失敗（不同 assertions）；未修改排程／斷言，保留原路徑與工具，詳見驗證記錄 |
| UNCERTAIN | 相鄰外部 repositories／舊環境／copy／trash | 本應用啟動不依賴大部分項目，但不能證明其他工作流程已淘汰；含不可任意刪除資料，全部保留 |

未發現可證實「整條已淘汰但仍互相引用」的額外應用功能鏈。
舊資源監控、多平面姿態、影片編碼、舊 encoder 與 recognition cache 已在本輪開始前移除；
本輪未重複計入移除成果。對硬體或外部使用證據不足的項目不作刪除。

## 專案結構

```text
./
├── docs/                         現況功能地圖、分類與本輪結果
├── README.md / DOCKER_HOST.md     專案入口與 host 部署
├── start.sh / compose.yaml        開發容器管理
├── run.sh / run_camera.sh         保留現用根入口
└── dinov3_ws/
    ├── run*.sh / launch_pipeline.py  相機與 ROS／HTTP 應用啟動
    ├── setup*.sh / requirements*    Python／系統環境
    ├── config/                      ROS、DDS、NvDCF
    ├── build_models.py              模型建置與數值驗證
    ├── native/                      CUDA／NvDCF 原始碼與統一 CMake 建置
    ├── src/dinov3_cases/
    │   ├── dinov3_cases/             推論、背景排程、ROS、HTTP、static／templates
    │   └── docs/                     runtime／3D／維護專題
    ├── tests/                        回歸、smoke、效能診斷與替身
    └── artifacts/ / data/ / .venv/   本機資產與使用者資料，保持原樣
```

既有模組已按責任區分；不為目錄外觀搬移排程／鎖／共享影格，也不引入 framework。
wheel 仍需搭配 workspace config、模型、資料與系統套件，尚非獨立 application image。

## 驗證

清理前：完整 pytest **470 passed（14.38 秒）**，無失敗／skip／xfail。

| 驗證 | 結果與限制 |
| --- | --- |
| 清理後完整 pytest | **472 passed（14.08 秒）**；無失敗／skip／xfail；原 470 項全數保留，新增兩項 HTTP 工作流程 smoke |
| 新 smoke 對照清理前程式 | **2 passed（1.34 秒）**；使用 Git 起點加上原有 dirty 程式的獨立副本，確認涵蓋原有行為 |
| 現用頁面／圖庫／來源 | 首頁與圖庫 HTTP 200、相互連結、MJPEG URL；上傳／RGB 儲存／網格／選圖庫不使結果失效／上傳目標與相機切換／壞圖片批次拒絕均通過 |
| 真實 SAM3／Cat2Real HTTP／ROS smoke | **通過**；刻意阻塞 renderer 時偵測可先完成，解除後收到 960×540 bgr8 ROS Image；四張合成 RGB 輸入均收到 Detection2D 結果 |
| 非同步模型／NvDCF 整合 | 見下方獨立記錄，不包含在 pytest 472 項中 |
| Native configure／build／install | `/tmp/isaac-cleanup-native.ymyzuN` 內 CUDA geometry 與 NvDCF 全部通過；安裝產物 `ldd` 無 missing library，未覆蓋 workspace binary |
| Python／shell／CLI | tracked Python AST、所有保留 shell 的 `bash -n`、launch／build_models `--help` 通過 |
| Wheel | `/tmp/isaac-cleanup-wheel-qz4rzime` 建置通過，確認所有應用 Python／static／templates／console entry；不代表 wheel 可脫離 workspace 部署 |
| 文件／差異 | 現況文件本機連結與 `git diff --check` 通過；已刪入口無執行引用 |
| 起點修改保護 | 起點 9 個已修改／未追蹤程式與測試檔 SHA-256 完全一致；四份既有 dirty 文件只更新現況／歷史標示，原內容另存 `/tmp/isaac-cleanup-original` |
| 前端 JS／實際瀏覽器 | **未驗證**：環境沒有 Node.js，未執行兩份 `.cjs`；HTTP 模板 smoke 不代表瀏覽器渲染／互動驗證 |
| Docker Compose／image build | **未驗證**：Docker CLI 沒有 Compose plugin；未重建 image 或重啟容器。native 建置通過不能替代 image build |
| 真實相機／RGB-D／效能 | **未驗證**：模型 smoke 使用儲存圖片與合成 RGB，停用 pose；3D／CUDA 在完整 pytest 中驗證，未量測實機端到端精度／延遲／吞吐，無效能持平宣稱 |

模型 smoke 報告：`/tmp/isaac-cleanup-model-4sdyblzx/sam3/memory_pipeline_validation.json`。
只使用暫存圖庫、獨立 ROS domain 與 port 0；部署模型、原本 validation／metadata 與使用者圖庫未改動。

未在本輪啟動真實相機、下載／重建模型、修改 engine metadata 或重新啟動既有服務。


### 非同步模型整合檢查的未解問題

`tests/check_async_nvdcf_pipeline.py` 不由 pytest 自動收集，須另載入真實模型。
本輪在隔離的 ROS domain 131、暫存圖庫與 `/tmp` 報告位置執行，未改動腳本斷言：

- 清理後在第 143 行 `pending and all(p['recognition_pending'] for p in pending)` 失敗。
  記錄：`/tmp/isaac-cleanup-async.log`。
- 用起點 Git 程式加原有 dirty 程式重建獨立來源副本，連結相同模型與設定後重跑，
  在同一再次交接階段的第 139 行 `recognition_source == 'pending'` 失敗，實際為 `cached`。
  記錄：`/tmp/isaac-cleanup-async-baseline.log`。

兩輪的確切 assertion 不同，不能宣稱它們是完全相同的失敗；只能確認清理前該整合檢查也未通過。
目前無足夠證據區分檢查觀察時序、分類沿用契約與 runtime 問題，暫留 UNCERTAIN。
本輪未變更 `node.py`／`async_nvdcf.py`／`tracking.py` 或這份檢查，未刪除、skip 或放寬 assertions。
472 項完整 pytest 包含非同步交接、延後分類、失追、分類沿用及 CPU／GPU 回歸，但不代表
這份真實模型非同步 smoke 已通過；尚無完整實機部署／效能持平結論。
