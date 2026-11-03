# 建置與執行

## 啟動

從工作區根目錄執行，相機與辨識服務使用同一位使用者（容器預設 `admin`）：

```bash
bash dinov3_ws/run_camera.sh
# 另一個終端
bash dinov3_ws/run.sh
```

開啟 <http://127.0.0.1:8090>。工作區根目錄的 `run.sh` 可從 host 轉入既有容器；容器啟動方式見 [DOCKER_HOST.md](../DOCKER_HOST.md)。

`run.sh` 載入 ROS Jazzy、專案 `.venv`、TensorRT 動態函式庫與 Fast DDS 設定，再由 `launch_pipeline.py` 載入 [dinov3_cases.yaml](config/dinov3_cases.yaml)。可直接追加 ROS 參數：

```bash
bash dinov3_ws/run.sh --ros-args -p port:=8091
```

相機使用已安裝的 `orbbec_camera`；若位於其他 ROS 工作區，可設定 `ORBBEC_SETUP=/path/to/install/setup.bash`。RGB、深度與 CameraInfo 訂閱採 RELIABLE、KEEP_LAST、depth=1。跨使用者啟動可能影響 Fast DDS 共享記憶體通訊。

相機入口使用 `enumerate_net_device:=false`，停用區網相機列舉。相機與辨識入口均強制
`ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`，並清除 `ROS_STATIC_PEERS`、`ROS_DISCOVERY_SERVER`
與舊的 `ROS_LOCALHOST_ONLY` 設定。預設 Fast DDS UDP 僅允許 `127.0.0.1`，保留共享記憶體傳輸。
若自行指定 DDS profile，請同樣限制網路介面。修改後須重啟相機與辨識程序才會生效。

## 環境與模型

需由系統提供 Python 3.12、ROS 2 Jazzy、CUDA driver/runtime、cv_bridge、vision_msgs、isaac_ros_apriltag_interfaces 與相機驅動。Python 相依套件與模型均位於 `dinov3_ws`，不需要相鄰模型專案。

```bash
# 在 Isaac ROS 容器內，以工作區使用者執行；不要對整支入口加 sudo
bash dinov3_ws/setup_all.sh
```

`setup_all.sh` 先呼叫 `setup_env.sh`，再檢查 NvDCF。如果工作區或容器映像已有 bridge 執行檔及 DeepStream 9.1 追蹤函式庫，就沿用；否則使用 `sudo` 呼叫 `native/install_deps.sh` 安裝系統依賴。最後以工作區使用者執行 `native/build.sh`，增量編譯全部 native 產物。檔案檢查不代表 GPU 相容性驗證，主機與容器仍須符合 [部署要求](../DOCKER_HOST.md)。入口不下載或建置模型。

個別維護時可執行 `bash dinov3_ws/setup_env.sh`；原生依賴安裝使用 `sudo bash dinov3_ws/native/install_deps.sh`，編譯使用 `bash dinov3_ws/native/build.sh`。

`setup_env.sh` 先安裝 CUDA 13.0 Torch／torchvision，再使用 [constraints-thor.txt](constraints-thor.txt) 安裝 [requirements.txt](requirements.txt)。可透過 `PYTHON_BIN`、`DINOV3_VENV` 指定 Python 與環境位置。

| 用途 | 模型與執行方式 | 本機位置 |
| --- | --- | --- |
| SAM3 vision | 官方 Torch-TensorRT FP16，1008×1008 | `artifacts/sam3/vision-fp16.ts` |
| SAM3 文字與 bbox head | 官方 Transformers PyTorch FP16 | `artifacts/sam3/source/` |
| Cat2Real | 官方 Torch-TensorRT FP32，384×384 | `artifacts/cat2real-dinov3-384/cat2real-fp32-trt.ts` |
| Cat2Real processor／原始權重 | Transformers | `artifacts/cat2real-dinov3-384/source/` |
| NvDCF | DeepStream 9.1 原生 C++ 子程序 | `artifacts/nvdcf/nvdcf_bridge` |
| 商品圖庫 | 依資料夾名稱分類 | `data/gallery/` |

SAM3 子程序與主程序共用 `.venv`，透過共享影格池傳遞 RGB。NvDCF 使用獨立程序載入系統 DeepStream 函式庫，與模型 TensorRT 11 環境隔離。

模型建置統一使用 `build_models.py`：

```bash
# 已有本機權重時重建全部模型
dinov3_ws/.venv/bin/python dinov3_ws/build_models.py

# 分別建置
dinov3_ws/.venv/bin/python dinov3_ws/build_models.py --target sam3
dinov3_ws/.venv/bin/python dinov3_ws/build_models.py --target cat2real

# 首次下載；SAM3 需先取得官方存取權並登入
dinov3_ws/.venv/bin/hf auth login
dinov3_ws/.venv/bin/python dinov3_ws/build_models.py --download

# 只驗證既有產物
dinov3_ws/.venv/bin/python dinov3_ws/build_models.py --validate-only
```

Cat2Real 固定 revision `a763964f2041722ffcc5bc4c72e3cb241530727e`，SAM3 固定 `3c879f39826c281e95690f02c7821c4de09afae7`。建置要求完整編譯 vision 圖；儲存後重新載入，與原始 FP32 權重比較，通過才寫入同名 `.json` 部署中繼資料。各模型目錄的 `validation.json` 保存 SHA256、版本、樣本結果與測速。

Cat2Real 在 Thor 固定使用 FP32，避免 FP16 非有限輸出。SAM3 只編譯 vision；bbox head 與文字編碼使用官方 PyTorch 模組，不執行 mask 或 SAM3 tracking。

## GPU 延遲優化

目前預設使用 CUDA RANSAC、NvDCF RGB 共享記憶體傳輸及 GPU 縮放、SAM3 GPU 前處理，以及 DINO 同一影格內的 ROI 批次推論與圖庫矩陣比對。RANSAC 的最後平面精修、深度連通區與小型投影擬合仍在 CPU；投影擬合使用解析 Jacobian，減少數值差分。`bbox3d_quantile` 目前部署值為 `0.10`。

在 Thor / CUDA 13 / DeepStream 9.1 上，從工作區根目錄建置原生程式：

```bash
bash dinov3_ws/native/build.sh

# 只建置個別 target（自動更新相依的 CUDA 函式庫）
bash dinov3_ws/native/build.sh nvdcf_bridge
bash dinov3_ws/native/build.sh gpu_geometry
```

需要 CUDA 開發工具、DeepStream headers、`cmake`（3.24 以上）、`g++`。`setup_all.sh` 與原生 Dockerfile 已包含建置步驟。預設 `CUDA_ARCH=110`，也相容舊的 `sm_110` 格式；其他 GPU 可覆寫 `CUDA_ARCH`。`DEEPSTREAM_DIR` 可指定 SDK 根目錄。原生程式使用獨立程序，避免 DeepStream 與 Python TensorRT 函式庫衝突。

CMake 建置快取預設放在 `dinov3_ws/build/native`，可用 `NATIVE_BUILD_DIR` 覆寫；產物仍放在 `artifacts/cuda`、`artifacts/nvdcf`。重複執行會跳過未變更的編譯，修改 `.cu` 後會更新函式庫與相依的 bridge。

Docker 先安裝依賴，再複製原始碼編譯，修改程式可沿用依賴安裝快取。系統安裝由 `cmake --install dinov3_ws/build/native --prefix /usr/local` 執行（需管理員權限），維持既有 `dinov3-*-bridge` 名稱。執行檔使用相對 RPATH 尋找共用函式庫，不依賴 Docker 編譯暫存目錄。

另建並驗證模型，不覆蓋原有單筆引擎：

```bash
dinov3_ws/.venv/bin/python dinov3_ws/build_models.py --target cat2real --max-batch-size 8
dinov3_ws/.venv/bin/python dinov3_ws/build_models.py --target sam3 --artifact-suffix native
```

驗證通過後寫入同名 `.json`，啟動時優先選用 `cat2real-fp32-batch8-trt.ts` 與 `vision-fp16-native.ts`。DINO 保持 FP32，支援 batch 1–8，僅合併本次影格已有的物件，不等待後續影格。沒有新引擎時沿用原有單筆版本。TensorRT 引擎應在實際部署 GPU 上重新建置。

`bbox3d_plane_backend: cpu` 可切回 Open3D RANSAC。疊圖統一走 CPU，發布 ROS Image 後由網頁 bridge 做背景 JPEG 轉換。GPU 前處理與 OpenCV/PIL 的插值可能有一個 RGB 灰階的差異，描述子與 SAM3 偵測框的數值驗證流程維持原樣。

2026-11-02 Thor 上相同已保存 RGB-D 的階段中位數：3D bbox `70.54 → 50.30 ms`，NvDCF 輸入與追蹤 `8.78 → 2.30 ms`，四個物件 DINO `180.25 → 160.52 ms`，SAM3 前處理 `28.16 → 4.43 ms`。3D 收益同時包含解析 Jacobian 與 CUDA RANSAC；各階段不能相加當成端到端收益。原始樣本數據在 `artifacts/gpu_optimization_20261102/stages.json`。

## 設定與排程

設定以 [dinov3_cases.yaml](config/dinov3_cases.yaml) 為準；以下是目前檔案值，而非節點內建預設值。

| 參數 | 目前值 | 用途 |
| --- | --- | --- |
| `grounding_prompt` | `box, can.` | 1–4 個逗號分隔類別，文字特徵按 prompt 快取 |
| `grounding_detection_interval` | 3 | 每 N 個處理影格提交背景 SAM3；1 改為同步且停用 NvDCF |
| `grounding_box_threshold` | 0.8 | SAM3 偵測門檻 |
| `grounding_containment_threshold` | 0.5 | 追蹤後去除被包含的小框；0 停用 |
| `recognition_threshold` | 0.75 | DINO cosine 候選門檻，再以 Lab 色彩相似度選分類 |
| `nvdcf_width`／`nvdcf_height` | 640／384 | 追蹤解析度，須為 32 的倍數 |
| `nvdcf_max_missed_detections` | 5 | 已提交 SAM3 結果連續漏偵測的淘汰門檻 |
| `nvdcf_max_missed_frames` | 15 | 失追寬限影格，期間隱藏舊框；0 停用 |
| `bbox_smoothing_frames` | 1 | 框移動平均視窗；1 停用 |
| `bbox3d_interval` | 1 | 背景深度量測間隔 |
| `bbox_visualization_mode` | `3d` | 疊圖模式，亦可選 `2d` |

包含框過濾在每次追蹤更新及新舊 context 合併後，使用目前框座標計算「交集面積 ÷ 較小框面積」；達到門檻便移除小框及其追蹤身分。IoU NMS 的參數與邏輯已移除。設定修改後需重啟服務。

NvDCF 原生參數位於 [nvdcf.yaml](config/nvdcf.yaml)，可用 ROS `nvdcf_config` 指定其他檔案。追蹤影像縮放不改變 SAM3、Cat2Real 裁切與深度處理的原圖座標。

非同步流程先交接 SAM3 框，再更新 DINO 分類，期間前景 NvDCF 持續追蹤。交接只更新仍存活的 seed owner，不回退目前框。漏偵測物件可保留原 context；失追物件在寬限內隱藏、重試偵測並保留公開 ID。配對是幾何關聯，沒有長期 ReID。

`recognition_margin` 為舊 API 相容欄位，已不影響 Lab 分類，因此未列入 YAML。`score` 是所選參考圖的 DINO 分數；`margin` 回傳值是候選類別間的 Lab 分數差。

## 驗證與程式導覽

[測試說明](tests/README.md) 列出目前回歸與 GPU 整合檢查。[程式導覽](src/dinov3_cases/README.md) 說明模組責任、影格租約、3D 深度框、ROS topics 與 ROS Image／MJPEG 介面。

API `result_fps` 只計不同 RGB 影格的完成率；背景結果刷新不增加此計數。SAM3、NvDCF、3D、發布與 JPEG 轉換耗時應分別判讀，模型單獨測速不代表完整流程 FPS。
