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

## 環境與模型

需由系統提供 Python 3.12、ROS 2 Jazzy、CUDA driver/runtime、cv_bridge、vision_msgs、isaac_ros_apriltag_interfaces 與相機驅動。Python 相依套件與模型均位於 `dinov3_ws`，不需要相鄰模型專案。

```bash
bash dinov3_ws/setup_env.sh
sudo bash dinov3_ws/native/install_nvdcf.sh
```

`setup_env.sh` 先安裝 CUDA 13.0 Torch／torchvision，再使用 [constraints-thor.txt](constraints-thor.txt) 安裝 [requirements-build.txt](requirements-build.txt)。可透過 `PYTHON_BIN`、`DINOV3_VENV` 指定 Python 與環境位置。

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

## 設定與排程

設定以 [dinov3_cases.yaml](config/dinov3_cases.yaml) 為準；以下是目前檔案值，而非節點內建預設值。

| 參數 | 目前值 | 用途 |
| --- | --- | --- |
| `grounding_prompt` | `box.` | 1–4 個逗號分隔類別，文字特徵按 prompt 快取 |
| `grounding_detection_interval` | 3 | 每 N 個處理影格提交背景 SAM3；1 改為同步且停用 NvDCF |
| `grounding_box_threshold` | 0.8 | SAM3 偵測門檻 |
| `grounding_containment_threshold` | 0.8 | 追蹤後去除被包含的小框；0 停用 |
| `recognition_threshold` | 0.8 | DINO cosine 候選門檻，再以 Lab 色彩相似度選分類 |
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

[測試說明](tests/README.md) 列出目前回歸與 GPU 整合檢查。[程式導覽](src/dinov3_cases/README.md) 說明模組責任、影格租約、3D 深度框、ROS topics 與 H.264／WebRTC 介面。

API `result_fps` 只計不同 RGB 影格的完成率；背景結果刷新不增加此計數。SAM3、NvDCF、3D、發布與編碼耗時應分別判讀，模型單獨測速不代表完整流程 FPS。
