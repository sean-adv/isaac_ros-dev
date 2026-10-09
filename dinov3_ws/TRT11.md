# TensorRT 11.1 試用流程

新流程使用獨立 `/workspaces/isaac_ros-dev/dinov3_trt11_venv`，固定 TensorRT
`11.1.0.106`。系統 TensorRT、`dinov3_venv`、`groundingdino_venv`、舊 engine 與
`run.sh` 保留。Grounding DINO 維持 800×1200 與 12-token 容量；224×224 是
DINOv3 的每張參考圖／ROI 輸入尺寸。

## 啟動與回切

另可用 `./dinov3_ws/run_split_trt.sh` 將 Grounding DINO 保留在舊 10.16 環境，
僅 DINOv3 使用 11.1，透過共享記憶體交換影像。詳見 [SPLIT_TRT.md](SPLIT_TRT.md)。

```bash
# 新流程：兩個模型都使用 TensorRT 11.1
./dinov3_ws/run_trt11.sh

# 舊流程：兩個模型都使用 TensorRT 10.16.2.10
./dinov3_ws/run.sh
```

兩個入口都預設使用 224×224 DINOv3 輸入：等比例縮放長邊至 224，右側和底部
複製邊緣補齊成正方形。`object_square_input=true` 控制補成正方形；不是將
原圖拉伸。熱圖移除補邊後映射回 ROI。圖庫啟動時重新建立，既有圖片不變。
舊 engine 本身仍保留原本的動態輸入範圍，只是目前應用統一餵入 224×224。
既有整體描述向量會包含補邊 patch，辨識門檻需要用真實商品驗證。

兩個入口預設都使用 8090 port，請先停止正在執行的流程再切換。
若需要同時比較，可指定不同 port：

```bash
./dinov3_ws/run_trt11.sh --ros-args -p port:=8091
```

恢復原本長邊 448、僅補齊 patch 網格的舊 grounded 前處理：

```bash
./dinov3_ws/run.sh --ros-args -p object_max_side:=448 -p object_square_input:=false
```

原始 YAML 與啟動腳本的內容備份在
`artifacts/tensorrt11/dinov3_cases_before_trt11.yaml` 和 `run_before_trt11.sh`。
腳本備份用於內容還原，請勿從備份目錄直接執行。

## 新產物

`artifacts/tensorrt11/` 與原本 `artifacts/tensorrt/` 分開保存：

- `dinov3_vits16_fp16_224.engine`：TensorRT-Model-Connect 的原始 engine.plan，
  FP16 計算／FP32 I/O；輸入 `[1,3,224,224]`，輸出 hidden state 與 pooler。
  loader 同時綁定兩個輸出，只將 hidden state 交給現有 patch 辨識流程。
- `grounding_dino_mixed_fp16.engine`：TensorRT 11.1 的 Grounding DINO engine。
- `grounding_dino_mixed_fp16.onnx`：NVIDIA ModelOpt AutoCast 顯式混合精度圖。
- `calibration.npz`：使用現有前處理、12-token 藥瓶／藥盒提示詞與一張 stomach
  圖庫照片建立的真實輸入。校準規模不代表完整資料集精度。
- `requirements-inference.lock.txt`：新推論環境套件版本快照。

## 重建 Grounding DINO

```bash
./dinov3_ws/build_grounding_dino_trt11.sh
```

此腳本重用已匯出的 `artifacts/tensorrt/grounding_dino_wingdzero.onnx`，先用
NVIDIA ModelOpt AutoCast 轉換精度，再由 TensorRT 11.1 建置。保留 FP32 I/O
與 normalization、softmax 和 reductions 的 FP32 邊界；不使用 TensorRT 11
已移除的 `BuilderFlag.FP16`。所有新輸出保存在 tensorrt11 目錄。

## 驗證命令

```bash
source /opt/ros/jazzy/setup.bash
source dinov3_trt11_venv/bin/activate
export PYTHONPATH="$PWD/dinov3_ws/src/dinov3_cases:$PYTHONPATH"
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/tensorrt_libs:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q dinov3_ws/tests \
  --ignore=dinov3_ws/tests/test_reference_alpha.py --ignore=dinov3_ws/tests/test_rope.py
python dinov3_ws/tests/check_grounded_offline.py
python dinov3_ws/tests/check_dinov3_trt11.py
python dinov3_ws/tests/check_grounding_wingdzero_trt.py \
  --engine dinov3_ws/artifacts/tensorrt11/grounding_dino_mixed_fp16.engine \
  --report dinov3_ws/artifacts/tensorrt11/verification-grounding.json
python dinov3_ws/tests/check_grounded_trt_pipeline.py --trt11
```

兩個排除的測試檔引用基線已不存在的函式，是既有收集問題。未放寬其餘測試
門檻。AutoCast 校準與 ONNX 數值比對使用 CPUExecutionProvider；實際應用使用
TensorRT GPU 推論。此 venv 中的 ONNX Runtime 不是 TensorRT ExecutionProvider。

套件複製與安裝都限於新 venv，ROS 從 `/opt/ros/jazzy` 載入。版本快照裡的
PyTorch/CUDA/aarch64 套件需使用相容 wheel 來源。NVIDIA cuSPARSELt 的官方 wheel
內部標記為 `sbsa`，可能使 `pip check` 報平台標記警告；這與已通過的 DINOv3
GPU 推論是分開的檢查。

## 本次驗證結果

- 新 venv：TensorRT 11.1.0.106、PyTorch 2.14.1+cu130、Transformers 5.2.0、
  NumPy 1.26.4；ROS Jazzy、rclpy、cv_bridge 和應用程式 import 成功。
- 新舊 venv 各 **129 項單元測試通過**；新環境的 ROS/HTTP 離線整合通過。
- 新 DINOv3 engine 經應用 loader 對照 PyTorch：196 個 patch，平均 cosine
  **0.99999046**、最低 **0.99985886**。
- Grounding DINO：完整提示詞、短提示詞補齊與旋轉圖片三組比對通過；NMS 後
  配對框 IoU 最低 **0.99885110**，分數最大誤差 **0.00160873**；過長提示詞拒絕通過。
- TensorRT 11.1 與保留的 10.16 流程均通過真實雙模型 ROS/HTTP 測試：
  圖庫辨識、每個 ROI 固定 224×224、逐物體熱圖、參考圖預覽與提示詞更新。
- 原始 `run.sh` 內容與 Git 基線一致；舊 Grounding DINO engine SHA256 與原 metadata 一致。

Grounding DINO AutoCast 約 85.59% 節點轉為 FP16；engine 初次編譯約 621 秒。
三組 Grounding DINO adapter 的小規模暖機量測約 92–95 ms，包含該 adapter 的
GPU forward，不包含完整辨識流程；不是相機 FPS 或可靠的版本效能比較。

混合 FP16 ONNX 無法直接在此 ONNX Runtime CPU 執行：其 CPU GELU kernel 不支援
FP16。正式精度驗證使用**原始 FP32 ONNX CPU 輸出對照新 TensorRT GPU 輸出**，
所有原有門檻均保留。Transformers 5.2 提示 Grounding DINO processor 預設改用 fast
版本；Wingdzero 影像前處理仍由既有 OpenCV adapter 執行，提示詞与後處理已納入上述驗證。

驗證紀錄位於 `artifacts/tensorrt11/`：`environment-validation.json`、
`unit-tests.log`、`legacy-unit-tests.log`、`offline-integration.log`、
`dinov3-adapter-validation.json`、`verification-grounding.json`、
`integration_grounding_wingdzero_vits16_224.json` 及各步驟日誌。
旧 10.16 整合結果位於 `artifacts/tensorrt/integration_grounding_wingdzero_vits16_224.json`。
圖片精度驗證僅涵蓋上述測試樣本，尚未進行完整商品資料集或真實相機效能評估。
