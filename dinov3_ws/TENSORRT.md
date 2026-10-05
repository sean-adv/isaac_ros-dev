# DINOv3 TensorRT 推論接入

目前 backbone 是 `facebook/dinov3-convnext-large-pretrain-lvd1689m`，預設使用
`dinov3_convnext_large_dynamic_bf16.engine`，維持整張原圖推論。舊的 ViT-H+
engine 與新模型不相容；新產物使用不同檔名，避免誤載。

## 動態 engine 建立與啟動

在 `/workspaces/isaac_ros-dev`、既有 `dinov3_venv` 執行：

```bash
source dinov3_venv/bin/activate
python dinov3_ws/export_dinov3_dynamic.py
bash dinov3_ws/build_dinov3_dynamic_trt.sh
source /opt/ros/jazzy/setup.bash
cd dinov3_ws
python -m colcon build --symlink-install --packages-select dinov3_cases
./run.sh
```

動態產物位於 `artifacts/tensorrt/`：`dinov3_convnext_large_dynamic.onnx` 與 `.onnx.data`、
`dinov3_convnext_large_dynamic_bf16.engine`、同名 `.json` metadata，以及匯出／建置／整合驗證日誌。
engine 僅在節點啟動時載入一次，支援 FP32 `[1,3,H,W]`，高寬各 32–2560，
最佳化尺寸為 720×1280；輸出 FP32 `[1, floor(H/32)*floor(W/32)+1, 1536]`。
不縮放、不切圖、不補邊；不滿 32 像素的右／下緣仍沿用既有 -1 分數規則。
範圍外輸入會報錯，需擴大 profile 後重建或明確選擇 PyTorch。

建置允許 BF16、停用 TF32，I/O 維持 FP32。ConvNeXt 沒有 ViT 的 RoPE，
因此不再套用 `trt_rope_precision.py` 約束。

`trt_backend.py` 使用 GPU tensor 位址與 CUDA stream 執行 TRT，鎖住共用 context 並等待完成，
讓輸入 buffer 生命週期與形狀切換安全；輸出保留在 GPU。
`node.py` 的參考圖片與相機／上傳圖片都透過同一個 Encoder 呼叫此 adapter，
`patches.py` 的 FP32 正規化、相似度、熱圖映射，以及 bbox／深度姿態流程沿用。
網頁與 `/api/status` 顯示實際後端；TRT 內部 BF16 不會因 FP32 I/O 被誤標為純 FP32。

切換回 PyTorch（重啟後重新建立 bank）：

```bash
./run.sh --ros-args -p inference_backend:=torch
```

在工作區根目錄驗證新後端（需要 GPU、ROS 環境與本機臨時 HTTP 埠）：

```bash
source /opt/ros/jazzy/setup.bash
PYTHONPATH="dinov3_ws/src/dinov3_cases:$PYTHONPATH" \
  dinov3_venv/bin/python dinov3_ws/tests/check_trt_integration.py
```

測試包含 87×53、1280×720、1403×2464 原圖、有限值、超出 profile 拒絕、
參考庫重新建立、上傳、閾值更新、debug views、detections 及模擬相機回呼。
此整合測試不代表真實相機／深度姿態的現場精度驗證。

## ConvNeXt Large 接入驗證（2026-10-05）

42 項單元測試、PyTorch CPU/GPU 離線 ROS／網頁測試，以及 TensorRT 真實 GPU
整合測試通過。三種輸入對 PyTorch BF16 的空間特徵 cosine 如下（不是偵測機率）：

| 圖片寬×高 | 平均 | 最低 |
| --- | --- | --- |
| 87×53 | 0.998419 | 0.998211 |
| 1280×720 | 0.999479 | 0.984236 |
| 1403×2464 | 0.999649 | 0.985017 |

均無 NaN/Inf。更換 backbone 後仍應以實際目標重新校準 cosine threshold。

## 舊 ViT-H+ 接入驗證（2026-10-04，不適用於 ConvNeXt）

41 項單元測試、PyTorch CPU 離線 ROS／網頁測試，以及 TRT 真實 GPU 整合測試通過。
本次三種輸入對既有 PyTorch BF16 的 patch cosine 如下（不是偵測機率）：

| 圖片寬×高 | 平均 | 最低 |
| --- | --- | --- |
| 87×53 | 0.997791 | 0.995596 |
| 1280×720 | 0.999714 | 0.970141 |
| 1403×2464 | 0.999665 | 0.975351 |

均無 NaN/Inf。BF16 結果不是數值完全相同，部署時仍應比較實際目標的框與閾值表現。
1280×720 的 trtexec 短測 GPU 平均約 99.5 ms，只是 backbone 合成輸入量測，
不是完整 ROS 流程 FPS，也不與先前 448×448 的 17.8 ms 直接比較。

## 舊版 ViT-H+ 固定 448×448 產物（2026-10-03，不適用於 ConvNeXt）

Model: `facebook/dinov3-vith16plus-pretrain-lvd1689m`, using the local Hugging Face cache.

在 `/workspaces/isaac_ros-dev` 執行，使用既有 `dinov3_venv`：

```bash
source dinov3_venv/bin/activate
python -m pip install -r dinov3_ws/requirements-export.txt
python dinov3_ws/export_dinov3.py
bash dinov3_ws/build_dinov3_trt.sh
python dinov3_ws/verify_dinov3_trt.py
```

建立 engine 與驗證需能使用 NVIDIA Thor GPU。匯出使用 CPU，讀取已快取權重，不重新下載模型。

產物位於 `dinov3_ws/artifacts/tensorrt/`，大型產物已排除於 Git：

- `dinov3.onnx`、`dinov3.onnx.data`：兩者必須一起保留。
- `dinov3_fp32.engine`：停用 TF32 的 FP32 基準。
- `dinov3_bf16.engine`：允許 BF16、保留 FP32 fallback，停用 TF32。
- `export.json`：來源模型 revision、輸入輸出與 opset。
- `export.log`、`build_fp32.log`、`build_bf16.log`：匯出與建置日誌；建置後包含短時間 trtexec 推論量測。
- `verification.json`：單張既有 stomach 參考圖片的有限值與特徵誤差檢查。

固定輸入 `pixel_values` 為 FP32 `[1, 3, 448, 448]`，需事先完成既有 processor 的 RGB 正規化。
輸出 `last_hidden_state` 為 FP32 `[1, 789, 1280]`，包含 1 個 CLS 與 4 個 register tokens。
去掉前 5 個 tokens 後，沿用現有 FP32 L2 normalization，可得到 28×28 patch 特徵。
batch 與空間尺寸皆固定；更改 `batch_size`、`tile_size` 或 `image_size` 需相應重新匯出。

這批固定尺寸產物僅供歷史驗證；目前 ROS 預設使用上述動態 TRT engine。
單張圖片檢查不代表完整 case matching 準確度驗證；接入 TensorRT 時應重建 reference bank，另行比較熱圖與框。
`trtexec` 量測只包含 backbone 執行，不是整張相機影像／ROS pipeline 的 FPS。
Engine 針對建置時的 GPU 與 TensorRT 版本，移到其他環境時應重新建置。

## 本次結果（2026-10-03）

NVIDIA Thor、TensorRT 10.16.2.10、PyTorch 2.14.0+cu130；ONNX checker 與兩個 engine 的 trtexec 推論皆成功。

| Engine | 大小約 | 單 tile GPU 平均時間 | 對 PyTorch FP32 的 patch cosine 平均／最小 |
| --- | --- | --- | --- |
| FP32（停用 TF32） | 3.2 GiB | 335.5 ms | 1.000000 / 0.9999996 |
| BF16 混合精度 | 1.6 GiB | 17.8 ms | 0.9999076 / 0.9991408 |

兩者輸出均無 NaN/Inf。相對現有 PyTorch BF16 後端，BF16 engine 的 patch cosine 平均為 0.999908、最小為 0.996170。
誤差量測只有一張 stomach 參考圖片；效能來自 trtexec 短測的合成輸入，BF16 時間變異係數約 11.8%，不宜視為穩定效能保證。
