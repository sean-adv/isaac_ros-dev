# DINOv3 case backbone export

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

此工作只建立與檢查模型產物，現有 ROS 節點仍使用 PyTorch。
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
