# Grounding DINO 舊環境 + DINOv3 新環境

```bash
cd /workspaces/isaac_ros-dev
./dinov3_ws/run_split_trt.sh
# 同時比較時使用另一個 HTTP port
./dinov3_ws/run_split_trt.sh --ros-args -p port:=8091
```

預設 HTTP port 為 8090；切換流程前先停止佔用同一 port 的程式。
相機沿用 `run_camera.sh` 與原有 ROS topic，網頁操作、圖庫、裁切與辨識流程相同。

| 程序 | Python 環境 | TensorRT | engine |
| --- | --- | --- | --- |
| ROS／HTTP／DINOv3 主程序 | `dinov3_trt11_venv` | 11.1.0.106 | `artifacts/tensorrt11/dinov3_vits16_fp16_224.engine` |
| Grounding DINO 子程序 | `dinov3_venv` | 10.16.2.10 | `artifacts/tensorrt/grounding_dino_wingdzero_fp16.engine` |

Grounding DINO 直接使用之前的 engine，不重新轉換，也不使用 AutoCast 版本。
子程序使用原本 ROS 推論環境 `dinov3_venv`；`groundingdino_venv` 是先前匯出／建置用環境。
DINOv3 使用 TensorRT-Model-Connect 建置的新 engine。參考圖與每個 ROI 固定
224×224，以等比例縮放與右／下方邊緣補齊處理；Grounding DINO 輸入仍為
800×1200、提示詞上限仍為 12 tokens。

## 影像傳輸與關閉

主程序將原始 RGB 放入可重複使用的共享記憶體，透過私人 Unix socketpair
傳遞共享記憶體名稱、尺寸、提示詞與 frame ID；子程序只回傳偵測框、標籤、
分數與時間。影像不經 JPEG 壓縮或 JSON/base64 傳輸。
這是 CPU 共享記憶體；仍包含 RGB 整理、記憶體複製與既有 GPU 輸入傳輸。

同時只有一筆偵測請求，主程序在收到回覆前不覆寫共享影像。
frame ID 必須相符；相機沿用主程序的最新影格欄位，避免累積待辨識影格。
單張 RGB 上限為 128 MiB，較大影像會明確報錯。

`grounding_ipc_timeout` 預設 60 秒，模型初次啟動至少允許 120 秒。
逾時、worker 中斷或 frame ID 不符時會清理 worker 和共享記憶體，顯示錯誤；
恢復方式是重新啟動流程。無效提示詞只拒絕該請求，不終止 worker。
正常關閉也會終止 worker 並釋放共享記憶體。

## 查詢與重現驗證

`/api/status` 的 `grounding_execution` 應為 `subprocess`；`grounding_worker`
提供子程序 PID、TensorRT 與 PyTorch 版本。
`performance.grounding_ipc` 提供本次偵測的 frame ID、影像位元組數和延遲：

- `worker_detect_ms`：原有 Grounding DINO 前處理、GPU 推論與後處理。
- `ipc_overhead_ms`：RPC 往返減去 worker 偵測時間，包含子程序讀取影像、排程與框回傳。
- `shared_copy_ms`：主程序寫入共享記憶體。
- `transport_total_ms`：共享記憶體寫入加 IPC overhead。
- `image_prepare_ms`：主程序取得 RGB numpy array 的時間。
- `input_and_transport_ms`：RGB 整理加 transport total。

暖機 GPU 量測與原環境結果比對：

```bash
source /opt/ros/jazzy/setup.bash
export PYTHONPATH="$PWD/dinov3_ws/src/dinov3_cases:${PYTHONPATH:-}"
LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu \
  dinov3_venv/bin/python dinov3_ws/tests/check_grounding_ipc_gpu.py --direct
dinov3_trt11_venv/bin/python dinov3_ws/tests/check_grounding_ipc_gpu.py
dinov3_trt11_venv/bin/python dinov3_ws/tests/check_grounded_trt_pipeline.py --split
```

量測報告為 `artifacts/tensorrt11/ipc-latency.json`，直接推論基準為
`ipc-direct-reference.json`。完整提示詞、短提示詞與旋轉圖片的框、標籤必須
一致，分數容許差異 1e-6；量測先暖機 3 次，再執行 20 次。
ROS／HTTP 整合報告為 `integration_split_trt_224.json`。

本機 Thor GPU 使用 1403×2464 RGB（10,370,976 bytes）樣本的暖機結果：

| 量測範圍 | 中位數 | P95 |
| --- | --- | --- |
| 共享記憶體寫入 + IPC overhead | 5.74 ms | 6.24 ms |
| RGB 整理 + 上述傳輸 | 10.49 ms | 13.22 ms |
| worker 原有 Grounding DINO 偵測 | 131.99 ms | 141.46 ms |
| 主程序完整 detect 呼叫 | 144.32 ms | 152.36 ms |

直接舊環境偵測的中位數為 128.80 ms，P95 為 140.20 ms。兩次為依序量測，
完整呼叫差異包含排程、輸入處理與推論波動，不應全部視為傳輸成本。
這是單一樣本的短期量測，不含 DINOv3、JPEG、相機曝光或瀏覽器顯示；
實際相機 FPS 仍需在目標解析度與物件數量下驗證。

新舊環境各 135 項單元測試通過；兩個基線已有收集問題的測試檔
`test_reference_alpha.py`、`test_rope.py` 維持排除。

## 回切

```bash
# 原本兩模型 TensorRT 10.16 流程
./dinov3_ws/run.sh
# 先前建立的兩模型 TensorRT 11.1 流程
./dinov3_ws/run_trt11.sh
```

上述啟動腳本與舊 engine 均保留；拆分入口是新增的 `run_split_trt.sh`。
