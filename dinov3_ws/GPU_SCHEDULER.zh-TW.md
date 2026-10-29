# SAM3 GPU 排程與量測（2026-10-28）

此版本預設關閉，沒有更換 Engine、調整 GPU 時脈、修改 ROS2 Executor 或追蹤演算法。
目標是減少「已接入的其他 GPU 工作」對 SAM3 的干擾；不保證硬即時或所有負載的 P99 改善。

## 實際架構與重疊判斷

| 路徑 | 程序／執行緒 | GPU／Stream／同步 | 本版本接入 |
|---|---|---|---|
| SAM3 `sam3_trt.py` | `grounding_ipc` 獨立 Python 子程序；socketpair RPC 串行，detector 有 RLock | TorchScript Torch-TensorRT FP16 vision；PyTorch text encoder/projection、bbox head；使用呼叫執行緒 current stream，沒有手動建立 stream；`.cpu()` 取得輸出會等待 | 高優先，包含文字 GPU、H2D、vision、head、scores、D2H |
| Cat2Real/DINO `encoder.py`、`trt_backend.py` | ROS2 主程序的背景辨識工作；backbone 有 instance Lock | Torch-TensorRT FP32；current stream；finite 檢查轉 Python bool 隱含等待；原 Lock 並非跨程序協調 | 一般優先，GPU normalization、TRT、patch features 都涵蓋 |
| Gallery／診斷 `grounded.py`、`object_visualization.py` | 背景辨識／render／HTTP 查詢使用的快照 | normalize、stack/cat、cosine matmul、patch similarity；原本未統一協調 | 一般優先，以小段 GPU 工作接入，CPU 圖像處理不占 GPU 執行權 |
| Depth `async_bbox3d.py`、`depth_bbox.py` | `depth-bbox` thread；一個執行中與一個最新工作 | NumPy、Open3D CPU geometry；此路徑未見 CUDA 提交 | 不鎖住 CPU depth/tracking |
| KCF `async_kcf.py` | 主工作迴圈 + detector refresh thread | OpenCV CPU tracking；SAM3 新結果另行回放追上最新幀 | 不修改 |
| Camera／publish／render | `rclpy.spin` 收件；多個 publisher/render thread | callbacks 保留最新訊息；H264 可選 NVENC 或 CPU libx264，NVENC 不等同 TensorRT CUDA 推論 | 不鎖 camera/control callback |
| `qwen_encoder.py` | 歷史 PyTorch BF16 image embedding adapter，主 node 沒有引用 | SDPA、current stream，finite 檢查；目前 runtime 已沒有其 QWEN 常數，無法直接匯入 | 非目前運行路徑，未修改；啟用前須修復並接入 |
| `build_dinov3.py::Engine` | legacy 建模／驗證工具，非目前 node 使用的 backend | 明確 `torch.cuda.Stream()`、`execute_async_v3()`；`run(synchronize=False)` 允許非同步回傳 | 建模工具未接入；不要在延遲驗證時另外執行 |
| `tests/check_dinov3_onnx.py`、`check_dinov3_q4_gpu.py` | 獨立 ORT CUDA 實驗程序 | ORT 自有 stream／I/O binding；Q4 GPU 計時會 synchronize outputs | 非 runtime，未接入 |
| `DART/` | 舊 demo 與 vendored SAM3/DART runtime，主 node 未匯入 | eager／TRT／compiled 路徑；`sam3/video_pipeline.py` 有多 stream、double-buffer、split backbone；`dartf/runtime/cuda_alloc.py` 有 cudart stream/event wrapper | 非本次 production adapter；需另接入，不能視為已受控 |
| RViz2、nvblox、Xorg／其他容器 | 其他程序 | 其他 CUDA／graphics 工作 | 不受此協同排程控制 |

**關鍵：主要 production 路徑原本就依序執行 SAM3 → DINO。**
`node.py` 把 `detect`、`recognize` 傳入 `AsyncKCFProposals`；
`async_kcf.py::_job()` 先等待 SAM3 RPC 完成，再執行 gallery/recognizer。
首次同步初始化也是同樣順序。`async_recognition.py` 是另一個已有的背景 adapter，
但目前 node 並未使用它。HTTP camera capture 可另外呼叫 SAM3，render 可提交診斷 GPU 工作。
因此「不同程序」只能說明存在重疊可能，不能直接證明主要模型流程有重疊。
若僅執行原有依序流程，Scheduler 的預期效益有限，建議維持關閉。

Benchmark B/C **刻意增加獨立 DINO 背景提交**，用來驗證壓力與協調效果，不代表原流程的負載分布。
`gpu_start_ns` 到 `gpu_completed_ns` 是同一主機 monotonic clock 上的 GPU 工作窗口，
包含實際 enqueue／等待完成，可用於判斷並行 outstanding work；它不是硬體 kernel busy interval。
在低頻實測中，B 的 100 次 SAM3 有 22 次窗口與 DINO 重疊，C 為 0 次。
本機 Nsight Systems 2025.3.2 的 trace 只有主程序 CUDA 記錄，缺少 SAM3 子程序，
因此無法藉此確認跨程序 kernel 是否物理同時執行。不可把缺少第二程序的 trace 解釋為沒有重疊。

## 排程語意與故障處理

`gpu_scheduler.py` 是 stdlib-only Unix Domain Socket broker；每個 GPU 區段建立一個 lease 連線。
所有參與程序／容器必須連到**同一個 broker 與同一個 socket inode**。
SAM3 優先於一般佇列；同類 FIFO。一般請求等待超過 `--max-normal-wait`（預設 0.5 秒）後，
優先獲得下一次執行權，避免持續 SAM3 請求造成飢餓。門檻不是執行 deadline；等待仍包含現有工作，
且老化的一般工作可能延後下一次 SAM3。

不做 kernel 搶占。SAM3 獲准前，已獲准的一般區段會執行到完成。
獲准後，其他接入點不能提交新 GPU 工作，直到持有者確認完成並明確 release。
同執行緒 nested region 可重入，避免 encoder → backbone 重複取得 lease。
這是單一物理 GPU 的協調域；不要用同一 domain 跨不同 GPU 混排。

使用 `torch.cuda.synchronize(device)` 在正常與 Python exception 路徑等待**目前程序該裝置的所有 streams**，
再 release；不會因 Torch-TensorRT forward 或 `execute_async_v3()` 返回就釋放。
此處保守涵蓋 runtime 可能使用的 auxiliary streams，代價是增加同步且可能等待同程序其他 CUDA 工作。
CUDA Event 的 current-stream 時間需與這個 all-stream completion guard 區分。
參考：[PyTorch synchronize](https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize)、
[CUDA stream semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)、
[NVIDIA TensorRT multi-stream](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html)。

- Broker 不可達／等待 timeout：推論回報錯誤，不自動繞過排程；等待者斷線會移出佇列。
- Python 推論 exception：先同步已提交工作，再釋放，原例外向外傳遞。
- 持有者斷線／被 kill／CUDA sync 失敗：broker 進入 poisoned 狀態，後續請求立即失敗。
  不能從 socket EOF 推論 GPU 已停止。此處選擇安全停止，不做可能仍有 kernel 存活的自動 lease 過期。
- Broker 掛掉：等待者錯誤返回；持有者完成後無法確認 release 也會回報錯誤。
  不要在仍有舊持有者時另啟新 broker，否則可能出現兩個協調域。
- CUDA driver 內部永久卡住，Python timeout 無法中斷 `cudaDeviceSynchronize`；需外部程序監督／人工處理。
  這不阻塞 ROS2 camera callback 或 CPU tracking，但辨識可能無新結果。
- 復原：先停止所有參與推論的程序，確認其 CUDA context 已消失，再停止舊 broker、移除 stale socket、重啟。
  不可只刪 socket 後在原推論程序仍運行時重建 broker。

此版本不處理載入 Engine、CUDA context 初始化／模型 warmup 與未經 adapter 的低階直接呼叫。
應先完成載入，再進入延遲量測；新增背景 GPU 程序或其他模型時，必須另外接入。

## 啟動、關閉與容器

以下命令在既有開發容器／workspace 根目錄執行，與 ROS 程序使用相同 UID。
不需安裝新套件，不修改 `compose.yaml`。

```bash
export PYTHONPATH="$PWD/dinov3_ws/src/dinov3_cases:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$PWD/dinov3_ws/.venv/lib/python3.12/site-packages/tensorrt_libs:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
mkdir -p /tmp/dinov3-gpu-$UID
chmod 700 /tmp/dinov3-gpu-$UID
# Terminal 1：保持前景；Ctrl-C 關閉
./dinov3_ws/.venv/bin/python -m dinov3_cases.gpu_scheduler \
  --socket /tmp/dinov3-gpu-$UID/scheduler.sock --max-normal-wait 0.5
```

```bash
# Terminal 2：node 與它啟動的 SAM3 worker 繼承同一設定
export DINOV3_GPU_SCHEDULER=/tmp/dinov3-gpu-$UID/scheduler.sock
export DINOV3_GPU_TIMEOUT=30
# optional：省略即不建立 CUDA timing events 或寫 JSONL
export DINOV3_GPU_METRICS=/tmp/dinov3-gpu-metrics
./dinov3_ws/run.sh
```

回退：停止這次啟動的 node／worker，`unset DINOV3_GPU_SCHEDULER DINOV3_GPU_METRICS DINOV3_GPU_TIMEOUT`，
照原命令重新啟動。兩者皆未設定時不新增 CUDA 同步／Event／socket。推論 API、proposals 與 tensor 格式不變。
既有已運行的 node 不會自動接上新程式或環境變數；本次沒有重啟它。

跨容器：共享掛載**socket 目錄**到各容器（不是只掛單一 socket 檔案），讓同 UID 使用者可存取，
各自設定指向該掛載內的 `DINOV3_GPU_SCHEDULER`，並安裝／使用這份 adapter。
不同路徑可指向同一共享目錄；只要是同一 socket inode 即可。`ipc: host`、`network_mode: host`
並不自動共享 `/tmp` socket。根目錄的 host `run.sh` 不會自動傳遞上述 env 給 `docker exec`，
應在容器內設定，或在 `docker exec -e ...` 明確傳入。

## 量測定義

設定 `DINOV3_GPU_METRICS` 後，每個程序各自 append `<pid>.jsonl`，SAM3 數值也經原 IPC 回到
`last_metrics`，在現有 status 的 `performance.grounding_ipc` 可取得（依現有發布節奏更新）。

| 欄位 | 定義／限制 |
|---|---|
| `scheduler_wait_ms` | socket connect、排隊直到 grant 的主機時間；未啟用為 0 |
| `model_lock_wait_ms` | 同一 Sam3Detector instance 的 RLock 等待，與 scheduler wait 分開 |
| `cpu_preprocessing_ms` | prompt 驗證與 PIL/NumPy image resize/normalize，不含 H2D；文字 tokenization 僅在 prompt cache miss 發生，包含在 GPU 區段 |
| `trt_forward_host_ms` | Torch-TensorRT vision 的 Python/TorchScript forward 主機呼叫時間；包含 wrapper 開銷，**不是純 TensorRT enqueue** |
| `trt_vision_gpu_ms` | CUDA Events 包住 vision forward，取 current-stream elapsed time；包含 stream 間依賴與干擾，非 kernel busy sum |
| `gpu_execution_ms` | CUDA Events 包住整個 SAM3 GPU 區段：text GPU/cache miss、H2D、vision、head、scores、D2H；主機提交空檔也可能包含其中 |
| `completion_wait_ms` | 區段最後 all-stream synchronize 的主機等待；此前 `.cpu()` 已等待大部分工作，所以此值可能很小 |
| `postprocessing_ms` | NumPy finite、threshold、座標轉換、clean_proposals/NMS |
| `end_to_end_ms` | worker `detect()` 入口到完成 postprocessing；不含 ROS 收件、IPC 與 metrics 寫檔 |
| `rpc_e2e_ms` | benchmark 呼叫 parent detector.detect 到取得輸出，含影像準備／IPC；表格 SAM3 latency 使用此值 |
| `start_interval_ms` | 兩次取得執行權後、開始 GPU 區段的主機時間差；包含上一筆 CPU/postprocess 與本筆等待 |
| `queue_delay_ms`／`simulated_drops` | benchmark 的 10 Hz 最新幀來源模擬，非真實 ROS／DDS drop；同一張固定圖模擬更新 |

目前 production wrapper 沒有暴露原生 TensorRT execution context，故無法直接提供純
`execute_async_v3()` CPU enqueue time。沒有將 forward host time 偽標為該指標。
若要量到原生 enqueue，需要進入 Torch-TensorRT C++ runtime 或使用完整 Nsight trace，超出此最小版本。
亦未在此測試宣稱 camera capture-to-result 延遲或真實 DDS drop 已改善；production 可另記錄
`performance.callback_to_result_ms`、`processing_ms` 與 `grounding_schedule` 的 skipped/replay 欄位。

## 可重現 benchmark 與驗證

```bash
export PYTHONPATH="$PWD/dinov3_ws/src/dinov3_cases:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$PWD/dinov3_ws/.venv/lib/python3.12/site-packages/tensorrt_libs:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 HF_HUB_DISABLE_PROGRESS_BARS=1
./dinov3_ws/.venv/bin/python dinov3_ws/tests/benchmark_gpu_scheduler.py \
  --image dinov3_ws/data/gallery/solid_cold_medicine/c86bab2ada154565a380717df07b4005_bbox.png \
  --prompt 'medicine box' --samples 1000 --warmup 15 \
  --output /tmp/thor-gpu-bench
```

A：只建立 benchmark SAM3，scheduler 關閉。
B：SAM3 worker + 主程序 DINO 背景 thread（顯式獨立 stream），scheduler 關閉。
C：與 B 同負載，scheduler 開啟。`--case all` 自行建立 private broker，依序啟動全新 A/B/C 程序。
各模型 warmup 完才記錄，三組量測同樣啟用 CUDA Events 與 completion guard。
`--other-period 0` 為 DINO 飽和壓力；`--other-period 0.8` 代表每筆 DINO 完成後 sleep 0.8 秒，
不是精確固定提交率。`--camera-hz 0` 可關閉最新幀來源模擬，改為 closed-loop。

結果包含 A/B/C JSON（原始 rows、P50/P95/P99、吞吐量、Engine SHA256、GPU／worker版本、外部 GPU 程序）
及 `comparison.md`、每程序 events JSONL。使用全新 output 目錄，避免混入先前 JSONL。
正式結論需多輪、改變 A/B/C 次序（可用 `--case A/B/C` 個別執行，C 使用外部 broker），
固定相機／prompt／DINO 負載，記錄功耗模式、GPU/CPU 時脈、溫度，並在可控制的維護時段排除外部負載。
本次未關閉使用者既有的 RViz2、nvblox、SAM3 node，所以 A 只是本工具單模型條件，並非整張 GPU 無干擾。

```bash
# CPU、IPC、追蹤等回歸測試（ROS 環境需 source；關掉不必要的 pytest plugin autoload）
source /opt/ros/jazzy/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  ./dinov3_ws/.venv/bin/python -m pytest -q -p no:cacheprovider \
  dinov3_ws/tests/test_gpu_scheduler.py dinov3_ws/tests/test_trt_backend.py \
  dinov3_ws/tests/test_sam3.py dinov3_ws/tests/test_grounding_ipc.py \
  dinov3_ws/tests/test_grounded.py dinov3_ws/tests/test_tracking.py \
  dinov3_ws/tests/test_reference_gallery.py dinov3_ws/tests/test_async_kcf.py \
  dinov3_ws/tests/test_async_recognition.py dinov3_ws/tests/test_patches.py \
  dinov3_ws/tests/test_cat2real.py dinov3_ws/tests/test_bbox_visualization.py \
  dinov3_ws/tests/test_async_bbox3d.py dinov3_ws/tests/test_async_bbox3d_node.py

# 真實 CUDA 多 Stream 完成 + SAM3 raw tensors/DINO features 排程前後 bit-exact
DINOV3_TEST_GPU=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  ./dinov3_ws/.venv/bin/python -m pytest -q -p no:cacheprovider \
  dinov3_ws/tests/test_gpu_scheduler_cuda.py

# 額外 profiler run；不要把其延遲與未 profiling 的正式結果混用
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  -o /tmp/thor-gpu-overlap \
  ./dinov3_ws/.venv/bin/python dinov3_ws/tests/benchmark_gpu_scheduler.py \
  --case B --image /path/to/frame.png --samples 20 --warmup 5 \
  --output /tmp/thor-gpu-profile
nsys export --type sqlite -o /tmp/thor-gpu-overlap.sqlite /tmp/thor-gpu-overlap.nsys-rep
```

檢查 trace 必須同時有 parent 與 `worker.pid` 的 CUDA kernels，並排除 load/warmup 區間後再判斷。
不要把 CPU 呼叫重疊或 CUDA Event elapsed time 當成 kernel 物理重疊的證據。

本次 233 項相關回歸測試與 2 項真實 GPU 測試通過。SAM3 比對原始 boxes/logits/presence，
DINO 比對完整 patches/descriptor，不以「空 proposals 相等」替代數值測試。
既有 `test_sam3_preprocessing.py` 有三項因原版 `prepare_sam3_pixels` 沒有 CLAHE 參數而失敗；
既有 `test_qwen_encoder.py` 因 runtime 缺少 QWEN 常數而無法收集。兩者均未在本次改動中修復。
追蹤通過離線回歸，不等同已驗證排程改變更新節奏後的所有實機追蹤軌跡完全相同。

實測表與原始數據見 [測試報告](tests/GPU_SCHEDULER_20261028.md)。

## 修改範圍

- 新增 `src/dinov3_cases/dinov3_cases/gpu_scheduler.py`：socket broker、lease、CUDA completion guard、量測。
- 修改 `sam3_trt.py`、`encoder.py`、`trt_backend.py`：SAM3／DINO GPU 區段。
- 修改 `grounded.py`、`object_visualization.py`：辨識／診斷剩餘 GPU 工作。
- 修改 `grounding_ipc.py`：沿用原 RPC 傳回新 metrics，推論輸出不變。
- 新增 `tests/benchmark_gpu_scheduler.py`、`test_gpu_scheduler.py`、`test_gpu_scheduler_cuda.py`。
- 新增本文件、`tests/GPU_SCHEDULER_20261028.md` 與 `tests/results/gpu_scheduler_20261028/` 原始結果。

`.gitignore`、Engine、ROS2 callback、KCF／depth 演算法與既有運行服務均未改動。
