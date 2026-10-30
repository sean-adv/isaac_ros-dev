# 測試與診斷

測試涵蓋目前的 SAM3、NvDCF、Cat2Real、3D 深度框、圖庫、ROS 與 HTTP／H.264 流程。下列指令均從工作區根目錄執行。

## 回歸測試

```bash
source /opt/ros/jazzy/setup.bash
export PYTHONPATH="$PWD/dinov3_ws/src/dinov3_cases:${PYTHONPATH:-}"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 dinov3_ws/.venv/bin/python -m pytest -q dinov3_ws/tests
```

使用專案 `.venv` 與系統 ROS 訊息套件，不需啟動相機或載入部署引擎。停用 pytest 自動外掛載入可避免 ROS launch-testing 的額外依賴。`ipc_fake_worker.py` 是 IPC 測試的必要子程序 fixture。

| 範圍 | 測試 |
| --- | --- |
| NvDCF 非同步交接、DINO 延後分類、失追／漏偵測與資源回收 | `test_async_nvdcf.py` |
| 幾何配對、公開 ID、生命週期與框平滑 | `test_tracking.py` |
| SAM3 prompt、官方前處理、原圖框與分數 | `test_sam3.py` |
| Cat2Real 前處理、patch 特徵、圖庫與 DINO／Lab 分類 | `test_cat2real.py`、`test_patches.py`、`test_reference_gallery.py`、`test_grounded.py`、`test_lab_similarity.py` |
| 分類快取與節點辨識流程 | `test_tracked_recognition.py` |
| 共享影格租約、IPC 失敗／逾時與清理 | `test_frame_pool.py`、`test_grounding_ipc.py` |
| 深度、3D 框、點雲篩選及背景刷新 | `test_depth_*.py`、`test_bbox3d_fit.py`、`test_cloud_filter.py`、`test_async_bbox3d*.py` |
| ROS 訊息、疊圖、即時排程與影像串流 | `test_apriltag.py`、`test_bbox_visualization.py`、`test_realtime.py`、`test_h264_stream.py`、`test_timing.py` |
| 圖庫管理、相機擷取與前端互動 | `test_image_manager.py`、`test_camera_capture.py`、`test_images_ui.cjs` |
| TensorRT 輸入與缺失產物處理 | `test_trt_backend.py` |

前端互動測試需要 Node.js：

```bash
node --test dinov3_ws/tests/test_images_ui.cjs
```

## GPU 整合檢查

需實際 GPU、ROS 與模型。先載入上述 ROS／PYTHONPATH，再設定動態函式庫：

```bash
export LD_LIBRARY_PATH="$PWD/dinov3_ws/.venv/lib/python3.12/site-packages/tensorrt_libs:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
dinov3_ws/.venv/bin/python dinov3_ws/tests/check_gpu_environment.py
dinov3_ws/.venv/bin/python dinov3_ws/tests/check_official_pipeline.py
dinov3_ws/.venv/bin/python dinov3_ws/tests/check_async_nvdcf_pipeline.py
dinov3_ws/.venv/bin/python dinov3_ws/tests/check_grounded_trt_pipeline.py
```

`check_official_pipeline.py` 檢查 FP32 bbox、1–4 類別、Cat2Real 圖庫、IPC parity 與模型來源；`check_async_nvdcf_pipeline.py` 驗證 DINO 阻塞時仍先輸出框並持續追蹤；`check_grounded_trt_pipeline.py` 檢查 HTTP／ROS 與編碼阻塞時的框發布。共同測試影像為 `artifacts/validation/person.jpg`。

原生 NvDCF 的基礎功能、跨 context 交接與遮擋恢復可分別執行：

```bash
dinov3_ws/.venv/bin/python dinov3_ws/tests/check_nvdcf.py
dinov3_ws/.venv/bin/python dinov3_ws/tests/check_nvdcf_handoff.py
dinov3_ws/.venv/bin/python dinov3_ws/tests/check_nvdcf_occlusion.py
```

模型數值驗證使用 `dinov3_ws/.venv/bin/python dinov3_ws/build_models.py --validate-only`；不會重新編譯。

## 效能診斷

保留的工具依問題選用，使用相同環境，執行 `--help` 查看必要參數：

- `profile_camera_input.py`：相機接收 FPS、影格年齡與 QoS。
- `profile_camera_pipeline.py`：真實相機與各模型階段耗時。
- `profile_grounding_stages.py`：儲存影像的 SAM3 worker 與共享記憶體耗時。
- `profile_live_localhost.py`：本機服務狀態與串流量測。
- `profile_depth_bbox.py`：指定基準程式與儲存深度，對照 3D 框計算。

報告輸出至 `artifacts/`。相機診斷需與服務使用相同 Linux 使用者及 QoS；目前部署使用 RELIABLE、depth=1。效能數值須配合場景與負載判讀。
