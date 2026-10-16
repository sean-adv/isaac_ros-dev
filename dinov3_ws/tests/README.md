# 測試與診斷

此目錄只保留目前 SAM3 + DINOv3 流程的回歸測試，以及部署驗證與效能診斷工具。舊 OmDet 的離線 prompt、FP16/FP32 精度比較、IPC GPU 比較與舊版整合腳本已移除；原始量測報告仍保留。

## 日常回歸測試

在工作區根目錄執行，不需啟動相機或載入部署引擎：

```bash
source /opt/ros/jazzy/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="$PWD/dinov3_ws/src/dinov3_cases:$PYTHONPATH" \
dinov3_trt11_venv/bin/python -m pytest -q dinov3_ws/tests
```

使用專案 TRT11 Python 環境，並先載入 ROS 2 Jazzy；若 `isaac_ros_apriltag_interfaces` 安裝於額外的工作區，還需載入該工作區環境。保留既有 `PYTHONPATH`，讓 ROS 訊息可匯入。停用自動載入 pytest 外掛，是為了避免未使用的 ROS launch-testing 外掛引入額外依賴；本套測試不依賴外掛。

| 測試 | 保留原因 |
| --- | --- |
| `test_sam3.py` | Prompt 驗證、分數、原圖座標與 head 輸入 |
| `test_grounded.py` | 物品辨識、unknown、框過濾、圖庫更新與熱圖 |
| `test_patches.py`、`test_reference_gallery.py` | DINOv3 前處理、patch 順序與參考圖對齊 |
| `test_trt_backend.py`、`test_build_dinov3.py` | 引擎輸入限制、建置產物與驗證證據 |
| `test_frame_pool.py`、`test_grounding_ipc.py` | 共享影格生命週期、子程序失敗、逾時與清理 |
| `test_realtime.py` | 最新影格排程、發布與繪圖互不阻塞 |
| `test_depth_bbox.py`、`test_depth_sync.py` | 深度同步、有效座標、3D 框與 ROS 訊息 |
| `test_apriltag.py` | Isaac ROS 訊息序列化、座標及空結果發布 |
| `test_bbox_visualization.py` | 2D/3D 框投影與首頁影像切換 |
| `test_image_manager.py`、`test_camera_capture.py` | 圖庫、去背、相機擷取與 bbox 預覽 |
| `test_tracking.py` | bbox 移動平均、跨偵測配對與重設；可選 ROI tracking 的 ID、門檻與生命週期 |
| `test_kcf.py` | SAM3 間隔、KCF 框更新、失敗補偵測、重設與實際 OpenCV 追蹤 |

`ipc_fake_worker.py` 是 IPC 測試必要的子程序 fixture，不能當作一般診斷腳本刪除。

前端互動測試另需 Node.js：

```bash
node --test dinov3_ws/tests/test_images_ui.cjs
```

這組測試檢查手動 prompt 輸入、輪詢保留文字、擷取驗證、重複點擊與 bbox 預覽。

## 部署整合驗證

以下工具需要實際 GPU、模型與 ROS 環境，修改引擎或部署流程時才執行，不納入日常 pytest：

- `check_gpu_environment.py`：檢查 CUDA driver、PyTorch allocation 與 TensorRT Builder。
- `check_sam3_trt_pipeline.py`：比較 SAM3 TRT10 direct／IPC 結果，並驗證 DINOv3 crop 特徵。
- `check_grounded_trt_pipeline.py`：實際雙環境 HTTP／ROS 流程，以及刻意阻塞 JPEG 時的 bbox 發布。

從工作區根目錄執行：

```bash
LD_LIBRARY_PATH="$PWD/dinov3_trt11_venv/lib/python3.12/site-packages/tensorrt_libs:/usr/local/cuda/lib64" \
PYTHONPATH="$PWD/dinov3_ws/src/dinov3_cases:$PYTHONPATH" \
dinov3_trt11_venv/bin/python dinov3_ws/tests/check_sam3_trt_pipeline.py
```

其他部署工具使用相同環境，替換腳本名稱即可。工具會載入實際模型並寫出驗證報告，執行前可查看腳本中的輸出位置。

## 效能診斷

保留三個不同層次的量測入口，按問題選用：

- `profile_camera_input.py`：只看 ROS 相機輸入的 FPS 與影格年齡。
- `profile_grounding_stages.py`：以儲存影像量測 SAM3 worker 與共享記憶體傳輸。
- `profile_camera_pipeline.py`：量測真實相機的整體辨識流程。

使用部署整合驗證的 Python 與環境，附上 `--help` 可查看參數。將結果輸出至 `dinov3_ws/artifacts/`，不放進測試程式目錄。
