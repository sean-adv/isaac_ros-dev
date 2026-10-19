# FPS 診斷與改善（2026-10-19）

已確認主要瓶頸是每幀同步執行的 KCF，以及背景數值運算與它的 CPU 資源競爭。背景執行可以避免直接等待模型或繪圖完成，但仍會共用 CPU、原生函式庫的執行緒池與記憶體頻寬。以下效能比較使用原始 1080p 解析度，保留 SAM3、Cat2Real、3D bbox 及兩路瀏覽器影像。另外獨立驗證 720p 與 1080p 的正式啟動流程。

實機場景為 4 個物件、4 類、16 張參考圖。SAM3 interval=30、3D interval=3、OpenCV threads=1。各組獨立暖機後量測 15–30 秒，並非同時執行；詳細統計見 [量測報告](artifacts/fps_diagnosis_20261019.json)。

| 設定 | 相機接收 FPS | 不同 RGB 結果 FPS | KCF 平均 ms |
|---|---:|---:|---:|
| 原設定 | 22.05 | 9.33 | 98.07 |
| 只增加 DDS SHM 至 64 MiB | 29.93 | 10.20 | 90.05 |
| 只限制 KCF ROI 為 256 px | 18.86 | 12.76 | 57.78 |
| KCF ROI 256 + DDS SHM 64 MiB | 29.90 | 14.35 | 58.22 |
| 再限制數值函式庫執行緒池 | 29.94 | 28.50 | 29.14 |

原本 KCF 對每個物件依序更新；4 個物件平均約 98 ms，已超過 30 FPS 的 33.3 ms 預算。對同一張實景、同樣 4 個物件交替測兩輪，原尺寸 KCF 中位數約 91–92 ms，256 px 工作區約 24–25 ms，靜態框最大座標偏移約 2 px。完整負載下，限制 OpenMP／BLAS 執行緒後，KCF 從約 58 ms 降至 29 ms；背景 3D 平均耗時也從約 186 ms 降至 111 ms。此實驗同時限制數個函式庫，不能把效果單獨歸因於其中一個。

相機輸入另有負載相關的問題。最初純訂閱約 16–21 FPS；重啟後，預設 DDS 在閒置時同樣可達 29.96 FPS，因此不能宣稱最初低速全部由 SHM 大小造成。不過兩組完整負載測試使用 64 MiB SHM 均維持約 30 input FPS，而預設值在負載下約 19–22 FPS。Fast DDS 官方也指出 [SHM 過小會增加資料遺失風險](https://fast-dds.docs.eprosima.com/en/2.x/fastdds/transport/shared_memory/shared_memory.html)。

## 正式 run.sh 與 HTTP 串流驗證

用正式 `run.sh` 在測試埠 18090 啟動，實際消費 `camera.mjpg`、`overlay.mjpg` 各一路，暖機後量測約 20 秒；兩種解析度皆為 4 個物件，未記錄到串流錯誤。結果 FPS 為 API 的 3 秒滑動平均取樣值，串流 FPS 為 HTTP 接收影格數；不是遠端瀏覽器繪製效能。

| RGB 解析度 | input FPS 平均 | result FPS 平均 | 預覽 HTTP FPS | 框線 HTTP FPS |
|---|---:|---:|---:|---:|
| 1280×720 | 29.96 | 27.56 | 30.00 | 27.68 |
| 1920×1080 | 29.96 | 27.96 | 29.98 | 27.33 |

1080p 測試明確傳入 `run_camera.sh color_width:=1920 color_height:=1080`。相機啟動檔在測試期間另有解析度調整；本次效能修改保留最新解析度設定，不將降解析度作為改善手段。不同解析度會改變偵測框、追蹤細節及點雲工作量；這次結果接近，不能推論 1080p 普遍快於 720p。

## 已套用設定

- `config/dinov3_cases.yaml`：`kcf_max_roi_side: 256`。只縮放 KCF 工作區，框座標會換回原圖；背景補追也用相同設定。小目標會限制縮小比例，盡可能保留至少 16 px。SAM3、Cat2Real 與深度處理仍取得原圖。
- `run.sh`：在 Python 載入函式庫前，預設 `OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`MKL_NUM_THREADS=1`；SAM3 子程序會繼承。已有環境設定時保留使用者值。
- `run.sh`、`run_camera.sh`：預設載入 `config/fastdds_rgbd.xml`，共用記憶體區為 64 MiB，保留 UDP 供網路通訊／探索。呼叫端已指定 DDS profile 時不覆寫。

需要重啟相機與 `run.sh` 才能套用啟動環境；兩者建議使用同一個 OS 使用者。這次測試用 `admin`。Fast DDS 的共用記憶體受使用者權限影響，但切換使用者本身並未解決最初的低速。

```bash
# 相機終端：先停止原本相機程序，再啟動
bash dinov3_ws/run_camera.sh

# 辨識終端
bash dinov3_ws/run.sh

# 若特定動態／小物件場景需要原尺寸 KCF，可關閉工作區縮放
bash dinov3_ws/run.sh --ros-args -p kcf_max_roi_side:=0
```

縮小 KCF 工作區可能影響細節與快速運動的追蹤，不能將靜態 2 px 偏移視為所有場景的誤差上限。保留 256 作為此次場景的折衷；若要增加細節，可設 320，代價是較多 CPU 工作。無法保證所有物件數量與場景都穩定達到 30 FPS。

## 驗證

91 項 KCF、非同步交接、3D 排程與即時處理測試通過，包括原圖座標換算、真實 OpenCV 平移追蹤、小目標縮放限制，以及背景候選接手／切換相機後保留縮放設定。測試時停用無關的 pytest 自動外掛載入，避免 ROS launch-testing 外掛缺少 `lark` 影響收集。

```bash
source /opt/ros/jazzy/setup.bash
export PYTHONPATH="$PWD/dinov3_ws/src/dinov3_cases:${PYTHONPATH:-}"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 dinov3_ws/.venv/bin/python -m pytest -q \
  dinov3_ws/tests/test_kcf.py dinov3_ws/tests/test_async_kcf.py \
  dinov3_ws/tests/test_async_bbox3d_node.py dinov3_ws/tests/test_realtime.py
```
