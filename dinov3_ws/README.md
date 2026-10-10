# Grounding DINO → DINOv3 商品辨識

文字提示找出物體框，裁切後以 DINOv3 ViT-S/16 特徵比對所有商品圖庫。
每個 case 是商品標籤；相似度或前兩名差距不足時顯示 `unknown`。

## 兩種執行方式

```bash
cd /workspaces/isaac_ros-dev
./dinov3_ws/run_camera.sh           # 需要相機時，另開終端
./dinov3_ws/run.sh --mode split     # 預設：分開程序、不同 TensorRT 環境
./dinov3_ws/run.sh --mode single    # 同一 ROS 節點、同一環境
```

兩個推論命令擇一執行。網頁預設 http://127.0.0.1:8090；同時比較時另設 port：
`./dinov3_ws/run.sh --mode single --ros-args -p port:=8091`。
網頁支援 ROS 相機與圖片上傳；`/images` 提供圖庫管理及 HSV 去背。
選取 case 只切換參考圖預覽，辨識仍比對全部圖庫。

| 模式 | ROS／HTTP／DINOv3 主程序 | Grounding DINO |
| --- | --- | --- |
| split | `dinov3_trt11_venv`，TRT 11.1.0.106 | `dinov3_venv` 子程序，TRT 10.16.2.10，原 engine |
| single | `dinov3_trt11_venv`，TRT 11.1.0.106 | 同一程序，11.1 混合 FP16／FP32 engine |

split 的 Grounding worker 是私人 IPC 子程序，不另建 ROS 影像 topic。
兩模式共用 DINOv3、ROI、圖庫、HTTP 與 ROS 輸出。
舊全圖熱圖偵測、PyTorch 推論及 ViT-B／ViT-H 執行入口已移除。
已識別物件現在可使用同步 RGB-D 估計可見表面的幾何姿態。
既有權重、ONNX、engine、processor、metadata 與 venv 均保留。

## 模型與設定

DINOv3 為 `facebook/dinov3-vits16-pretrain-lvd1689m`，使用 NVIDIA
`TensorRT-Model-Connect` 建置，FP16 計算、FP32 I/O。參考圖和 ROI 等比例縮放長邊至
224，再於右／下方複製邊緣，補成 **224×224**。
Grounding 使用 Wingdzero Swin-T OGC 匯出，固定 **800×1200**、直接 resize、**12 tokens**。
短提示詞補齊，超長提示詞拒絕；預設 `medicine bottle. medicine box. bottle. box.`。
single engine 使用原 ONNX 經 ModelOpt AutoCast，再以 11.1 建置；split 直接載入原 10.16 engine。

路徑集中在 `runtime.py`，以下路徑相對於 `artifacts/`：

| 用途 | engine |
| --- | --- |
| 共用 DINOv3 | `tensorrt11/dinov3_vits16_fp16_224.engine` |
| split Grounding | `tensorrt/grounding_dino_wingdzero_fp16.engine` |
| single Grounding | `tensorrt11/grounding_dino_mixed_fp16.engine` |

圖庫位於工作區的 `dinov3_cases_data/<case>/*`。
門檻、JPEG、推論頻率集中在 `config/dinov3_cases.yaml`，可用 `--ros-args -p` 覆寫。

## 架構與效能

| 模組 | 職責 |
| --- | --- |
| `runtime.py`、`launch_pipeline.py` | 模式、路徑、啟動 |
| `node.py` | ROS、最新影格、推論與發布執行緒 |
| `depth_pose.py` | RGB-D 同步檢查、深度分割、反投影與可見表面 PCA 姿態 |
| `web.py` | HTTP 路由、上傳、狀態與影像串流 |
| `encoder.py`、`trt_backend.py` | DINOv3 特徵與 TensorRT 執行 |
| `grounded.py`、`grounding_trt.py` | 偵測、ROI、圖庫、辨識、診斷 |
| `grounding_ipc.py` | 舊環境 worker、共享記憶體、影格 ID、清理 |
| `patches.py` | 特徵 normalization、patch 比對、網格 |
| `image_manager.py`、`templates/`、`static/` | 圖庫 API、HTML、CSS、JavaScript |

保留最新影格和單筆待發布結果，跳過過期資料；DDS 發布不阻塞推論。
有訂閱者才建立 ROS Image。MJPEG 有新 JPEG 才更新；診斷圖按需產生，使用同次推論
特徵，不重跑模型。輸出沿用 `/medicine_recognition/detections`、
`/medicine_recognition/debug_image` 與相容的 `/dinov3/heatmap/overlay`。

split 使用可重複使用的 CPU RGB 共享記憶體，socketpair 只傳名稱、尺寸、設定、frame ID
和偵測結果。同時只有一筆請求，回覆 ID 必須相符；影像不經 JPEG/base64。
RGB 上限 128 MiB；正常關閉、故障或逾時會清理 worker 和共享記憶體，故障後重啟流程。
`grounding_ipc_timeout` 預設 60 秒，初次載入至少允許 120 秒。

`/api/status.performance` 提供 detect、recognize、render、processing、FPS 與
callback→result 延遲，後者不含曝光、callback 前 DDS 傳輸與瀏覽器顯示。
split 的 `grounding_ipc` 另提供 RGB 整理、共享記憶體寫入、IPC overhead、worker 偵測時間；
`grounding_worker` 提供 PID 和版本。
先前 Thor、1403×2464 RGB、暖機後 20 次量測：傳輸中位數 5.74 ms／P95 6.24 ms，
含 RGB 整理為 10.49 ms／13.22 ms；實際相機 FPS 需依解析度與物件數量驗證。

## 已識別物件的 6D 姿態

`run_camera.sh` 預設啟用 `depth_registration:=true`，將 IR 雙目產生的深度對齊到
彩色影像。使用 `/camera/depth/image_raw` 和 `/camera/depth/camera_info`；
不需要另外開啟 IR 灰階影像串流。單獨的 IR 灰階影像不含公尺距離。

```bash
./dinov3_ws/run_camera.sh
# 另一個終端：
./dinov3_ws/run.sh --mode split
# 再開一個已 source /opt/ros/jazzy/setup.bash 的終端：
ros2 topic echo /medicine_recognition/poses
```

輸出型別為 `vision_msgs/msg/Detection3DArray`：

- `header`：原始 RGB 影格的時間戳與 optical frame。
- `detections[].results[].hypothesis.class_id`／`score`：商品標籤與辨識相似度。
- `detections[].results[].pose.pose.position`：`x, y, z`，單位公尺。
- `detections[].results[].pose.pose.orientation`：`x, y, z, w` 單位四元數。
- `detections[].bbox`：可見表面點雲的有向包圍框，尺寸單位公尺。

位置是**可見表面點雲的質心**；不是完整商品中心。旋轉由點雲 PCA 估計：
局部 X 為最長主軸，Z 為最小變異的表面法向、符號指向遠離相機的一側，
Y 形成右手座標系。X 的影像平面主要分量固定為正。相機 optical frame 為
X 向右、Y 向下、Z 向前。沒有 CAD 或物件姿態標定，這個幾何座標系無法代表
商品標準正面，也不能解決對稱物件的真實朝向；PCA 軸在不同視角可能切換。
訊息 covariance 使用未經標定的保守常數（位置對角 0.01 m²、旋轉 1 rad²），
不是感測器精度量測。沒有追蹤器，3D detection 的持續 ID 留空。

只對非 `unknown` 物件估計姿態。RGB／深度必須同尺寸、同 frame，時間差不超過
`pose_sync_tolerance`（預設 0.05 秒），並有匹配的相機內參。
推論前保留深度快照，避免模型處理期間誤用後來的影格。
`16UC1` 由毫米轉公尺；`32FC1` 視為公尺，其他編碼拒絕。
框中心深度作為種子，保留同深度帶內相連區域，排除零值、NaN、超距和背景。
點數不足或主軸不明確（例如圓形／正方形表面）時不發布該物件的 6D 姿態。
辨識框中心落在背景、重疊物件或背景與物件深度接近時，這種分割仍可能誤估，
需以實機資料確認。所有物件的處理狀態可在 `/api/detections` 的 `pose_status`
查看，成功者另含 `pose`。

每次辨識結果均可發布空陣列，表示當次沒有可用姿態；原本的 2D 辨識輸出保留。
上傳圖片沒有對應深度，因此只提供 2D 辨識。
參數在 `config/dinov3_cases.yaml`：距離範圍預設 0.1–5 m、深度帶 ±0.08 m、
至少 80 點、相鄰 PCA 特徵值比例至少 1.15。可用 `--ros-args -p` 覆寫，例如
`-p pose_enabled:=false` 停用姿態，或 `-p pose_depth_band:=0.12` 調整分割深度帶。

## 驗證與重建

```bash
source /opt/ros/jazzy/setup.bash
source dinov3_trt11_venv/bin/activate
export PYTHONPATH="$PWD/dinov3_ws/src/dinov3_cases:${PYTHONPATH:-}"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q dinov3_ws/tests
python dinov3_ws/tests/check_grounded_offline.py
python dinov3_ws/tests/check_grounded_trt_pipeline.py --mode split
python dinov3_ws/tests/check_grounded_trt_pipeline.py --mode single
python dinov3_ws/tests/check_dinov3_trt11.py
```

GPU 報告存於 `artifacts/tensorrt11/`。IPC 比對先用舊 Python 建基準，再用新 Python：

```bash
LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu \
  dinov3_venv/bin/python dinov3_ws/tests/check_grounding_ipc_gpu.py --direct
python dinov3_ws/tests/check_grounding_ipc_gpu.py
```

保留必要的 Grounding 重建腳本：`build_grounding_dino_wingdzero_trt.sh` 使用原上游與
`groundingdino_venv` 重建 10.16 engine；`build_grounding_dino_trt11.sh` 使用現有
`calibration.npz`、原 ONNX、ModelOpt 和新 venv 重建 11.1 engine。
DINOv3 轉換來源位於工作區的 `TensorRT-Model-Connect`，engine JSON 保存模型 revision
和建置資料；套件快照為 `artifacts/tensorrt11/requirements-inference.lock.txt`。
