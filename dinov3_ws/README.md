# DINOv3 case heatmaps in my_isaac_ros_dev

ROS 2 Jazzy node using `facebook/dinov3-vith16plus-pretrain-lvd1689m` on NVIDIA Thor.

## First: Hugging Face access

Accept/request access at https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m using your own account. Then log in interactively inside the container (do not put your token in source code or chat):

```bash
docker exec -it my_isaac_ros_dev /workspaces/isaac_ros-dev/dinov3_venv/bin/hf auth login
```

## Run

Start the Gemini 336 driver in one terminal if it is not already running:

```bash
docker exec -it my_isaac_ros_dev bash /workspaces/isaac_ros-dev/dinov3_ws/run_camera.sh
```

此腳本以 1280×720、30 FPS 啟動 RGB 與深度，開啟 `depth_registration:=true` 並對齊到 COLOR。保留此終端，在另一個終端啟動 DINOv3（bbox 使用 RGB，最高分框的姿態使用深度）：

```bash
docker exec -it my_isaac_ros_dev /workspaces/isaac_ros-dev/dinov3_ws/run.sh
```

`run.sh` 直接啟動 DINOv3，訂閱 `/camera/color/image_raw`，不再啟動或依賴 RGB-D proposal。既有 `--ros-args -p ...` 會傳給 DINOv3，例如 `--ros-args -p port:=8091`。

更新後重建 `dinov3_cases` 即可；舊的 RGB-D build/install 產物不再使用。相機腳本仍接受 launch 參數覆寫，例如 `color_fps:=15 depth_fps:=15`。

Open http://localhost:8090 on the Jetson. For remote use, forward port 8090 over SSH. The service binds to loopback by default. `-p host:=0.0.0.0` exposes the unauthenticated interface and should only be used on a trusted network.

1. Enter a case name and select multiple pictures, then click **Add pictures to case**. Repeat for additional cases.
2. Select a case and click **Use selected case**.
3. Use the default ROS camera, or upload a target picture.
4. 相機預覽獨立顯示原始 RGB，選擇 case 後結果影像顯示 similarity heatmap 與所有合格連通區的 bbox。紅色表示較高 cosine similarity，並非偵測機率。

Reference pictures should be tightly cropped around the desired object. All reference patches, including backgrounds, can produce matches. This initial implementation matches the best reference patch from the selected group at each target patch; it does not train a classifier or produce verified object boundaries. Visually similar packaging can match even when text or medicine differs.

預設相機為 `/camera/color/image_raw`，可使用 `--ros-args -p image_topic:=/d457/d457/color/image_raw` 覆寫。

## ROS visualization

- `/medicine_recognition/detections`：全圖 mask 產生的合格框經 IoU 合併後輸出；每個合併框的 score 是組內最高的元件平均 cosine similarity。無物件時發佈空陣列。ID 僅為單幀元件編號，不是追蹤 ID。
- `/medicine_recognition/debug_image`：全圖熱圖與最終 bbox。
- `/dinov3/heatmap/overlay`: `sensor_msgs/Image`, BGR8; add an Image display in RViz2 and select this topic.
- `/dinov3/heatmap/similarity`: `sensor_msgs/Image`, 32FC1, raw cosine similarity.
- Camera timestamps and frame IDs are preserved.

## Data and parameters

最高分 bbox 使用框內匹配 mask 的對齊深度估計可見表面姿態，畫面顯示 X 紅、Y 綠、Z 藍三軸（預設長度 5 cm）。位置是有效表面點的平均中心，X 為最長主軸，Z 為朝向相機的表面法線，Y 補成右手座標系。這是幾何表面座標系，不是 CAD 物體中心或具有語義的正反朝向；對稱表面、遮擋及背景混入會限制姿態估計，沒有跨幀追蹤。

有效姿態發布到 `/medicine_recognition/pose`（`geometry_msgs/PoseStamped`），位置單位為公尺，旋轉使用四元數，header 沿用 RGB 的時間及光學座標系。RViz 可使用 Pose display 顯示。`/api/detections` 的最高分項新增 `selected: true` 和 `pose`（含 `position`、`quaternion_xyzw`、以列儲存且各欄為 X/Y/Z 的旋轉矩陣 `axes`、frame/stamp）；失敗時 `pose: null` 並附 `pose_error`。無有效姿態時不發布 PoseStamped；訂閱端應檢查時間戳，勿沿用舊姿態。上傳的單張 RGB 不會搭配即時相機深度。

深度參數：`depth_topic`（`/camera/depth/image_raw`）、`camera_info_topic`（`/camera/color/camera_info`）、`depth_sync_tolerance`（0.05 秒）、`pose_depth_band`（中位深度前後 0.04 公尺）、`axis_length`（0.05 公尺）。深度必須已對齊 RGB，且尺寸與 frame_id 一致；支援 16UC1 毫米或 32FC1 公尺，以及 plumb_bob / rational_polynomial 標定。缺少同步深度、標定、足夠有效點或可辨識主軸時，保留 2D 檢測並顯示原因。

Reference groups persist in `/workspaces/isaac_ros-dev/dinov3_cases_data/<case>/`. You may also create folders and add PNG/JPEG/WebP/BMP files there, then select the case in the browser. Uploaded images are converted to PNG; uploads append to a case. No deletion endpoint is provided. DICOM and volumetric scans are not supported.

Parameters: `cases_dir`, `image_topic`, `model_id`, `image_size` (448; multiple of 16, 224–1024), `inference_hz` (5.0), `max_reference_patches` (8192), `host` (127.0.0.1), `port` (8090).

Reference images use edge-replicated letterbox padding and exclude both padding and the first two valid patch rows/columns along the content boundary; extremely narrow references without interior patches are rejected. `image_size` controls references only. Target images use overlapping tiles at original resolution with a small replicated context border. Scores from the outer tile margin are excluded before merging, then the context border is cropped away. The reference feature bank is bounded and rebuilt when switching to changed case data. Only the latest camera frame is processed; actual rate depends on inference time. The model and selected bank stay in GPU memory. Reference files stay local; Hugging Face is contacted to download model files.

## Environment / rebuild

The venv is `/workspaces/isaac_ros-dev/dinov3_venv`, with system site packages enabled for ROS/OpenCV. Dependencies were installed without replacing system packages. To recreate after changing the container:

```bash
python3 -m venv --system-site-packages /workspaces/isaac_ros-dev/dinov3_venv
source /workspaces/isaac_ros-dev/dinov3_venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install 'transformers>=4.56,<6' flask pillow
source /opt/ros/jazzy/setup.bash
cd /workspaces/isaac_ros-dev/dinov3_ws
python -m colcon build --symlink-install
```

The model's separate license and Hugging Face access conditions apply to its weights. If startup reports HTTP 401/403, complete account approval/login and restart the node.

## Live stream troubleshooting

If the page says no recent camera frames, ensure `run_camera.sh` is running and the USB camera is connected. The camera driver and node must share the same `ROS_DOMAIN_ID` and DDS configuration. Refresh the browser after updating the node to load the live-preview interface. `/camera.mjpg` is the MJPEG preview endpoint; `/api/status` reports camera freshness separately from inference status.

Verified on the connected Gemini 336: RGB frame reception and successive MJPEG frames. The preview runs independently of the heatmap, whose speed depends on the model and inference rate.

## Cosine threshold

Use the **Cosine threshold** slider below the live preview. Release the slider to apply the value (0–1, default 0.70). Pixels below the threshold keep their original image color; pixels at or above it receive heatmap color. This also filters the ROS overlay; the raw similarity topic stays unfiltered. The next result uses the new value, including for uploaded static targets. Set the startup default with `--ros-args -p cosine_threshold:=0.8`. The slider value lasts for the current process.

## Tiling + batch inference → bounding boxes

原始 RGB → overlapping tiles → batch DINOv3 patch features → reference cosine similarity → 每張 tile 放大至 pixel map → overlap 合併 → **全圖** threshold → closing → hole filling → `connectedComponentsWithStats(connectivity=8)` → 面積過濾 → 原圖 bbox + padding → 可選 RGB 外圍微調 → bbox IoU 合併。

目標圖不先縮小。預設 448×448 tile、25% overlap、stride=336；推論前先在整張原圖四周各補 48 pixel 上下文，1920×1080 因此約 24 tiles。合併時忽略每張 tile 最外側 48 pixel 的分數，再裁去整圖補邊，避免 tile 上下緣的高分進入原圖。最後一列／排向內貼齊右／下邊界，完整覆蓋原圖；小於 tile 的尺寸在右／下以 replicate 補邊。tile_size 必須為模型 patch_size 的倍數；預設 patch_size=16，因此每張 tile 保留 28×28 patch grid，排除 CLS/register tokens。

`max` 取 overlap 區域最大 cosine 分數（保留負值）。`weighted` 使用中央較高的 2D Hann 權重，邊界權重保留正值，避免影像外圍除以零；累加分數×權重後除以總權重。所有 tiles 合併後才做 threshold 與連通元件，跨 tile 的同一物件可形成單一 bbox。無額外 detector 或 segmentation neural network。

GPU 上保留原圖 tensor、tile views、batch preprocessing、features、similarity 與 merge；只堆疊當前 batch，避免同時保留全部高維 features。使用 `torch.inference_mode()`；支援 BF16 的 CUDA 裝置採用 BF16 權重與 autocast，其他 GPU／CPU 使用 FP32，similarity/merge 使用 FP32。避免 FP16：正式 ViT-H+ 模型曾在正常相機輸入中溢位，造成部分 tiles 全為 NaN、整片熱圖消失。reference features 與最終全圖分數皆檢查有效性，有 NaN/Inf 就回報錯誤，不輸出誤導性的部分結果。網頁 status 顯示實際精度。一般模式只把最終全圖分數傳回 CPU 做 OpenCV 後處理；debug 模式額外一次傳回所有低解析度 tile 分數。batch_size 依 GPU 記憶體調整，最後一批不足時保留實際張數。

| ROS 啟動參數 | 預設 | 說明 |
| --- | --- | --- |
| `tile_size` | 448 | 原圖切圖尺寸；模型 patch_size 的正整數倍 |
| `overlap` | 0.25 | [0, 1)，且整數 stride 必須至少為 1 |
| `batch_size` | 4 | 每次 model forward 的 tile 張數上限，可改為 8 |
| `merge_mode` | weighted | `weighted` 或 `max` |
| `closing_kernel` | 3 | **原圖 pixel** 的奇數 kernel；1 停用 closing，仍填洞 |
| `min_area` | 1024 | 清理後連通元件的最小面積，單位 pixel² |
| `bbox_padding` | 8 | bbox 每側向外擴張的原圖 pixel 數 |
| `merge_iou_threshold` | 0.3 | 最終 bbox 的 IoU 達到此值時合併；取外接框、最高分與元件面積總和 |
| `debug_visualization` | false | 開啟網頁五種 tiling 除錯視圖 |
| `rgb_border_fraction` | 0.1 | bbox 外側搜尋帶相對寬／高；0 完全停用，範圍 0–0.25 |
| `edge_min_gradient` | 20.0 | 8-bit RGB 各通道 Sobel 絕對值最大值，scale=1/4 |
| `edge_min_support` | 0.6 | 邊上至少此比例超過 gradient 閾值，另須連續涵蓋至少 40% |

左右使用 x gradient，上下使用 y gradient；僅讀四側外部窄帶及 Sobel 所需的一個像素鄰域，候選評分考慮強度、覆蓋率與距離。每側獨立判斷，沒有可靠邊緣就保留原位置。搜尋僅朝外，保留 DINO bbox 與 padding，不用 RGB 重新分割或縮掉物件。影像邊界會裁齊；小於 1 pixel 的搜尋帶不調整。

```bash
./run.sh --ros-args -p tile_size:=448 -p overlap:=0.25 -p batch_size:=4 \
  -p merge_mode:=weighted -p min_area:=1024 -p bbox_padding:=8 \
  -p rgb_border_fraction:=0.0 -p debug_visualization:=true
```

closing 可能合併相隔很近的區域；需要保留分離時可減小 `closing_kernel`。這些設定提供空間上的清理與寬鬆框，未加入跨幀追蹤或時間平滑。

舊 ROS 參數 `min_component_patches`、`padding_patches` 已由 `min_area`、`bbox_padding` 取代，啟動命令請同步更新。`closing_kernel` 現在以原圖 pixel 為單位，舊 patch 單位設定不可直接視為相同效果。

開啟 debug 後，網頁的 **Tiling diagnostics** 可切換原圖＋tile grid、逐 tile similarity、合併後完整 heatmap、threshold 後經 closing/填洞的 binary mask、最終 bbox。亦可直接讀取 `/debug/grid.jpg`、`/debug/tiles.jpg`、`/debug/heatmap.jpg`、`/debug/mask.jpg`、`/debug/boxes.jpg`。heatmap 沿用固定 0–1 色階，負分數顯示為最低顏色；原始 ROS similarity topic 保留負值。debug 預設關閉以減少 JPEG 與 CPU 成本。

`GET /api/detections` 回傳 `{"detections": [{"bbox": [x1, y1, x2, y2], "score": 0.85, "area": 1500}]}`。bbox 使用原圖半開座標（右／下界不包含），padding 與可選 RGB 微調皆裁齊原圖。IoU 達到 `merge_iou_threshold` 的框形成同一組，輸出其外接框；area 是組內清理後元件面積總和，不含 bbox padding，score 是組內最高的元件平均分數。畫面中最高分框以紅線標示，其餘框為綠線。ROS Detection2DArray 保持原介面與影像 header。

驗證（在工作區根目錄，使用既有 venv 與 ROS 環境）：

```bash
source /opt/ros/jazzy/setup.bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="dinov3_ws/src/dinov3_cases:$PYTHONPATH" \
  dinov3_venv/bin/python -m pytest -q -p no:cacheprovider dinov3_ws/tests/test_boxes.py dinov3_ws/tests/test_tiling.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="dinov3_ws/src/dinov3_cases:$PYTHONPATH" \
  dinov3_venv/bin/python dinov3_ws/tests/check_offline.py
```

單元測試使用 CPU tensor 與可記錄 batch 呼叫的模型；離線整合測試使用小型隨機權重 DINOv3，驗證 ROS／網頁流程，並非正式權重準確度或 Jetson GPU 效能量測。
