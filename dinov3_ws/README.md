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

此腳本以 640×480、30 FPS 啟動 RGB 與深度，開啟 `depth_registration:=true` 並對齊到 COLOR。保留此終端，在另一個終端啟動 DINOv3（bbox 僅需要 RGB）：

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

- `/medicine_recognition/detections`：DINO mask 的所有合格連通元件，case 名稱與原始 threshold 通過 patch 的平均 cosine similarity；無物件時發佈空陣列。ID 僅為單幀元件編號，不是追蹤 ID。
- `/medicine_recognition/debug_image`：全圖熱圖與最終 bbox。
- `/dinov3/heatmap/overlay`: `sensor_msgs/Image`, BGR8; add an Image display in RViz2 and select this topic.
- `/dinov3/heatmap/similarity`: `sensor_msgs/Image`, 32FC1, raw cosine similarity.
- Camera timestamps and frame IDs are preserved.

## Data and parameters

Reference groups persist in `/workspaces/isaac_ros-dev/dinov3_cases_data/<case>/`. You may also create folders and add PNG/JPEG/WebP/BMP files there, then select the case in the browser. Uploaded images are converted to PNG; uploads append to a case. No deletion endpoint is provided. DICOM and volumetric scans are not supported.

Parameters: `cases_dir`, `image_topic`, `model_id`, `image_size` (448; multiple of 16, 224–1024), `inference_hz` (5.0), `max_reference_patches` (8192), `host` (127.0.0.1), `port` (8090).

Aspect ratio is preserved with letterboxing, and padding patches are excluded from references. Heatmaps are mapped back to original image coordinates. The reference feature bank is bounded and rebuilt when switching to changed case data. Only the latest camera frame is processed; actual rate depends on inference time. The model and selected bank stay in GPU memory. Reference files stay local; Hugging Face is contacted to download model files.

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

## DINO mask → bounding boxes

原始 feature grid → cosine threshold → binary mask → closing → hole filling → `connectedComponentsWithStats(connectivity=8)` → 面積過濾 → bbox → padding → 原圖座標 → 可選 RGB 外圍微調。

連通元件在低解析度 patch mask 上計算，排除 letterbox 無效 patch；只有 bbox 映射回原圖。放大的 similarity map 只用於熱圖。無額外 detector、TokenCut、graph cut、SAM 或神經網路。

| ROS 啟動參數 | 預設 | 說明 |
| --- | --- | --- |
| `closing_kernel` | 3 | patch grid 的奇數 kernel；1 停用 closing，仍填洞 |
| `min_component_patches` | 4 | 清理後元件最小面積，單位為 patch 數 |
| `padding_patches` | 0.5 | 每側向外擴張的 patch 數，依 letterbox 比例映射回原圖 |
| `rgb_border_fraction` | 0.1 | bbox 外側搜尋帶相對寬／高；0 完全停用，範圍 0–0.25 |
| `edge_min_gradient` | 20.0 | 8-bit RGB 各通道 Sobel 絕對值最大值，scale=1/4 |
| `edge_min_support` | 0.6 | 邊上至少此比例超過 gradient 閾值，另須連續涵蓋至少 40% |

左右使用 x gradient，上下使用 y gradient；僅讀四側外部窄帶及 Sobel 所需的一個像素鄰域，候選評分考慮強度、覆蓋率與距離。每側獨立判斷，沒有可靠邊緣就保留原位置。搜尋僅朝外，保留 DINO bbox 與 padding，不用 RGB 重新分割或縮掉物件。影像邊界會裁齊；小於 1 pixel 的搜尋帶不調整。

```bash
./run.sh --ros-args -p padding_patches:=0.5 -p rgb_border_fraction:=0.0
```

closing 可能合併相隔很近的區域；需要保留分離時可減小 `closing_kernel`。這些設定提供空間上的清理與寬鬆框，未加入跨幀追蹤或時間平滑。

驗證（在工作區根目錄，使用既有 venv 與 ROS 環境）：

```bash
PYTHONPATH=dinov3_ws/src/dinov3_cases python -m pytest -q dinov3_ws/tests/test_boxes.py
PYTHONPATH=dinov3_ws/src/dinov3_cases python dinov3_ws/tests/check_offline.py
```
