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

此腳本以 640×480、30 FPS 啟動 RGB 與深度，開啟 `depth_registration:=true` 並對齊到 COLOR。保留此終端，在另一個終端啟動 RGB-D proposal 與 DINOv3：

```bash
docker exec -it my_isaac_ros_dev /workspaces/isaac_ros-dev/dinov3_ws/run.sh
```

`run.sh` 現在同時啟動 `rgbd_object_proposal_node` 與開啟 `use_proposals:=true` 的 DINOv3。Ctrl+C 會關閉兩個節點；任一程序退出也會關閉另一個。既有 `--ros-args -p ...` 仍傳給 DINOv3，例如 `--ros-args -p port:=8091`。兩個節點明確使用既有 venv 的 Python。

proposal 參數預設讀取 `src/rgbd_object_proposals/config/proposals.yaml`；請以公尺設定 `background_depth`（預設 1.1），並依需求調整 `foreground_threshold` 與連通域面積。自訂 YAML 可這樣啟動：

```bash
docker exec -it -e RGBD_PARAMS_FILE=/workspaces/isaac_ros-dev/dinov3_ws/src/rgbd_object_proposals/config/proposals.yaml \
  my_isaac_ros_dev /workspaces/isaac_ros-dev/dinov3_ws/run.sh
```

`RGBD_PARAMS_FILE` 使用容器內的路徑。首次使用須先建置 `rgbd_object_proposals` 及 `dinov3_cases`；完整建置、調參與視覺化步驟見 [RGB-D 文件](src/rgbd_object_proposals/README.md)。相機腳本仍接受 launch 參數覆寫，例如 `color_fps:=15 depth_fps:=15`。

Open http://localhost:8090 on the Jetson. For remote use, forward port 8090 over SSH. The service binds to loopback by default. `-p host:=0.0.0.0` exposes the unauthenticated interface and should only be used on a trusted network.

1. Enter a case name and select multiple pictures, then click **Add pictures to case**. Repeat for additional cases.
2. Select a case and click **Use selected case**.
3. Use the default ROS camera, or upload a target picture.
4. 候選模式下，相機預覽來自 proposal 轉發的 RGB；必須 RGB、depth 和 CameraInfo 均符合輸入條件才會更新。The browser shows an independent live RGB preview (up to 15 FPS) and the latest heatmap result. In proposal mode, the live preview draws every detected bounding box with its candidate ID and pixel bounds. The preview works before selecting a case and while model loading/inference is in progress. Camera disconnection and heatmap result age are shown. The browser displays the overlay. Red means stronger feature similarity on a fixed cosine scale, not detection confidence.

Reference pictures should be tightly cropped around the desired object. All reference patches, including backgrounds, can produce matches. This initial implementation matches the best reference patch from the selected group at each target patch; it does not train a classifier or produce verified object boundaries. Visually similar packaging can match even when text or medicine differs.

預設相機為 `/camera/color/image_raw`。其他 RGB-D 相機請在 proposal YAML 中一起修改 RGB、已對齊 depth 與 CameraInfo topic。若僅使用 D457 的 RGB 與舊版全圖熱圖模式：

```bash
docker exec -it my_isaac_ros_dev /workspaces/isaac_ros-dev/dinov3_ws/run.sh --ros-args -p use_proposals:=false -p image_topic:=/d457/d457/color/image_raw
```

上述 `use_proposals:=false` 只切換辨識端；`run.sh` 仍會啟動 proposal。

## ROS visualization

- `/rgbd_proposals/debug_image`：全部候選框與 ID。
- `/medicine_recognition/detections`：通過閾值的候選、case 名稱與相似度。
- `/medicine_recognition/debug_image`：裁切推論的热圖與辨識框線。

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
