# DINOv3 case heatmaps in my_isaac_ros_dev

ROS 2 Jazzy node with selectable DINOv3 ViT-H+/16 and ViT-B/16 TensorRT backbones on NVIDIA Thor.

預設使用 **TensorRT 動態 BF16 engine**，參考圖依設定限制長邊；目標圖維持原解析度、batch 1，預設推論前四周 reflect 補 64 pixels。
engine 支援高、寬各 32–2560 像素（不需為 16 倍數），範圍外會明確報錯。
`/api/status` 的 `model_id`、`inference_backend` 與 `backend_description` 顯示目前模型及後端。
詳細建置、驗證及產物位置見 [TENSORRT.md](TENSORRT.md)。

切回 PyTorch：`./run.sh --ros-args -p inference_backend:=torch`。
使用其他相容 engine：`./run.sh --ros-args -p engine_path:=/absolute/path/model.engine`。
缺少 engine 或 CUDA 時不會默默切換後端；重啟後選擇 case 即以所選後端重建參考庫。

## First: Hugging Face access

Accept/request access to `facebook/dinov3-vith16plus-pretrain-lvd1689m` and
`facebook/dinov3-vitb16-pretrain-lvd1689m` using your own Hugging Face account. Then log in interactively inside the container (do not put your token in source code or chat):

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

預設啟動 ViT-H+；使用 `--model` 在啟動時選擇權重與相符的 TensorRT engine：

```bash
# ViT-H+/16，1280 維特徵（預設）
./run.sh --model vith16plus

# ViT-B/16，768 維特徵
./run.sh --model vitb16
```

切換模型會在新程序中重建 reference bank，兩種模型的 reference features 不會混用。
`run.sh` 訂閱 `/camera/color/image_raw`，不再啟動或依賴 RGB-D proposal。既有 `--ros-args -p ...` 仍會傳給 DINOv3，例如 `./run.sh --model vitb16 --ros-args -p port:=8091`。

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

Reference groups persist in `/workspaces/isaac_ros-dev/dinov3_cases_data/<case>/`. You may also create folders and add PNG/JPEG/WebP/BMP files there, then select the case in the browser. Uploaded images are converted to PNG; uploads append to a case. 圖片可在 `/images` 頁面確認後刪除。 DICOM and volumetric scans are not supported.

Parameters: `cases_dir`, `image_topic`, `model_id`, `inference_backend` (`trt` / `torch`, default `trt`), `engine_path`, `inference_hz` (20.0), `max_reference_patches` (8192), `reference_max_side` (config default 640; 0 disables resizing), `host` (127.0.0.1), `port` (8090).

Target images retain their original resolution. `reference_max_side` limits the reference long edge while preserving aspect ratio without enlarging small images; set it to 0 to retain reference resolution. Each model forward takes one image, normalized with the model processor's rescale/mean/std settings. No letterboxing or tiling is applied. Targets receive configurable context padding before inference; their features are cropped back to the original patch grid before matching. The reference feature bank is bounded and rebuilt when switching to changed case data. Only the latest camera frame is processed. Reference files stay local; Hugging Face is contacted to download model files.

### Target padding 設定

為降低影像外框的位置相關高分，預設 `target_padding: 64`、`target_padding_mode: reflect`。相機與上傳 target 在單次 forward 前四周補邊，reference bank 維持既有流程。輸出先裁回原圖的完整 patch grid，再匹配、插值、產生 bbox 與姿態，因此輸出影像尺寸與原圖座標一致；不足一個 patch 的右／下條帶仍為 -1。

`target_padding` 必須為模型 feature stride 的倍數（目前 ViT 為 16），設為 `0` 可關閉；模式可用 `reflect`（鏡射）或 `replicate`（複製邊緣）。兩者都是啟動參數，修改後需重啟；`/api/status` 顯示設定。

```bash
# 預設：每側補 64 pixels
./run.sh
# 比較每側補 128 pixels
./run.sh --ros-args -p target_padding:=128
# 比較複製邊緣
./run.sh --ros-args -p target_padding_mode:=replicate
# 關閉以比較原始結果
./run.sh --ros-args -p target_padding:=0
```

TRT 的 2560×2560 上限包含補邊：每側 64 時，原圖每邊上限為 2432。超出 profile 會明確報錯，可減少 padding、改用 PyTorch 或重建 engine。補邊增加 token 數與計算量；控制圖實驗已確認位置效應下降，實際物體匹配效果仍需以原始大圖比較。此功能不改動 RoPE 或 cosine 的固定色階。

### Reference 尺寸設定

`reference_max_side` 在 `config/dinov3_cases.yaml` 設定參考圖長邊上限（pixels），目前設定為 `640`；設為 `0` 可保留原始解析度。程式啟動時會自動載入此 config。縮放使用 LANCZOS、維持長寬比且不放大小圖；原始檔案不會被修改，也不影響相機影像、目標圖或其座標。修改 config 後需重啟，reference bank 會重新建立。

```yaml
dinov3_cases:
  ros__parameters:
    reference_max_side: 640
```

如需臨時覆寫，仍可使用 `./run.sh --ros-args -p reference_max_side:=480`；命令列值優先於 config。

例如在 `reference_max_side: 640` 設定下，800×600 → 640×480、600×900 → 427×640；300×200 維持原尺寸。縮放後任一邊若小於 ViT patch size 16，會回報錯誤；請提高上限或調整過於狹長的參考圖。`max_reference_patches` 仍限制縮放後的參考特徵總量。

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

## 原始解析度、batch 1 patch inference → bounding boxes

原始 RGB → target context padding → rescale / normalization → DINOv3 單張全圖 forward → 裁回原圖 patch grid → reference cosine similarity → patch 分數映射回原圖 pixel → threshold → closing → hole filling → connected components → 面積過濾 → bbox → 可選 RGB 外圍微調 → bbox IoU 合併。

目標圖不縮放；參考圖依 `reference_max_side` 等比例縮小。目標圖預設四周 reflect 補 64 pixels，forward 後先裁回原圖 patch grid，再計算 similarity；reference 不補邊。兩者不做 letterbox、不拆 tiles。固定 batch size 為 1，ViT-H+ 與 ViT-B 都以 16×16 pixel patch 產生空間 tokens；CLS 與 4 個 register tokens 不參與比對。ViT-H+ 每個特徵為 1280 維，ViT-B 為 768 維。參考圖保留全部空間特徵，之後仍由 `max_reference_patches` 限制 bank 大小。

兩個模型的 feature grid 都是 `H // 16 × W // 16`。例如 1920×1080 會產生 67×120=8040 個空間特徵；最下方 8 pixel 不形成 patch。熱圖只將分數插值回完整 patch grid 覆蓋的區域，其餘右／下側像素記為 -1（未覆蓋），不拉伸分數去填補。寬或高小於 16 pixel 時回報錯誤。

整張圖在同一次 forward 中參與 attention。全圖解析度越高，GPU 記憶體與推論時間可能大幅增加。CUDA 支援 BF16 時使用 BF16，其他情況使用 FP32；similarity 使用 FP32，檢查 features 與輸出是否含 NaN/Inf。網頁顯示 patch grid、batch 1 與實際精度。

已移除 `image_size`、`tile_size`、`overlap`、`merge_mode`、`batch_size` ROS 參數，請從啟動命令刪除。`bbox_padding` 仍是推論後的框擴張量，與輸入影像補邊無關。

| ROS 啟動參數 | 預設 | 說明 |
| --- | --- | --- |
| `closing_kernel` | 3 | **原圖 pixel** 的奇數 kernel；1 停用 closing，仍填洞 |
| `min_area` | 2048 | 清理後連通元件的最小面積，單位 pixel² |
| `bbox_padding` | 8 | bbox 每側向外擴張的原圖 pixel 數 |
| `merge_iou_threshold` | 0.3 | 最終 bbox 的 IoU 達到此值時合併；取外接框、最高分與元件面積總和 |
| `debug_visualization` | false | 開啟網頁五種 patch 除錯視圖 |
| `rgb_border_fraction` | 0.1 | bbox 外側搜尋帶相對寬／高；0 完全停用，範圍 0–0.25 |
| `edge_min_gradient` | 20.0 | 8-bit RGB 各通道 Sobel 絕對值最大值，scale=1/4 |
| `edge_min_support` | 0.6 | 邊上至少此比例超過 gradient 閾值，另須連續涵蓋至少 40% |

左右使用 x gradient，上下使用 y gradient；僅讀四側外部窄帶及 Sobel 所需的一個像素鄰域，候選評分考慮強度、覆蓋率與距離。每側獨立判斷，沒有可靠邊緣就保留原位置。搜尋僅朝外，保留 DINO bbox 與 padding，不用 RGB 重新分割或縮掉物件。影像邊界會裁齊；小於 1 pixel 的搜尋帶不調整。

```bash
./run.sh --ros-args -p min_area:=1024 -p bbox_padding:=8 \
  -p rgb_border_fraction:=0.0 -p debug_visualization:=true
```

closing 可能合併相隔很近的區域；需要保留分離時可減小 `closing_kernel`。這些設定提供空間上的清理與寬鬆框，未加入跨幀追蹤或時間平滑。

舊 ROS 參數 `min_component_patches`、`padding_patches` 已由 `min_area`、`bbox_padding` 取代，啟動命令請同步更新。`closing_kernel` 現在以原圖 pixel 為單位，舊 patch 單位設定不可直接視為相同效果。

開啟 debug 後，網頁的 **Patch diagnostics** 可切換原圖＋patch grid、patch similarity、完整 heatmap、threshold 後經 closing/填洞的 binary mask、最終 bbox。亦可直接讀取 `/debug/grid.jpg`、`/debug/patches.jpg`、`/debug/heatmap.jpg`、`/debug/mask.jpg`、`/debug/boxes.jpg`。heatmap 沿用固定 0–1 色階，負分數顯示為最低顏色；原始 ROS similarity topic 保留負值。debug 預設關閉以減少 JPEG 與 CPU 成本。

`GET /api/detections` 回傳 `{"detections": [{"bbox": [x1, y1, x2, y2], "score": 0.85, "area": 1500}]}`。bbox 使用原圖半開座標（右／下界不包含），padding 與可選 RGB 微調皆裁齊原圖。IoU 達到 `merge_iou_threshold` 的框形成同一組，輸出其外接框；area 是組內清理後元件面積總和，不含 bbox padding，score 是組內最高的元件平均分數。畫面中最高分框以紅線標示，其餘框為綠線。ROS Detection2DArray 保持原介面與影像 header。

驗證（在工作區根目錄，使用既有 venv 與 ROS 環境）：

```bash
source /opt/ros/jazzy/setup.bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="dinov3_ws/src/dinov3_cases:$PYTHONPATH" \
  dinov3_venv/bin/python -m pytest -q -p no:cacheprovider dinov3_ws/tests/test_boxes.py dinov3_ws/tests/test_patches.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="dinov3_ws/src/dinov3_cases:$PYTHONPATH" \
  dinov3_venv/bin/python dinov3_ws/tests/check_offline.py
```

單元測試使用 CPU tensor 與可記錄 batch 呼叫的模型；離線整合測試使用小型隨機權重 DINOv3，驗證 ROS／網頁流程，並非正式權重準確度或 Jetson GPU 效能量測。

## 本機圖片管理與 HSV 去背

既有 DINOv3 服務啟動後，開啟 <http://localhost:8090/images>，或從首頁點選「圖片管理／HSV 去背／標籤分類」。同樣使用 `cases_dir` 下的 case 資料夾作為標籤。

如只需要整理圖片與去背，可獨立啟動，不需 ROS、相機或載入 DINOv3 模型。獨立服務預設使用 8091，避免佔用原本 DINOv3 的 8090；開啟 <http://localhost:8091/images>：

```bash
cd /workspaces/isaac_ros-dev/dinov3_ws
./run_image_manager.sh
# 如需指定另一個連接埠：
./run_image_manager.sh --port 8092
# 自訂圖片資料夾：
./run_image_manager.sh --cases-dir /path/to/cases --port 8091
```

1. 上傳圖片到標籤，或選擇既有標籤中的圖片。標籤限 1–80 個英文字母、數字、底線、連字號。
2. 在原圖上點選背景色。調整 H 色相容差（角度）、S 飽和度容差與 V 亮度容差（0–255）；頁面自動更新透明預覽。
3. 預設只去除與取色位置連通的相近顏色，避免刪除物體內部斷開的同色區域；取消勾選則套用於全圖。可設定 0–10 px 邊緣柔化。
4. 指定新圖片的標籤並儲存透明 PNG，或下載預覽。儲存會新增檔案並保留原圖；可再選取新圖，重複去除其他背景區域。
5. 「刪除此圖片」會先要求確認，確認後永久刪除該檔案。空的標籤資料夾保留。

圖片與處理結果均留在本機；單次上傳上限 64 MB，單張上限 2500 萬像素。影像的 EXIF 方向會正規化，既有透明度會保留。色相差採環狀計算，灰白背景主要由 S/V 判斷。HSV 適合顏色較均勻的背景，物體與背景同色時仍可能誤刪，請先檢查預覽。

DINOv3 服務會在目前標籤的圖片新增或刪除後重新建立參考特徵。DINOv3 輸入仍是 RGB：透明區域並非特徵遮罩，本工具將完全透明區域的隱藏 RGB 清為黑色，避免原背景在 RGB 轉換時重新出現；黑色背景仍可能產生匹配。原圖保留在原標籤時也仍會參與匹配，可將去背結果存到另一個標籤並選用該標籤。
