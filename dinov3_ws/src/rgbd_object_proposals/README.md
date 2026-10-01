# RGB-D 幾何候選框與 DINOv3 串接

此套件使用 **Background Depth Subtraction → Depth-continuous Components → bbox**，適用背景可近似為固定深度平面的環境。物體可依前景遮罩或相鄰深度跳變分離；同深度接觸、或以連續深度路徑相連的物體仍可能合併為一個框。使用 NumPy、OpenCV，無需 Open3D。節點名稱及 executable 均為 `rgbd_object_proposal_node`，套件名稱為 `rgbd_object_proposals`。

## 檔案結構

```text
rgbd_object_proposals/
├── package.xml / setup.py / setup.cfg
├── resource/rgbd_object_proposals
├── rgbd_object_proposals/
│   ├── __init__.py
│   ├── geometry.py        # 不依賴 ROS 的幾何演算法
│   ├── node.py            # 同步、驗證、訊息發布
│   └── visualization.py   # 候選框除錯繪圖
├── config/proposals.yaml
├── launch/proposals.launch.py
├── examples/synthetic_scene.py
├── examples/proposals_demo.png
└── test/                  # 幾何及 ROS 訊息介面測試
```

既有 `dinov3_cases` 增加 `proposals.py`，並於 `node.py` 提供 `use_proposals` 選項；預設 `false`，原有整張影像熱圖流程照常運作。

## 必要輸入條件

- 在 `background_depth` 設定背景平面的 Z 深度，單位為公尺；預設 1.1 m。節點不採集背景，第一張有效影格即可產生候選。
- Depth 必須由相機驅動**對齊彩色影像**，使用相同像素網格、解析度與 color optical frame。相同解析度本身不能證明已對齊；請確實開啟驅動的 registration。
- 使用對應彩色影像的 `CameraInfo`。光學座標為 x 向右、y 向下、z 向前，深度為 Z，而非射線距離。此版本不做 TF／外參對齊。
- 支援 `16UC1`，乘 `depth_scale`（預設 0.001，即 mm → m）；`32FC1` 固定視為公尺，不再乘比例。忽略 0、負值、NaN、Inf 及深度範圍外像素。
- 支援原始彩色影像 `plumb_bob`／`rational_polynomial` 畸變：先以 OpenCV 校正射線，再投影回原始彩色像素；零畸變使用針孔公式。魚眼模型、含 ROI 偏移或 binning 的 CameraInfo 會被拒絕。若用 rectified 影像，RGB 與 depth 都必須對齊該網格，並提供相符 K 及零畸變 D；不要混用 raw 的內參。
- RGB/depth 用 `ApproximateTimeSynchronizer`，預設允許 50 ms 時差；移動場景建議更小。CameraInfo 快取最近一筆，避免靜態／低頻 calibration 阻塞影像；`camera_info_max_age: 0.0` 不限制年齡，正值則檢查時間差。
- 輸入使用 sensor-data best-effort QoS，可接收 best-effort 或 reliable 的相機發布端。

## 訊息流與 topic

```text
RGB + aligned depth + color CameraInfo
                 ↓
       rgbd_object_proposal_node
                 ├── /rgbd_proposals/rgb ────────────┐
                 ├── /rgbd_proposals/detections ─────┤ exact stamp
                 ├── /rgbd_proposals/debug_image     ↓
                 └── /rgbd_proposals/points      dinov3_cases
                                                    ├── /medicine_recognition/detections
                                                    └── /medicine_recognition/debug_image
```

| Topic | 型別 | 說明 |
|---|---|---|
| `/camera/color/image_raw` | `sensor_msgs/msg/Image` | 預設 RGB 輸入 |
| `/camera/depth/image_raw` | `sensor_msgs/msg/Image` | 預設 depth 輸入，必須已開啟對齊 |
| `/camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` | 彩色內參 |
| `/rgbd_proposals/rgb` | `sensor_msgs/msg/Image` | 通過驗證的原始 RGB 訊息，保留 header／encoding |
| `/rgbd_proposals/detections` | `vision_msgs/msg/Detection2DArray` | 與 RGB 完全相同 header；沒有候選仍發布空陣列 |
| `/rgbd_proposals/debug_image` | `sensor_msgs/msg/Image` | BGR 框線與候選 ID |
| `/rgbd_proposals/points` | `sensor_msgs/msg/PointCloud2` | 可選，前景遮罩內取樣的 XYZ 公尺點雲；包含未通過連通域面積或尺寸篩選的點 |
| `/medicine_recognition/detections` | `vision_msgs/msg/Detection2DArray` | 達到閾值且相似度最高的單一候選、case 名稱及相似度 |
| `/medicine_recognition/debug_image` | `sensor_msgs/msg/Image` | 候選裁切熱圖及接受的框線 |

proposal 輸出 RGB 與 detections 使用 reliable、volatile、keep-last 10；除錯影像及點雲 depth 1。下游也使用 reliable depth 10 並精確同步；推論只處理最新可用的一組，過載時可能略過影格。

`Detection2D.bbox.center.position.{x,y}` 是框中心，`size_x/size_y` 是像素寬高，`theta=0`。採 `[xmin,xmax)`、`[ymin,ymax)` 半開區間，可直接 `rgb[ymin:ymax, xmin:xmax]`。框線已含 padding 並限制於影像內。`id` 依左到右／上到下排序為 `candidate_0` 等；**僅當前影格有效，不是跨影格追蹤 ID**。proposal 的 `results` 留空，不捏造類別或信心值；AABB、OBB、重心、特徵值保留在演算法的 `Candidate`，不透過 Detection2D 傳 3D 資料。

不為各候選建立獨立 crop topic。既有辨識端裁切 RGB，沿用 patch embedding 與目前選取 case 的 reference bank。每個 crop 的分數是各像素最佳 reference-patch cosine 相似度的平均值；每張影格只輸出分數最高且達到既有 `cosine_threshold` 的單一候選，並在除錯熱圖上以紅框標示。輸出包含 `results[0].hypothesis.class_id` 與 `score`。這不是機率，也不是全圖熱圖最大值；閾值需用實拍資料調整。目前比較使用者選取的一組 case，而非一次分類整個資料庫。上傳靜態圖片仍走原有全圖熱圖，不產生幾何辨識訊息。

介面對照：[Detection2D 官方定義](https://github.com/ros-perception/vision_msgs/blob/ros2/vision_msgs/msg/Detection2D.msg)、[BoundingBox2D 官方定義](https://github.com/ros-perception/vision_msgs/blob/ros2/vision_msgs/msg/BoundingBox2D.msg)、[message_filters 原始碼](https://github.com/ros2/message_filters/blob/jazzy/src/message_filters/__init__.py)。目標為 Humble／Jazzy 的 `vision_msgs/Pose2D` 介面；本環境尚未進行 ROS runtime 驗證。

## 演算法與調參

1. 深度轉為公尺，忽略 0、負值、NaN、Inf 與 `min_depth/max_depth` 範圍外的值。
2. `background_depth` 是整張影像共用的背景 Z 深度，預設 1.1 m。使用 `background_depth - current_depth > foreground_threshold` 提取比背景更靠近相機的像素。預設門檻為 0.015 m；較遠的表面、移除物體留下的空洞和無效深度不當成新物體。靜止物體會持續保留。
3. 將遮罩限制在 `roi: [xmin,ymin,xmax,ymax]`；全零代表全圖。OpenCV 在完整解析度遮罩上做 `connectivity: 8`（可選 4）的連通域分析。`depth_jump_threshold > 0` 時，改用浮動範圍 flood fill，僅連接深度差不超過此門檻的相鄰前景像素；分割之後才做面積篩選。`min_component_area/max_component_area` 為原始影像的像素面積，預設 40–30000，用來濾除小雜點與過大區域。未做形態學填洞或連接，因此深度缺失可能分裂物體，沒有足夠深度落差的接觸前景仍會合併。
4. 直接使用各連通域的像素外接矩形，加上 `bbox_padding_ratio`（Config 預設 0.05，工作區 YAML 現值為 0.000001），並裁切至影像範圍；右、下界為 exclusive。`pixel_stride` 僅影響除錯點雲，不影響遮罩、面積與框邊界。
5. 保留各連通域的重心、AABB、PCA OBB 等 3D 資料。`filter_physical_size: false` 預設不以 3D 尺寸篩選，以便平面正面或薄物體也能輸出 bbox。若啟用，`width/height/depth_size` 為由大到小排序的觀測 OBB 邊長，非真實包裝尺寸；單視角通常無法量到完整厚度。

### 深度連續性分割

工作區 YAML 啟用 `depth_jump_threshold: 0.02`（2 cm），`bbox_padding_ratio` 目前為 `0.000001`，因向外取整仍可能增加一像素；若要完全取消外擴，請設為 `0.0`。這是待實拍調整的起始值；程式 Config 的門檻預設仍為 `0.0`，保留未載入 YAML 時的舊行為。ROS 節點會自動宣告 Config 中的參數，修改 YAML 後重新啟動 `run.sh`。

對每對 4／8 鄰接像素 p、q，只有兩者都在前景遮罩內，且 `abs(Z[p] - Z[q]) <= depth_jump_threshold` 才能連接。實作用 [OpenCV floodFill](https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html) 的 floating range，比較鄰居而非起始像素；因此相鄰只差 1 cm、整體落差 30 cm 的斜面仍可保持完整。使用原始公尺深度，不使用 heatmap 顏色；無效深度、ROI 外和背景皆禁止穿越，不刪除物體邊界像素。

- 框仍跨過明顯深度邊緣時，可試著調低門檻；同一物體碎成多框時，可調高門檻。門檻需高於實際相鄰深度雜訊，低於希望切開的邊緣落差。
- `depth_jump_threshold: 0.0` 恢復原本二值連通域。`foreground_threshold` 決定哪些像素進入前景，`depth_jump_threshold` 決定前景內如何分組，兩者不同。
- 目前 YAML 的 `background_depth: 1.1` 與 `foreground_threshold: 0.5` 仍保留，代表只接受 Z < 0.6 m 的有效像素；連續性分割無法找回已被這層排除的物體。
- 回傳的 `foreground_mask` 仍是所有前景的聯集；分割標籤在幾何流程內使用。DINOv3 仍裁切矩形並對整張 crop 評分，此改動不等於 mask 加權辨識。
- 同深度接觸或緩慢深度橋接仍會合併；大幅彎折的單一物體也可能被切開。未新增形態學閉合，以免重新接回已分離的區域。

### 傾斜桌面的後續接法

深度連續性只負責分组，不辨識哪一塊是桌面。若桌面被固定 Z 門檻選入前景，應先替換 `foreground_mask()` 的背景模型，再接上述分割：

1. 固定相機：移除物體後採集多幀對齊深度，對每個像素的有效值取中位數，得到 `B(u,v)`；以 `B(u,v) - Z(u,v) > foreground_threshold` 建立前景。背景缺值必須標為未知，相機移動後需要重新採集。
2. 平面模型：以有效背景 3D 點估計桌面 `n·P + d = 0`，將 n 正規化並朝向相機；以前景點高於桌面的有號距離 `n·P + d > height_threshold` 篩選。需要平面品質檢查，避免把大型物體表面誤認為桌面。

以上背景圖／平面模型是擴充接點說明，尚未實作；現有節點仍使用固定背景 Z。

所有設定列於 `config/proposals.yaml`，修改後需重啟。先量測並設定背景 Z 深度，再調 ROI、`foreground_threshold`、連通域面積與 padding。深度門檻過大會漏掉薄物體，過小會引入深度雜訊。舊的平面、體素、離群點和 DBSCAN 參數已移除，自訂 YAML 應改用新參數。

`background_depth` 可在 YAML 或啟動時覆寫，例如 `--ros-args -p background_depth:=1.25`。相機或背景位置改變後，重新量測並修改此值；回傳為 `(candidates, foreground_points, foreground_mask)`。

## 建置與啟動

已整合到工作區既有入口：在 ROS 容器內分別執行 `dinov3_ws/run_camera.sh`（RGB＋對齊深度）及 `dinov3_ws/run.sh`（proposal＋DINOv3）。`run.sh` 支援原有辨識參數 `--ros-args -p ...`，proposal 的自訂 YAML 以 `RGBD_PARAMS_FILE` 環境變數指定。Ctrl+C 一起關閉 proposal 與辨識節點。以下分開啟動的命令供獨立除錯，勿與 `run.sh` 重複啟動。

以下命令在**已安裝 ROS 2 的 shell／容器**執行；依實際環境調整 workspace 與 ROS 發行版。DINOv3 沿用原來可載入 Torch、Transformers、Flask、Pillow 的 Python 環境，不在本次重新下載模型或修改環境。

```bash
source /opt/ros/jazzy/setup.bash   # Humble 環境改為 humble
cd /home/ubuntu/workspaces/isaac_ros-dev/dinov3_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select rgbd_object_proposals dinov3_cases
source install/setup.bash
```

相機終端，先 source 已建置的 Orbbec workspace，再開啟對齊：

```bash
source /home/ubuntu/workspaces/isaac_ros-dev/orbbec_sdk/install/setup.bash
ros2 launch orbbec_camera gemini_330_series.launch.py \
  depth_registration:=true align_mode:=SW align_target_stream:=COLOR
```

本工作區驅動的 `depth_registration` 預設 false；開啟 COLOR 對齊後 `/camera/depth/image_raw` 才符合本節點預期。實際 topic 隨 camera_name／namespace 而異，請用 `ros2 topic list` 確認，並修改 YAML 三個輸入 topic。RGB／depth 必須有相同尺寸和光學 frame_id。不要同時重複啟動同一台相機。

proposal 終端，先設定背景深度、差分門檻與其他設定：

```bash
ros2 launch rgbd_object_proposals proposals.launch.py \
  params_file:=/home/ubuntu/workspaces/isaac_ros-dev/dinov3_ws/src/rgbd_object_proposals/config/proposals.yaml
```

辨識終端，在既有 DINOv3 環境與 workspace overlay 中執行：

```bash
ros2 run dinov3_cases case_heatmap --ros-args \
  -p use_proposals:=true \
  -p proposal_image_topic:=/rgbd_proposals/rgb \
  -p proposals_topic:=/rgbd_proposals/detections \
  -p cases_dir:=/home/ubuntu/workspaces/isaac_ros-dev/dinov3_cases_data
```

開啟 `http://127.0.0.1:8090`，選取／上傳 reference case，再選 Use ROS camera。既有服務若已佔用 8090，停止舊實例或使用 `-p port:=8091`。模型與 case 未就緒時不會輸出辨識結果。

## 輸出檢查與視覺化

```bash
ros2 topic list
ros2 topic info /camera/depth/image_raw --verbose
ros2 topic echo /camera/color/camera_info --once --qos-reliability best_effort
ros2 topic echo /rgbd_proposals/detections --once
ros2 topic echo /medicine_recognition/detections --once
ros2 topic hz /rgbd_proposals/detections
rqt_graph
ros2 run rqt_image_view rqt_image_view /rgbd_proposals/debug_image
rviz2
```

RViz2 加入 **Image** display，選 `/rgbd_proposals/debug_image`。若要看點雲，將 YAML 的 `publish_pointcloud: true` 後重啟，Fixed Frame 設成輸入 RGB 的 `header.frame_id`（通常為 `camera_color_optical_frame`），加入 **PointCloud2** display、topic `/rgbd_proposals/points`。標準 RViz2 不一定有 Detection2DArray display，直接看已畫好框線的 Image 即可。

`rqt_graph` 應呈現相機 → proposal → DINOv3；空候選也持續有 detections。若完全沒有輸出，檢查節點 log、CameraInfo、encoding、frame_id、同步時間及 QoS。若沒有候選框，檢查 `background_depth` 是否大於物體深度、差值是否超過 `foreground_threshold`，以及連通域面積限制。

## 測試與範例

不需要 ROS 或模型即可跑幾何測試及射線投射的合成 RGB-D 圖：

```bash
cd /home/ubuntu/workspaces/isaac_ros-dev/dinov3_ws
OPENBLAS_NUM_THREADS=1 python3 -m pytest -q src/rgbd_object_proposals/test
PYTHONPATH=src/rgbd_object_proposals OPENBLAS_NUM_THREADS=1 \
  python3 src/rgbd_object_proposals/examples/synthetic_scene.py \
  --output /tmp/rgbd_proposals_demo.png
```

`examples/proposals_demo.png` 是**合成資料**產生的兩個盒子候選框，不是相機實測。測試涵蓋深度單位／無效值、畸變投影、兩物體、純桌面、空場景、固定背景深度、無效深度、前景方向與門檻、連通性與面積、平面物體、取樣步長獨立性、薄面尺寸、框線裁切、ROI 和錯誤參數。ROS 訊息測試在無 rclpy 時跳過，在 ROS shell 中使用實際 vision_msgs 並以假 encoder 驗證裁切、閾值、ID、空輸出與 header；不下載模型。

有 ROS 的環境可再執行：

```bash
colcon test --packages-select rgbd_object_proposals --event-handlers console_direct+
colcon test-result --verbose
```

效能隨解析度與點數而異，沒有承諾相機幀率；可限制 ROI、停用除錯輸出；增加 stride 僅減少除錯點雲取樣。NumPy BLAS 過度開執行緒時，可在啟動前設 `OPENBLAS_NUM_THREADS=1`。要改用其他 segmentation／neural proposals，只需維持 RGB 同時間戳與 Detection2DArray 介面即可替換前端。
