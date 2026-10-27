# dinov3_cases 程式導覽

目前部署入口是 `node.py:main`，使用 SAM3 找候選物件，再以 DINOv3 特徵與 Lab 色彩直方圖比對圖庫。HTTP 網頁與 ROS 訂閱共用同一個節點狀態。

## 資料流程

1. `node.py` 接收最新 RGB 影格，先配對深度，再取得 `frame_pool.py` 的共享影像租約。
2. `async_kcf.py` 決定同步偵測或背景偵測；`grounding_ipc.py` 將影像交給同一份 `.venv` 的 TensorRT 11 子程序中的 `sam3_trt.py`，使用官方 TensorRT vision 與 PyTorch bbox head。主程序使用 Cat2Real FP32 Torch-TensorRT。
3. 非同步模式每 `grounding_detection_interval` 個已處理影格提交一次背景工作：SAM3 找框後，DINOv3 在同一張偵測影像上辨識；圖庫更新也在此背景工作完成。前景 KCF 持續追蹤舊集合。背景以偵測影像初始化新 tracker，再每次跳到當下最新已處理影格補追，追上後一起交接框、分類及診斷快照；部分追蹤遺失只移除該物件，不在前景重跑 DINO。追蹤遺失或接近 ROI 邊界可提早觸發整組工作；靜態上傳圖片每秒可刷新。間隔為 1 時停用 KCF，由推論 worker 依序執行兩個模型。
4. `tracking.py` 平滑候選框；可選的 ROI 追蹤另外管理 ID、命中比例及顯示門檻。
5. `encoder.py` 將 Cat2Real 裁切圖置中補白至 384，從一次 forward 保留 pooler 描述向量與 patch 熱圖特徵；`grounded.py` 建立圖庫、分類及管理辨識快取。先以 `recognition_threshold` 篩選 DINO cosine 分數，再從候選參考圖中選 Lab Bhattacharyya 分數最高者的分類；無候選時為 unknown。`score` 為所選參考圖的 DINO 分數，不再計算加權分數。
6. `async_bbox3d.py` 獨立排程背景深度工作，透過 `depth_bbox.py` 計算中心、尺寸與旋轉；預設每 5 個處理影格量測一次。ROS 發布和疊圖各用獨立 worker，待處理輸出只保留最新一份；共享影格池滿時，先釋放尚未開始的舊疊圖工作，讓新推論繼續。
7. `object_visualization.py` 在網頁要求時才產生物件診斷圖；`visualization.py` 負責 2D／3D 疊圖。

## 修改位置

| 要修改的功能 | 主要檔案 |
| --- | --- |
| ROS 參數、排程、發布 | `node.py` |
| 分類分數、unknown 規則、辨識快取 | `grounded.py` |
| 模型輸入與推論 | `encoder.py`、`patches.py`、`trt_backend.py`、`sam3_trt.py` |
| 偵測間隔、背景回放、KCF 遺失處理 | `async_kcf.py`、`kcf.py` |
| 背景 3D 框間隔、快取、深度與旋轉 | `async_bbox3d.py`、`depth_bbox.py` |
| 圖庫路徑驗證、支援格式、檔案列舉、讀圖 | `gallery_files.py` |
| HTTP 狀態與辨識操作 | `web.py` |
| 圖庫上傳、相機裁切、HSV 去背 | `image_manager.py` |
| 診斷圖與比較參考圖 | `object_visualization.py` |
| 網頁互動與外觀 | `static/`、`templates/` |

`grounded.py` 保留原有診斷函式的匯入入口，既有整合工具仍可使用；新程式可直接從 `object_visualization.py` 匯入。

## 必須維持的生命週期

- SAM3 框屬於較舊的影格；在最新影格初始化是位置近似，偵測期間物體快速移動可能使新 tracker 追錯區域。
- 相同影格重算不能增加追蹤命中次數或重複累加平滑樣本。
- KCF 組更換或物件數改變時重新編碼；圖庫或分類門檻改變時可用既有 descriptor 重新分類。
- 診斷裁切屬於辨識當時的影格，不可套用後續 KCF 框冒充目前影像。
- 共享影格在所有租約釋放前不可覆寫；舊 generation 的背景結果不可發布。
- 診斷快照保留完整舊圖庫版本，避免更新圖庫時混用特徵與標籤。

測試命令見 `../../tests/README.md`。重構前已存在的限制：`qwen_encoder.py` 引用 `runtime.py` 中不存在的 `QWEN_*` 設定，導致 `test_qwen_encoder.py` 無法收集；目前主節點僅接入 DINOv3。Qwen 程式與既有修改予以保留，未在此次整理中刪除或補上另一套推論流程。

網頁 `result FPS` 是完整辨識主迴圈的結果頻率，相機的 `input FPS` 是輸入回呼頻率。主迴圈只保留可配對的深度快照並提交背景 3D 工作，不等待深度或量測完成；非同步模式的 DINOv3 在背景執行，`recognize_ms` 記錄交接結果的背景辨識耗時；細分計時和影像輸出方式見 `../../BUILD_GUIDE.md`。

啟動或套用設定後，第一份結果會等待同一張 RGB 影格的 SAM3、DINOv3 與 KCF 初始化完成，再恢復持續追蹤；相機擷取與網頁預覽仍獨立運作。同步深度會在等待推論前保留。這可避免首次初始化等待期間持續補追而反覆超限；模型載入完成後，網頁會顯示準備首次偵測的狀態。`/api/status` 的 `grounding_schedule` 可查看目前初始化、背景工作及最近一次補追丟棄原因。

### 深度框的旋轉估計

`depth_bbox.py` 先從偵測框中央深度選出連通點雲，再用 RANSAC 擬合主要平面。
將平面內點投影到平面座標，以最小面積矩形的長邊作為局部 X 軸，平面法向量作為 Z 軸，Y 軸以右手規則建立。
`bbox3d.rotation` 是局部座標轉至相機座標的 3×3 矩陣；法向量朝遠離相機的方向，長軸絕對值最大的相機分量取正。
局部座標的分位數上下界決定 `position` 與 `size`，包括 Z 中心；不再固定使用平均相機深度作為旋轉框中心。
3D 疊圖頂點與 ROS `Detection3D` 的 bbox／result quaternion 共用此旋轉。

RANSAC 使用經深度 quantile 篩選的連通點雲、嘗試 128 次，預設平面距離門檻 `plane_threshold=0.005` 公尺，至少半數點須支持主要平面。
`orientation_status` 為 `ok`、近正方形時的 `ambiguous_in_plane`，或 `camera_aligned_fallback:<原因>`；
平面估計失敗時保留原本相機對齊框及平均 Z，並透過 `method` 標記回退。
`plane_inlier_count` 提供平面支持點數，`depth_range` 仍是相機 Z 的分位數範圍。

此結果描述可見點雲：局部 Z 尺寸是表面起伏，不能視為完整藥盒厚度，中心也不一定是實體盒體中心。
長軸存在 180° 歧義，近正方形或遮擋時可能跳動；尚未加入跨影格姿態平滑。
中央深度帶仍會裁切深度跨度過大的斜面，且偵測框內若包含同深度背景，可能影響平面與矩形擬合。

### RANSAC 前的點雲篩選

先選中央深度帶連通區，再依 `bbox3d_quantile` 的相機 Z 分位數剔除深度極端點。
只有保留的點才交給 RANSAC；旋轉框仍以原有局部座標 quantile 計算尺寸，避免重複裁掉空間邊界。
角落與最近點診斷保留原連通區的樣本。

節點在 RANSAC 前執行點雲可靠性檢查：

- `bbox3d_min_cloud_coverage: 0.15`：quantile 後點數 / 裁切後 bbox 像素數，低於 15% 的碎片直接拒絕，重複出現也不放行。
- `bbox3d_min_coverage_ratio: 0.35`：覆蓋率低於同物件上一筆接受值的 35% 時視為可疑。使用比例避免 2D bbox 大小變更直接影響點數比較。
- `bbox3d_max_depth_jump: 0.10`：quantile 後中位深度與上一筆接受值相差超過 10 公分時視為可疑。
- `bbox3d_depth_confirmations: 3`：可疑新區塊需連續 3 個不同深度時間戳支持；新候選間深度差需在跳動門檻的一半以內，覆蓋率彼此至少一半。回復原區塊、低覆蓋或量測失敗會中斷確認。

拒絕狀態為 `insufficient_cloud_coverage` 或 `cloud_change_pending:<次數>/<需求>`。
拒絕整個區塊後不執行 RANSAC、不建立相機對齊回退框，該次量測沒有 3D 框；不以舊框冒充新量測。
接受的輸出包含 `plane_input_count`、`cloud_coverage`、`cloud_depth`，供 HTTP 診斷使用。
背景 worker 以既有幾何配對身分保存個別歷史，物件消失或來源 context 變更時清除；過期工作不會回寫新來源歷史。
底層獨立呼叫 `estimate_bbox3d` 會先裁切 RANSAC 深度 quantile；若要套用覆蓋與連續性檢查，需傳入 `CloudFilter` 與各物件獨立的 `cloud_state`。
參數在節點啟動時讀取，修改 YAML 後需重啟。此變更未更動現有的單平面擬合流程。

### 3D 框貼齊 2D 邊界

`attach_bboxes3d` 在點雲量測完成後固定原本旋轉，調整中心與局部長、寬、高，
讓八頂點投影的外接矩形貼近同次量測的 2D bbox 四邊；三軸始終互相垂直。
使用相同相機畸變模型及 2D 繪圖右／下界的 `x1-1`、`y1-1` 像素慣例。
瀏覽器縮放亦保留這個像素端點，避免放大影像時右／下邊多出數個像素。
深度與尺寸以原始中心深度正規化，加入弱量測先驗以限制尺度／深度歧義；
中心深度可在原始值的 0.25–4 倍內調整，頂點必須留在相機前方。
擬合最多 100 次函數評估，四邊最大誤差不超過原圖 0.25 像素標記 `fit_status: matched`；
幾何限制下無法精確貼合則標記 `approximate`，求解失敗或結果不如原框時保留原始量測。

HTTP 3D 資料保留 `measured_position`、`measured_size`、`fit_target_bbox`；
接受擬合時另含 `fit_projected_bbox` 與 `fit_error_px`。
最終 `position`、`size` 同時供繪圖與 ROS Detection3D 使用；尺寸是受 2D 偵測約束的長方體，
不是物體實際厚度量測。點雲、平面與取樣診斷仍描述原始量測。
獨立呼叫 `estimate_bbox3d` 仍回傳原始點雲框，無擬合。
背景快取沿用量測當時的 2D 目標框；兩次深度更新間物件移動時，可能暫時與最新 KCF 框不同。

### 背景 3D 框排程

`bbox3d_interval: 5` 與 `grounding_detection_interval` 各自計數、互不控制。
3D 在第 1、6、11…個實際處理的不同 RGB 影格提交工作；跳過的相機影格不計數，
同一影格因 SAM3 交接或 3D 完成而重跑也不重複計數。設定為 1 仍在背景執行。
背景涵蓋深度轉換、點雲分割、RANSAC、長邊方向，以及中心和尺寸計算；只有一個執行中的工作與一份最新待算工作，忙碌時覆蓋舊的待算工作。
RGB 主迴圈不等待深度到達；它保留當下可配對的深度，若推論後才有對應深度則再取一次快照。

繪圖與 ROS 發布前會讀取最新完成的一整份量測，不混搭不同次的中心、尺寸及旋轉。
`bbox3d_max_age: 1.5` 是從來源影格接收時間起算的有效秒數；首次尚未完成、量測失敗或結果過期時不畫舊框。
背景結果透過相鄰 2D 框的幾何配對維持內部物件身分；順序更換與部分遺失不會直接以陣列索引套用舊結果，
消失的身分不再使用，來源 generation、影像大小或座標框架變更時清空快取。
這仍有相近物件交叉或遮擋的配對歧義，尚未加入外觀式的持久物件識別。

`bbox3d_source` 記錄來源 RGB 影格、座標框架和時間戳，`bbox3d_age_seconds` 記錄量測年齡。
ROS 每筆 `Detection3D.header` 保留該量測的來源時間，陣列 header 使用其中最新的量測時間；
2D 辨識與疊圖影像仍使用各自目前影格的時間。
`/api/status` 的 `performance.bbox3d_schedule` 顯示獨立間隔與背景量測耗時；`bbox3d_ms` 則是前景提交／讀取快取的耗時。
參數在啟動時讀入，修改 YAML 後需重啟節點。

### 共用 H.264 即時影像

主頁與圖庫頁共用一條 960×540 H.264 影像：原圖縮放、相機內參縮放、畫框各一次，
單一編碼器的 Annex-B 封包同時提供給 WebRTC 與 ROS，兩路不解碼後重新編碼。
主頁／圖庫頁都使用 `<video>`；圖庫的手動相機擷取仍從原始 RGB 取圖，與觀看串流無關。
當來源設為上傳圖片時，兩頁觀看的都是該上傳結果。

- 網頁：`POST /api/video/offer` 交換 SDP，使用 WebRTC H.264 播放；關閉時 `DELETE /api/video/peers/<peer_id>`。
- ROS：`/medicine_recognition/image_annotated/h264`，型別 `sensor_msgs/msg/CompressedImage`，`format: h264`。
  每則 `data` 是一個完整 Annex-B access unit，保留來源 RGB 的 `header`。
  接收端須使用有狀態的 H.264 解碼器，從含 SPS/PPS 的 IDR 開始；不能將每張當成獨立 JPEG。
  這是本專案的 H.264 封包格式，不相容一般 JPEG/PNG `compressed_image_transport` 或 `rqt_image_view`。
  ROS QoS 為 reliable、volatile、keep-last 30；只需觀看的客戶端不必訂閱原始 RGB。

`config/dinov3_cases.yaml` 的 `video_fps`（1–30，最大輸出頻率）、`h264_bitrate`（bit/s）、
`h264_encoder`（auto／h264_nvenc／libx264）、`h264_topic` 與 `bbox_visualization_mode` 於啟動時讀入。
`auto` 實際嘗試 NVENC 編碼首張影像，無法使用時改用 libx264；兩者都停用 B-frame 與編碼緩衝。
無網頁或 ROS 訂閱者時不繪圖、不編碼；同一份最新畫面必要時重送為 IDR，供新觀看者加入。
每秒及新觀看者加入時請求 IDR；各接收端最多緩存 30 個 access unit，溢位後捨棄整組待傳封包，
等待下一個 IDR，避免任意丟 P-frame 造成花屏。DDS 發布由獨立 worker 處理，不阻塞 WebRTC 編碼。

相機模式有有效框且結果未超過 3 秒時顯示疊圖，否則顯示最新相機預覽；上傳結果不過期。
上傳圖片沒有 ROS header，使用編碼當下的節點時間，`frame_id` 為空。
WebRTC 目前使用 host ICE candidates，適用本機／可直連區網；跨 NAT 尚未配置 STUN/TURN。
最多同時 8 個網頁觀看者。播放器每 3 秒續租，連線中斷或分頁強制關閉後，
伺服器會在約 15–17 秒內回收未續租的已連線觀看者；協商未完成的連線最多保留 30 秒。狀態 API 的 `video_error` 與 `performance.video_*` 提供編碼診斷。

RViz2 可新增 **Image** display，選擇 `/medicine_recognition/image_annotated`，
Transport Hint 設為 `raw`、Reliability 設為 `Reliable`。
此 topic 由 `annotated_image_topic` 設定，發布編碼前的 960×540 `sensor_msgs/msg/Image`（`bgr8`），
保留來源 header，與網頁共用同一份疊圖及 `video_fps` 上限。
僅有 RViz2 訂閱時也會繪圖及發布，不啟動 H.264 編碼。
原始影像由獨立 worker 發布，待傳佇列只保留最新一張；DDS 傳送期間不持有共用鎖，
避免 RViz2 接收變慢拖住繪圖、追蹤與網頁編碼。RViz2 Image 的 Queue Size 建議設為 1，
並將全域 Frame Rate 設為 30。實際更新率仍受相機、追蹤與傳輸速度限制。

舊 `/view.mjpg`、`/camera.mjpg`、`/overlay.mjpg`、`/overlay.jpg`，
以及 `/dinov3/heatmap/overlay`、`/medicine_recognition/debug_image` 影像輸出已移除。
物件診斷與圖庫的靜態圖片仍保留。結構化結果維持 `/medicine_recognition/detections`、
`/medicine_recognition/detections_3d` 與 AprilTag topic。

ROS 接收端解碼範例（需先載入 ROS 與本專案虛擬環境）：

```python
import av
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

rclpy.init()
node = Node("h264_viewer")
decoder = av.CodecContext.create("h264", "r")

def receive(message):
    try:
        frames = decoder.decode(av.Packet(bytes(message.data)))
    except av.FFmpegError:
        return  # 初次訂閱可能尚未收到下一個 IDR。
    for frame in frames:
        bgr = frame.to_ndarray(format="bgr24")  # shape: (540, 960, 3)
        # 在這裡把 bgr 與 message.header 提供給應用。

subscription = node.create_subscription(
    CompressedImage, "/medicine_recognition/image_annotated/h264", receive, 30)
try:
    rclpy.spin(node)
finally:
    node.destroy_node()
    rclpy.shutdown()
```

ROS `CompressedImage.format` 允許自訂傳輸格式，詳見
[訊息定義](https://github.com/ros2/common_interfaces/blob/rolling/sensor_msgs/msg/CompressedImage.msg)。
WebRTC 使用 aiortc 的已編碼 `Packet` 路徑，詳見
[API](https://aiortc.readthedocs.io/en/latest/api.html#aiortc.MediaStreamTrack)。
