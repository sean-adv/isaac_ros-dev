# 3D 深度框與背景量測

[回程式導覽](../README.md) · [執行流程](runtime.md)

### 深度框的旋轉估計

`geometry/depth_bbox.py` 先從偵測框中央深度選出連通點雲，再用 RANSAC 擬合主要平面。
中央取樣先使用 bbox 約 20% 寬 × 20% 高；有效深度少於 5 點時，直接擴大到約 40% 寬 × 40% 高。
擴大後仍不足 5 點才回報 `no_valid_center_depth`；深度中位數與連通區選擇共用最終取樣區域。
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
背景快取沿用量測當時的 2D 目標框；兩次深度更新間物件移動時，可能暫時與最新 NvDCF 框不同。

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
`/tag_detections` 借用 AprilTag 訊息格式發布物件資料；每筆 `pose` 的 XYZ 與 quaternion
沿用 `Detection3D` 的框中心及旋轉，`pose.header` 保留來源量測時間與座標系。
陣列 header 與水平／垂直的 2D `corners` 仍對應目前影格。
朝向是深度表面估算，保留 180°／近正方形方向歧義；平面失敗時使用相機對齊旋轉，
缺少旋轉的舊資料也使用單位 quaternion。Covariance 未估算，保留訊息預設值。
`/api/status` 的 `performance.bbox3d_schedule` 顯示獨立間隔與背景量測耗時；`bbox3d_ms` 則是前景提交／讀取快取的耗時。
參數在啟動時讀入，修改 YAML 後需重啟節點。

