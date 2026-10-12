# OmDet-Turbo-Tiny → DINOv3 商品辨識

OmDet 使用離線文字特徵找出物體框，裁切後以 DINOv3 ViT-S/16 比對全部商品圖庫。
每個 case 是商品標籤；相似度或前兩名差距不足時顯示 `unknown`。

## 啟動

```bash
cd /workspaces/isaac_ros-dev
./dinov3_ws/run_camera.sh           # 需要相機時，另開終端
./dinov3_ws/run.sh                 # 預設 single：兩個模型共用 TensorRT 11.1
./dinov3_ws/run.sh --mode split     # OmDet 使用同環境的私人 IPC 子程序
```

兩個推論命令擇一。網頁為 http://127.0.0.1:8090，支援 ROS 相機及圖片上傳；
`/images` 提供圖庫管理及 HSV 去背。選取 case 只切換預覽，辨識仍比對全部圖庫。
門檻、JPEG、推論頻率與 RGB-D 參數在 `config/dinov3_cases.yaml`，可用 `--ros-args -p` 覆寫。

最小 bbox 可設定 `grounding_min_bbox_width`、`grounding_min_bbox_height`（像素）及
`grounding_min_bbox_area`（平方像素）。三者預設為 `0`，各自表示不限制；
啟用後，框必須同時滿足所有門檻，等於門檻時保留。尺寸以原圖座標裁切至影像範圍並取整後計算，
不是 OmDet 的 640×640 或 DINOv3 的 224×224 輸入尺寸。
小框在套用偵測數量上限及 DINOv3 辨識前移除，single／split、相機與上傳图片均適用。
例如設定最小寬高 32 像素、面積 1024 平方像素：

```yaml
grounding_min_bbox_width: 32
grounding_min_bbox_height: 32
grounding_min_bbox_area: 1024
```

修改 config 後重新啟動推論程序；也可在啟動命令加上
`--ros-args -p grounding_min_bbox_width:=32 -p grounding_min_bbox_height:=32 -p grounding_min_bbox_area:=1024`。

## 官方匯出與離線提示

來源是 [OmDet 官方儲存庫](https://github.com/om-ai-lab/OmDet)，沿用官方
`OmDetV2TurboInfer`、ONNX opset 17 與四個固定尺寸輸入：

| 輸入 | 尺寸 | 意義 |
| --- | --- | --- |
| img_tensor | 1×3×640×640 | RGB，直接 resize，均值／標準差正規化 |
| label_feats | max_labels×1×512 | CLIP 類別 EOT 特徵，含 text_projection |
| task_feats | max_prompt_tokens×1×512 | 任務句特徵，補至指定容量 |
| task_mask | 1×max_prompt_tokens | 有效 token 為 1，padding 為 0 |

類別名稱與任務句分開編碼；使用官方修改版 CLIP，並載入 OmDet checkpoint 中的
`language_backbone` 權重。**不要使用其他 CLIP 的一般 pooled embedding 代替任務 token 特徵。**
ONNX 不含文字編碼與後處理。TensorRT 11 引擎預設使用 FP16（NVIDIA ModelOpt AutoCast，FP32 I/O）。
偵測解碼器、LayerNormalization、Softmax、ReduceMean／Sum／Max 及校驗判定的敏感節點保留 FP32，
卷積與矩陣乘法採 FP16 張量、明確 FP32 累加，再轉回 FP16，避免 GPU 低精度累加造成分數偏差；
後處理對類別 logits 套用 sigmoid，
將 normalized cxcywh 還原為原圖 XYXY，再做 NMS 與重複物體抑制。

建置時指定容量，只需在首次建置或增加容量時重建：

```bash
./dinov3_ws/build_omdet.sh --max-labels 4 --max-prompt-tokens 77
# 新工作區下載官方權重：同上加 --download
```

`--max-labels` 是最大類別數，`--max-prompt-tokens` 是任務文字的最大 token 數，
包含 CLIP 起訖標記，範圍 2–77；不是字元數。每個類別名稱也受 CLIP 77-token 上限限制。
容量會記錄在 `prompts.json` 和 engine metadata，決定 ONNX／engine 的固定輸入尺寸。
只有匯出／完整建置時可設定容量；從已有 ONNX 建 engine 時沿用匯出的容量。

`config/omdet_prompts.json` 每個 profile 只包含 `prompt` 與 `labels`：

```json
{
  "profiles": [
    {
      "prompt": "Detect medicine bottle, medicine box.",
      "labels": ["medicine bottle", "medicine box"]
    }
  ]
}
```

`prompt` 同時是實際任務句和網頁／ROS 選擇鍵；`labels` 是有序的實際輸出類別。
`config/dinov3_cases.yaml` 的 `grounding_prompt` 指定上述 prompt。
不再需要 `task`、`output_labels`，也不會自動從 prompt 文字猜測 labels。

日常修改文字或 labels 後只執行：

```bash
./dinov3_ws/build_omdet.sh --stage-prompt
# --stage prompts 仍是同一模式的別名
```

此模式只用已有官方 CLIP／checkpoint 編碼特徵及更新 metadata，不建立偵測模型、
不匯出 ONNX、不建置 engine、不重新進行 AutoCast 校驗。
各 profile 可有不同類別數，只要 `len(labels) ≤ max_labels`，且任務文字 token 數不超過
`max_prompt_tokens`。超限直接報錯，不截斷、不改寫 config，也不替換既有提示產物。
類別較少時複製本次最後一個有效類別的特徵補足固定槽位；不引入其他類別。
推論後只取實際 labels 對應分數，再進行 top-k／NMS；不輸出補位槽位。
任務特徵補零至容量，`task_mask` 標示有效 token。固定容量下減少 labels 不會縮小引擎。

執行前停止推論，完成後重新啟動，讓文字快取與選單重新載入。
若修改了選擇鍵，也須同步 YAML 的 `grounding_prompt`。
更新同步覆蓋 `prompts.npz`、`prompts.json` 與容量相同的 `omdet_tiny*.engine` 的 JSON metadata，
保留 engine／ONNX hash、精度與原建置資訊；資料全部驗證成功後才替換，寫入失敗會回復。
現有舊引擎會沿用原 `class_count` 作為 `max_labels`，任務容量為 77，不需立即重建。
若目錄中保留其他容量的舊精度引擎，提示更新會跳過它；須重新建置該精度後才能使用新提示檔。
官方原始碼 commit 必須與既有引擎相同。换提示後仍應用相關實圖檢查數值與辨識結果。

```bash
# 從已有 ONNX 建立 FP16 引擎／獨立 FP32 基準：
./dinov3_ws/build_omdet.sh --stage engine --precision fp16
./dinov3_ws/build_omdet.sh --stage engine --precision fp32
```

FP16 建置以 `--calibration-images` 指定多張實圖；未指定時使用本工作區既有三張商品圖，
並對所有離線提示進行校驗。校驗只用於建置，不影響執行時的離線文字特徵。
TensorRT timing cache 存於 `artifacts/omdet/omdet_timing.cache`，供後續重建沿用。

匯出環境使用現有 `detectron2/` 原始碼與 `dinov3_trt11_venv`，另需 timm 0.9.16、ftfy、
lmdb、joblib、fvcore、iopath、hydra-core、pycocotools、cloudpickle、termcolor、yacs、tabulate。
舊官方 requirements 的 Torch／Transformers／Pillow 版本不適用本機 Python 3.12／Thor，
因此使用既有可用的 CUDA 13 PyTorch 環境。

| 產物 | 路徑（相對 artifacts/） |
| --- | --- |
| 官方 FP32 ONNX | omdet/omdet_tiny.onnx |
| AutoCast FP16 ONNX | omdet/omdet_tiny_fp16.onnx |
| OmDet FP16 TensorRT | omdet/omdet_tiny.engine |
| FP32 比對基準 | omdet/omdet_tiny_fp32.engine |
| 引擎 metadata | omdet/omdet_tiny.json |
| 離線提示特徵及說明 | omdet/prompts.npz、omdet/prompts.json |
| DINOv3 TensorRT | tensorrt11/dinov3_vits16_fp16_224.engine |

引擎 metadata 保存官方 commit、版本及 SHA256；載入時核對 engine 和提示特徵，避免混用。
TensorRT engine 與 GPU／TensorRT 版本綁定，移到其他機器需重建。
DINOv3 為 `facebook/dinov3-vits16-pretrain-lvd1689m`，保持 FP16 計算及 FP32 I/O；
參考圖和 ROI 等比例縮放長邊至 224，再於右／下方複製邊緣，補成 224×224。

## 流程與輸出

`omdet_trt.py` 負責離線提示、TensorRT、前後處理；`grounded.py` 負責 ROI、圖庫及辨識。
`runtime.py` 集中路徑，`launch_pipeline.py` 選擇 single／split。
`node.py` 處理 ROS、最新影格與發布執行緒，`web.py` 提供 HTTP。
`grounding_ipc.py` 在 split 模式傳輸可重用 RGB 共享記憶體、設定及結果，核對 frame ID，
於正常關閉、故障或逾時清理 worker 與共享記憶體。

保留最新影格與單筆待發布結果，跳過過期資料；DDS 發布不阻塞推論。
診斷圖重用同次 DINOv3 特徵，不重跑模型。輸出沿用 `/medicine_recognition/detections`、
`/medicine_recognition/debug_image` 與 `/dinov3/heatmap/overlay`。
圖庫位於 `dinov3_cases_data/<case>/*`。

相機辨識使用簡單 ROI 追蹤：以擴張框 IoU 篩選候選，再依原框重疊程度做
一對一配對；標籤改變或變成 `unknown` 不會直接換 ID。疊圖、網頁結果及
`/api/detections` 的 `track_id` 與 ROS 2D／3D 訊息的 `detections[].id` 使用相同 ID。
預設容忍連續 5 個已執行偵測的影格漏偵測，第 6 個漏偵測影格移除追蹤；
模型跳過的相機影格不計數，重跑同一影格也不增加漏偵測次數。
漏偵測期間僅保留內部 ROI，不輸出舊框、辨識或姿態。
切換輸入／設定、上傳新圖片或影像尺寸／frame 改變時清除 ROI，ID 不重複使用。
可調整 `tracking_max_missed`、`tracking_iou_threshold`（預設 0.2）及
`tracking_roi_padding`（預設每側擴張原框寬／高的 0.5 倍）。
這是幾何配對；物件交叉、重疊或移動超出 ROI 時仍可能換 ID，需要實機確認。

### 追蹤穩定後才顯示 bbox

`config/dinov3_cases.yaml` 加入兩個顯示門檻：

```yaml
tracking_bbox_min_hit_ratio: 0.8
tracking_bbox_min_detections: 10
```

每個 track ID 從首次偵測開始，累計「命中的唯一偵測影格數」及
「已執行偵測的唯一影格總數（包含漏偵測）」。命中率為前者除以後者。
只有**命中率 > 0.8 且命中幀數 > 10** 時才畫 bbox、ID／商品文字、位置文字與姿態軸。
因此連續命中到第 10 幀仍不顯示，第 11 幀才顯示；12／15 恰好 80% 不顯示，
13／16 為 81.25% 才通過。每幀重新判斷，比例降低時會再次隱藏。

模型跳過的相機影格不納入分母，同一影格因圖庫更新而重跑不重複計數。
尚未顯示的物件仍持續追蹤與辨識；漏偵測時不畫舊框。
track 到期或輸入／追蹤情境重置後，新 ID 從 1 個命中幀重新累計。
單張上傳圖片只有一個唯一影格，因此預設不會通過 10 幀門檻；若要顯示單張圖片的框，
可將 `tracking_bbox_min_detections` 設為 `0`。

這兩個參數控制影像疊圖顯示；`/api/detections`、ROS 2D／3D 結果與商品診斷圖仍保留當次原始辨識資料。
HTTP 每個 detection 額外含 `track_detected_frames`、`track_total_frames`、
`track_hit_ratio`、`bbox_visible`，可檢查尚未顯示的原因。
門檻修改後重新啟動程序，亦可用 `--ros-args -p` 在啟動時覆寫。

網頁「辨識結果」的「框顯示」可選 2D bbox 或 3D bbox 線框，預設 2D。
3D 模式將與 ROS 相同的八個長方體頂點，用該影格相機內參／畸變投影回 RGB 影像。
只顯示已有 3D bbox 的非 unknown 物件，沿用追蹤穩定門檻；深度缺失時不畫替代 2D 框。
兩種疊圖共用同次推論結果，切換不重跑模型，各瀏覽器可獨立選擇。
`/overlay.mjpg?bbox=3d` 與 `/overlay.jpg?bbox=3d` 提供 3D 疊圖；省略參數或 `bbox=2d` 為 2D。

## 由 2D 框推算 3D 長方體

`run_camera.sh` 預設啟用 `depth_registration:=true`，將深度對齊彩色影像。
使用 `/camera/depth/image_raw` 和 `/camera/depth/camera_info`，RGB／深度須同尺寸、
同 optical frame，時間差不超過 `pose_sync_tolerance`（目前 YAML 設定 0.2 秒）。
推論前保留同步深度快照；`16UC1` 毫米轉公尺，`32FC1` 視為公尺。

讀取 2D bbox 四角各自的深度（右／下邊界為 exclusive，取框內最後一個像素）。
每角沿 2D 框中心到角落的直線，由內往外搜尋。
先取得框內最近的有效 Z；只比較相鄰且都有效的兩個深度值，不跨越零值／NaN 等孔洞。
候選跳變須至少 `bbox3d_edge_min_jump`（目前 0.03 m），且靠中心一側深度須在
`[最近 Z, 最近 Z + bbox3d_depth_band]`（目前 0.08 m）內，以避免選遠處背景內部的邊緣。
取候選中絕對深度落差最大的位置，選其靠中心一側的像素與深度；相同落差時取靠中心者。
沒有明顯跳變時，取線上離角落最近的合理深度點；連合理點都沒有時不輸出該 3D 框，
狀態為 `no_reasonable_corner_depth:<角落編號>`。
最大深度跳變是邊界線索，不保證屬於商品；重疊物件、反光與有效範圍內的雜訊仍可能誤選。
另找整個框內最近的有效深度 Z 與其像素位置；排除零值、NaN、Inf 及超出距離範圍的值。
此處「最近」指 optical Z 最小，不是歐氏距離最小；最小值仍可能受有效範圍內的雜訊影響。

將四條線找到的實際像素位置與框內最近點依各自深度反投影到 3D，先校正相機畸變。
不把線內深度套回原始角落位置，避免放大範圍。
無畸變時每點為 `X=(u-cx)*Z/fx`、`Y=(v-cy)*Z/fy`。
取這五點各軸的最小／最大值：`size = maxXYZ - minXYZ`，
`center = (maxXYZ + minXYZ) / 2`，得到與相機座標軸對齊的長方體。
四元數固定為 `[0,0,0,1]`，不再使用 PCA 或 `max(寬,高)` 假設厚度。
Z 厚度為四角最遠深度與框內最近深度之差；所有深度相同時厚度為 0，不捏造厚度。

這是五個取樣點的包圍框，未涵蓋未取樣點或不可見背面。
角落落在背景或框內有其他較近物件時，結果可能包含背景／其他物件，
不能直接視為完整商品尺寸；後續需物件遮罩或多視角資料才能進一步改善。

```bash
./dinov3_ws/run_camera.sh
# 另一個終端：
./dinov3_ws/run.sh --mode split
# 再開一個終端：
source /opt/ros/jazzy/setup.bash
source dinov3_ws/install/setup.bash
ros2 topic echo /medicine_recognition/detections_3d
```

3D topic 改為 `/medicine_recognition/detections_3d`，型別 `dinov3_cases_msgs/msg/Detection3DPointsArray`：

- `header`：原始 RGB 影格的時間戳與 optical frame（X 向右、Y 向下、Z 向前）。
- `detections[].id`：與當次 2D detection 相同的追蹤 ID。
- `detections[].bbox.center`：上述中心位置（公尺）與固定單位四元數。
- `detections[].bbox.size`：推算長方體的寬、高、厚度（公尺）。
- `detections[].hypothesis.class_id`／`score`：商品標籤與 DINOv3 相似度。
- `detections[].bbox_2d`：原始 2D 框中心與尺寸（pixel）。
- `detections[].bbox_2d_corners`：原始 2D 框四角（pixel）；右／下為原始 exclusive 邊界。
- `detections[].sampled_2d_points`：四條角落到中心線上實際找到的像素位置（pixel）。
- `detections[].sampled_3d_points`：對應四個實際取樣位置的 3D 點（公尺）。
- `detections[].nearest_2d_point`／`nearest_3d_point`：框內最近深度點的 2D／3D 位置。
- `detections[].bbox_3d_corners`：最終長方體的八個 3D 頂點（公尺）。

四角及取樣點順序為左上、右上、右下、左下；八頂點先列 min-Z 面四角，再列 max-Z 面四角，
每面順序為 min-X/min-Y、max-X/min-Y、max-X/max-Y、min-X/max-Y。
「所有 3D 點」包含四個反投影取樣點、最近深度點與八個包圍框頂點，不是整張深度圖點雲。

自訂訊息原始檔位於 `src/dinov3_cases_msgs`。`run.sh` 首次啟動自動編譯並載入訊息；
修改 `.msg` 後需重新建置：

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base dinov3_ws/log build --base-paths dinov3_ws/src/dinov3_cases_msgs \
  --build-base dinov3_ws/build --install-base dinov3_ws/install \
  --packages-select dinov3_cases_msgs --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source dinov3_ws/install/setup.bash
```

原本訂閱標準 `vision_msgs/Detection3DArray` 的程式須改用新型別與上述欄位。

自訂訊息不包含 `results[].pose` 或 covariance。位置與尺寸請讀 `bbox`。
已停用 `/medicine_recognition/poses`。原本 2D topic 保留；ROS 2D／3D detection
都過濾 `unknown`，內部追蹤與 HTTP 辨識資料仍保留。已識別物件須在四條角落到中心
的線上及框內有有效深度，才輸出 3D 框。全為 `unknown` 或無可用結果時發布空陣列。
上傳圖片沒有對應深度，因此只提供 2D 辨識。

HTTP `/api/detections` 改用 `bbox3d_status`；成功者含 `bbox3d` 的 `position`、`size`、
`corners`、`corner_depths`（依左上、右上、右下、左下排序）、`nearest_point`、
`nearest_depth`、`nearest_pixel`、`depth_range`、`foreground_depth_band`、`corner_pixels`（原始四角）、
`corner_sample_pixels`（實際取樣位置）、`corner_edge_jumps`、
`corner_sampling_methods`（max_depth_jump 或 reasonable_depth_fallback）、`edge_min_jump` 與 `method`，不再附帶 `pose`。
疊圖仍顯示中心位置與 `sqrt(X²+Y²+Z²)` 距離（公尺，三位小數），無框時顯示 N/A。
顯示穩定門檻只影響疊圖，不延遲 ROS 2D／3D 輸出。

為保留啟動參數相容性，深度功能仍使用 `pose_enabled`、`pose_sync_tolerance`、
`pose_min_depth`、`pose_max_depth`；目前有效深度範圍為 0.1–5 m。
`bbox3d_edge_min_jump` 控制最小可接受深度跳變（公尺）。
`bbox3d_depth_band` 控制沿線搜尋與最近深度的最大差距；若真實可見表面深度跨度
超過 8 cm，可提高此值，但可能再次納入背景。最近深度仍採框內最小有效 Z，
框內若有其他更近物件或有效範圍內雜訊，仍可能選錯前景或拒絕輸出。
`-p pose_enabled:=false` 可停用 3D 框；PCA 的深度帶、最少點數與軸比例參數已移除。

## 驗證

先建置上方的自訂訊息套件，再執行：

```bash
source /opt/ros/jazzy/setup.bash
source dinov3_ws/install/setup.bash
source dinov3_trt11_venv/bin/activate
export PYTHONPATH="$PWD/dinov3_ws/src/dinov3_cases:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/tensorrt_libs:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q dinov3_ws/tests
python dinov3_ws/tests/check_omdet_trt.py
python dinov3_ws/tests/check_grounded_trt_pipeline.py --mode single
python dinov3_ws/tests/check_grounded_trt_pipeline.py --mode split
python dinov3_ws/tests/check_grounding_ipc_gpu.py
```

新工作區進行 FP16／FP32 比對前，先執行 `build_omdet.sh --stage engine --precision fp32` 建立獨立基準。
本工作區已保留原 FP32 引擎。

GPU FP16 比對報告存於 `artifacts/omdet/validation_fp16.json`（FP32 驗證為 `validation.json`），ROS／HTTP 整合報告為
`artifacts/omdet/integration_single_224.json` 與 `integration_split_224.json`。
已在 Thor 驗證 PyTorch／ONNX／TensorRT 候選框配對、實圖偵測、77 token task padding，
以及 single／split 的 ROS／HTTP／DINOv3 圖庫流程。
FP16 驗證比對保留的 FP32 引擎與 FP32 ONNX，包含未用於 AutoCast 校驗的圖片。
有效偵測要求類別及數量一致、框 IoU ≥ 0.98、分數差 ≤ 0.03，並檢查所有輸出有限。
暖機後 FP32／FP16 的實圖速度記錄於報告的 `timings`；量測包含前後處理，
不含 DINOv3，實際相機效能需依影像與物件數量量測。
本機四張實圖（含旋轉圖）驗證：類別及數量一致，原圖最大分數差約 0.0036，
旋轉圖約 0.0224；因此以 0.03 的分數差作為 FP16 回歸上限。
相同實圖、暖機後 20 次量測：FP32 約 92.3 ms，FP16 混合精度約 72.8 ms（約 1.27×）。
`/api/status.grounding_precision` 顯示目前 OmDet 引擎精度。

Grounding DINO 程式碼、重建腳本與專用虛擬環境已移除；grounding 僅使用 OmDet。
先前轉換的 Grounding DINO ONNX／TensorRT 權重及附屬資料保留於
`artifacts/` 與工作區的 `GroundingDINO-TensorRT-and-ONNX-Inference/weights/`，供歷史封存。
`grounding_text_threshold`、HTTP `text_threshold` 與 `grounding_text_token_limit` 狀態欄位已移除。
