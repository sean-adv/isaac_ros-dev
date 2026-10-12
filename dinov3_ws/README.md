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
不是感測器精度量測。3D detection 的 ID 沿用同次 2D ROI 追蹤 ID。

只對非 `unknown` 物件估計姿態。RGB／深度必須同尺寸、同 frame，時間差不超過
`pose_sync_tolerance`（預設 0.05 秒），並有匹配的相機內參。
推論前保留深度快照，避免模型處理期間誤用後來的影格。
`16UC1` 由毫米轉公尺；`32FC1` 視為公尺，其他編碼拒絕。
框中心深度作為種子，保留同深度帶內相連區域，排除零值、NaN、超距和背景。
點數不足或主軸不明確（例如圓形／正方形表面）時不發布該物件的 6D 姿態。
辨識框中心落在背景、重疊物件或背景與物件深度接近時，這種分割仍可能誤估，
需以實機資料確認。所有物件的處理狀態可在 `/api/detections` 的 `pose_status`
查看，成功者另含 `pose`。
偵測疊圖在每個框旁顯示 `position (m): X=… Y=… Z=…`，保留三位小數；
沒有可用姿態的物件顯示 `position: N/A`。
有可用姿態時，疊圖也在表面質心畫出物件局部座標軸：X 紅、Y 綠、Z 藍，
各長 8 公分，依姿態四元數、相機內參與畸變投影。朝向視線的軸可能呈現為點。

每次辨識結果均可發布空陣列，表示當次沒有可用姿態；原本的 2D 辨識輸出保留。
上傳圖片沒有對應深度，因此只提供 2D 辨識。
參數在 `config/dinov3_cases.yaml`：距離範圍預設 0.1–5 m、深度帶 ±0.08 m、
至少 80 點、相鄰 PCA 特徵值比例至少 1.15。可用 `--ros-args -p` 覆寫，例如
`-p pose_enabled:=false` 停用姿態，或 `-p pose_depth_band:=0.12` 調整分割深度帶。

## 驗證

```bash
source /opt/ros/jazzy/setup.bash
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
