# Grounding DINO → DINOv3 物體辨識

預設流程已改為：**RGB + 文字提示 → Grounding DINO 物體框 → 每框裁切 → DINOv3 特徵 → 所有商品參考圖庫 → 商品名稱或 unknown**。

## 啟動

Grounding DINO 保留舊 TensorRT 10.16、DINOv3 使用新 TensorRT 11.1 的雙程序入口為
`./run_split_trt.sh`；共享記憶體傳輸、版本、延遲量測與回切方式見 [SPLIT_TRT.md](SPLIT_TRT.md)。

```bash
cd /workspaces/isaac_ros-dev/dinov3_ws
./run_camera.sh
# 另一個終端；預設 Grounding DINO TRT + DINOv3 ViT-S/16 TRT
./run.sh
# 明確指定 ViT-S/16
./run.sh --model vits16
# 仍可選用既有 ViT-B/16、ViT-H/16+
./run.sh --model vitb16
./run.sh --model vith16plus
```

預設使用 [wingdzero/GroundingDINO-TensorRT-and-ONNX-Inference](https://github.com/wingdzero/GroundingDINO-TensorRT-and-ONNX-Inference) 的 Swin-T OGC 權重，依上游外部文字 mask 的匯出方法建立 FP16 TensorRT 引擎；DINOv3 使用 `facebook/dinov3-vits16-pretrain-lvd1689m` 的 BF16 TensorRT 引擎（RoPE 使用 FP32、輸入輸出 FP32）。Grounding DINO 的 tokenizer／結果後處理使用 Hugging Face 實作，模型載入失敗會在網頁顯示原因。

Grounding DINO 圖片直接縮放為固定 800×1200（長寬比可能改變），RGB/ImageNet normalization，輸出框映射回原圖。預設提示詞 `medicine bottle. medicine box. bottle. box.` 共 12 tokens；預設使用第二次匯出的 12-token engine，較短提示詞會補齊 padding，超過 12 tokens 會拒絕。可即時修改容量內的提示詞與門檻；增加文字容量需重新編譯。DINOv3 保留動態尺寸範圍 32–2560，預設引擎針對 448×448 最佳化。ViT-S/16 特徵維度為 384。目前 grounded 設定統一使用 224×224 正方形補邊輸入；engine 本身仍保留動態尺寸。TensorRT 11.1 試用與回切方式見 [TRT11.md](TRT11.md)。

```bash
# 使用獨立 groundingdino_venv 與 wingdzero 上游腳本重新匯出／编譯
bash build_grounding_dino_wingdzero_trt.sh
# 以較長提示詞重新設定 engine 的固定文字容量
bash build_grounding_dino_wingdzero_trt.sh 'medicine bottle. medicine box. bottle. box. package.'
# DINOv3 ViT-S/16 的重建方法
../dinov3_venv/bin/python export_dinov3_vits_dynamic.py
bash build_dinov3_vits_trt.sh
```

預設 Grounding DINO engine 為第二次匯出的 `artifacts/tensorrt/grounding_dino_wingdzero_fp16.engine`（12 tokens），使用藥瓶／藥盒提示詞。`build_grounding_dino_wingdzero_trt.sh` 可重建該模型。第一次匯出的 4-token engine 保留於 `artifacts/tensorrt/grounding_dino_wingdzero_car_fp16.engine`。processor 小檔案位於 `artifacts/grounding_dino_wingdzero_processor`。原社群 FP32 engine 與其建置腳本仍可透過引擎路徑和 model_id 參數使用。TensorRT 引擎需在部署 GPU／相容 TensorRT 版本建置。

獨立轉換 venv 為 `../groundingdino_venv`；執行 ROS／DINOv3 流程仍使用 `../dinov3_venv`。`./run.sh` 已切換至新的 Grounding DINO engine；如需 PyTorch Grounding DINO，可用 `./run.sh --ros-args -p grounding_backend:=torch`。DINOv3 可獨立使用 `-p inference_backend:=torch`。

開啟 <http://localhost:8090>：

1. 每個 case 放同一商品的多角度裁切照片，例如 `medicine_A`、`medicine_B`。所有非空 case 都參與辨識；「檢視此圖庫」僅選擇預覽圖片。
2. 在文字提示輸入 `medicine bottle. medicine box. bottle. box.`，按「套用設定」。英文類別用句點分隔；提示詞不得超過目前 engine 的 12 token 容量（包含特殊 tokens 與句點）。
3. 使用 ROS 相機或上傳目標圖。每個框顯示商品名稱、偵測文字、`det` 偵測分數及 `sim` 圖庫相似度。
4. 調整偵測門檻、文字門檻、辨識相似度與前兩名最小差距。設定立即生效並重新處理靜態目標，僅保存至當前程序；永久設定請修改 `config/dinov3_cases.yaml`。

## 辨識方式與限制

Grounding DINO 的框會映射回原始 RGB 座標、裁齊邊界，經跨文字類別的 NMS 移除重複框。NMS 保留較高分的框，不取外接聯集；最多辨識 `grounding_max_detections` 個框。

參考圖片與每個 ROI 使用相同的等比例縮放：長邊設為 `object_max_side`，預設以右側／底部複製邊緣補成 224×224 正方形；`object_square_input=false` 可恢復短邊至少 32、補齊 feature stride 的模式。DINOv3 產生正規化 patch 特徵，取平均後再次 L2 正規化，作為整個物體的描述向量。每張參考圖保留一個向量，各 case 取其最高 cosine similarity；以最高分與不同 case 的第二高分之差進行接受判定。這是無需訓練的初版視覺檢索，門檻需以真實商品驗證集校準。

- `low_similarity`：最高分低於辨識門檻。
- `ambiguous`：最高分達標，但不同 case 的前兩名分差不足。
- `empty_gallery`：沒有參考圖片，仍輸出偵測框與 `unknown`，相似度為 `null`。
- `matched`：接受商品身分。只有一個 case 時僅檢查絕對相似度，`margin` 為 `null`。

綠框代表 matched，黃框代表 unknown。相似度不是辨識機率；相近包裝及文字／劑量差異可能混淆。透明參考圖片目前仍轉為 RGB，不用 alpha 作為特徵遮罩。Grounding DINO 漏掉的物體不會進入辨識階段。

圖庫透過網頁更新時即失效重建，外部新增／修改／刪除檔案約每秒檢查一次（推論繁忙時延後）；只完整替換成功建立的圖庫。錯誤圖片會顯示錯誤，不使用過期圖庫繼續辨識。相機只處理最新影格，保留原始時間戳與 frame ID，沒有跨幀追蹤。整體速度依物體數量和模型而變，`inference_hz` 是輪詢頻率上限，不保證 FPS。

## 逐物體視覺化

grounded 模式的頁面底部預設顯示每個 Grounding bbox 的 DINOv3 熱圖，物體編號與上方完整 bbox 結果一致，不需要開啟 `debug_visualization`。可切換 ROI 原圖、實際模型輸入與 patch 網格、逐 patch 相似度、純熱圖，以及最佳參考圖。每張卡片顯示原圖 bbox、ROI／輸入尺寸、patch grid、商品相似度與參考檔名。

「比對 reference 標籤」選單可選擇自動最佳參考圖或任一非空標籤。指定標籤後，**所有** Grounding object 都會與該標籤全部參考圖片的 patch 比對，不會依目前辨識結果篩掉物體。卡片另顯示 `標籤 sim`（該標籤內最高的整體描述向量相似度）、參考圖數量，以及該標籤內對此物體最佳的參考檔名。商品辨識結果與 ROS 輸出保持原本的全圖庫比對；選單僅控制視覺化比較。

自動模式的熱圖對該 ROI 的每個 patch 計算與最佳參考圖所有 patch 的最高 cosine similarity；指定標籤時，改用該標籤全部圖片的 patch bank。兩者固定以 0–1 色階顯示（負值呈深藍）。ROI 與參考圖特徵均重用辨識階段的 forward 結果，切換標籤不再執行 Grounding DINO 或 DINOv3 forward。商品 `score` 仍是池化後整體描述向量的相似度，熱圖則用於檢查局部圖案匹配；高分局部區域不代表商品已被接受。`unknown` 也可檢查熱圖；空圖庫只有 ROI 與網格可顯示。

網格與逐 patch 視圖包含實際模型輸入的補邊，藍框標示有效 ROI。疊加／純熱圖先去除右側與底部補邊，再映射回 ROI 長寬比。預覽長邊最多 640 pixels，bbox 與尺寸資訊仍使用原始座標。模型輸入與參考圖快取對應同一次圖庫建立，不會讀取已變動檔案混用舊特徵。

新模式移除舊 mask／形態學處理選項，因為目前沒有物體分割；完整 bbox 已在上方顯示。辨識門檻保留在「偵測與辨識設定」，不拿整體商品門檻來切局部熱圖。圖庫選擇僅更新參考圖預覽，不清除物體結果或重新跑辨識。`pipeline:=heatmap` 仍保留原有五種全圖除錯視圖與 `debug_visualization` 開關。

## 參數

| 參數 | 預設 | 用途 |
| --- | --- | --- |
| `pipeline` | `grounded` | `grounded` 新流程；`heatmap` 舊全圖流程 |
| `grounding_backend` | `trt` | Grounding DINO 推論後端：`trt`／`torch` |
| `grounding_engine_path` | `artifacts/tensorrt/grounding_dino_wingdzero_fp16.engine` | Grounding DINO 引擎路徑 |
| `grounding_prompt` | `medicine bottle. medicine box. bottle. box.` | 文字提示 |
| `grounding_box_threshold` | 0.30 | 候選框分數門檻 |
| `grounding_text_threshold` | 0.25 | 文字片語門檻 |
| `grounding_nms_iou` | 0.50 | 重複框抑制 IoU |
| `grounding_max_detections` | 20 | 每幀最大 ROI 數量 |
| `object_max_side` | 224 | 參考圖與 ROI 的共同長邊尺寸 |
| `object_square_input` | config 為 true | 將參考图與 ROI 補成固定正方形輸入 |
| `recognition_threshold` | 0.75 | 接受商品身分的 cosine 門檻 |
| `recognition_margin` | 0.05 | 不同商品前兩名最小分差 |

`reference_max_side`、`target_padding`、`cosine_threshold`、形態學與框合併參數僅影響舊 heatmap 流程。新流程的參考圖網格預覽依 `object_max_side` 顯示。

## API 與 ROS

`POST /api/grounding` 接受部分或全部設定：

```json
{"prompt":"medicine bottle. medicine box.","box_threshold":0.3,"text_threshold":0.25,"recognition_threshold":0.75,"margin":0.05}
```

`GET /api/status` 增加 `pipeline`、`grounding`、`grounding_model_id`、`grounding_backend`、`object_max_side`。
`GET /api/detections` 的每個物體包含 `bbox`（原圖半開 XYXY）、`area`（矩形面積）、`label`、`candidate`（最接近商品）、`score`（DINOv3 相似度或 null）、`margin`、`reason`、`detection_label`、`detection_score`、`pose`、`pose_error`。
有可用參考圖時另含 `reference`（case、name）。`GET /api/object_visualizations` 回傳逐物體尺寸、視圖清單與 `revision`；影像使用 `/debug/objects/<revision>/<index>/<view>.jpg`。兩個端點皆可加上 `?reference_case=<標籤>`，未指定則使用自動模式。manifest 另回傳 `reference_cases`、`reference_case`、`ready`；物體資料另含 `comparison_case`、`comparison_score`、`comparison_reference`、`comparison_reference_count`。不存在或無圖片的標籤回傳 400。

快取保留最近三個已請求 revision 的 ROI 特徵與當時的參考圖庫，標籤比較在請求時產生；每幀最多保存自動模式及兩個指定標籤的影像。設定／來源／圖庫變更會清空；已失效 revision 回傳 410，頁面會重新取得最新結果。切換標籤時會清除舊標籤卡片，快速切換所產生的過期請求不會覆蓋新的選擇。

- `/medicine_recognition/detections`：`Detection2DArray`；每個框的 `class_id` 是商品名稱或 `unknown`，`score` 是圖庫相似度（無圖庫為 0），不是 Grounding DINO 分數。無物體時發布空陣列。ID 是單幀編號。
- `/medicine_recognition/debug_image` 與既有 `/dinov3/heatmap/overlay`：原圖加物體框與標籤，後者保留為顯示相容介面。
- 新流程不產生全圖 similarity topic，也不發布 `/medicine_recognition/pose`：矩形框不是物體 mask，目前 `pose=null` 並說明需要物體分割。之後接入分割才能沿用表面姿態估計。

## 驗證

```bash
cd /workspaces/isaac_ros-dev
source /opt/ros/jazzy/setup.bash
export PYTHONPATH="/workspaces/isaac_ros-dev/dinov3_ws/src/dinov3_cases:$PYTHONPATH"
/workspaces/isaac_ros-dev/dinov3_venv/bin/python -m pytest -q dinov3_ws/tests/test_grounded.py
/workspaces/isaac_ros-dev/dinov3_venv/bin/python dinov3_ws/tests/check_grounded_offline.py
```

離線整合測試使用隨機小型 DINOv3 與固定偵測框，驗證 API、圖庫生命週期、靜態重算及 ROS header；不代表正式權重的辨識準確率。

本次驗證：119 個單元測試通過，新舊流程的離線整合測試通過，ROS 套件重建成功。全套收集另受既有 `test_reference_alpha.py` 與 `test_rope.py` 阻擋：兩者引用基線程式已不存在的 `load_image`／`disable_rope` 等函式；上述 119 個測試排除這兩個舊測試檔。

正式 GPU smoke test 使用 Grounding DINO tiny 與 ViT-B TensorRT，載入 2 個 case、10 張參考圖。在圖庫中的一張 `stomach` 照片上偵測到 `medicine box` 並比對為 `stomach`（偵測分數 0.643、相似度 0.989、分差 0.211，單次約 1.86 秒）。此為圖庫照片的功能驗證，不是獨立測試集準確率或穩態 FPS。

## 舊 heatmap 模式與共用工具

原有全圖 patch 熱圖、深度姿態與圖片管理說明見 [HEATMAP_LEGACY.md](HEATMAP_LEGACY.md)。全圖熱圖／姿態功能需以 `./run.sh --ros-args -p pipeline:=heatmap` 啟動；舊預設值以實際 config 為準。

DINOv3 模型授權／登入設定見 [舊版環境說明](HEATMAP_LEGACY.md#first-hugging-face-access)；TensorRT engine 建置見 [TENSORRT.md](TENSORRT.md)。


## 即時延遲與效能

預設 `inference_hz=0.0`：新影格直接喚醒推論，處理最新影格，不追趕舊影格。
若指定正值，採用每輪開始時間的頻率上限；處理已超時就直接繼續，不再額外等待。
相機訂閱使用 BEST_EFFORT、KEEP_LAST depth=1。grounded 模式不訂閱深度與校正資料；heatmap 估姿仍保留。

相機預覽 `/camera.mjpg` 與辨識結果 `/overlay.mjpg` 都在新 JPEG 完成時通知 HTTP 串流，
移除原本 15 FPS 預覽與約 4 FPS 結果圖片輪詢限制。`/overlay.jpg` 保留供單張擷取。
串流只有最新影格，不建立應用層的每客戶端影格佇列；TCP、代理與瀏覽器仍可能緩衝。
`jpeg_quality` 預設 80（1–100），`preview_max_width` 預設 960；結果圖保留原始解析度。
這些設定不改變模型輸入與辨識門檻。

逐物體診斷圖片預設不更新，勾選「開啟診斷圖片」後才按需產生並快取，沿用既有特徵，不重跑模型。
開啟診斷仍會消耗 CPU/GPU。grounded 的 ROS 輸出改由獨立執行緒發布，
只保留最新待發布結果，且只有影像 topic 有訂閱者時才轉換 ROS Image。
慢速 ROS 訂閱者可能看到跳幀，目的是保留即時性；相機接收與瀏覽器不必等待 DDS 發布。

`/api/status.performance` 提供 grounded 的 `detect_ms`（含輸入轉換）、`recognize_ms`、
`render_ms`、`processing_ms`、`result_fps`（相鄰結果完成間隔倒數）及 `callback_to_result_ms`。
後者是 Python 相機 callback 到 JPEG 就緒的時間，**不含相機曝光、DDS callback 前的傳輸、網路與瀏覽器顯示時間**。
上傳静態圖片時該欄位為 null；FPS 是最近兩筆結果的產出速率，停流時不代表仍有更新。

重啟 `./dinov3_ws/run.sh` 並重新整理瀏覽器即可使用新流程。可透過
`./dinov3_ws/run.sh --ros-args -p inference_hz:=0.0 -p jpeg_quality:=80` 指定參數。
實際上限仍受 Grounding DINO、逐 ROI DINOv3 運算、JPEG 編碼及顯示網路限制；需在真實相機與目標物數量下量測。

TensorRT 真實 GPU 數值比對（不同提示詞、直／橫式影像、DINOv3 動態尺寸）：

```bash
source /opt/ros/jazzy/setup.bash
source ../dinov3_venv/bin/activate
source install/setup.bash
python tests/check_grounding_trt.py
python tests/check_grounded_trt_pipeline.py
```

數值報告輸出至 `artifacts/tensorrt/verification_grounding_community_vits16.json`，實際雙 TRT ROS／HTTP 流程報告為 `integration_grounding_community_vits16.json`。數值比對使用 ONNX Runtime 執行原始社群 ONNX 與修補版本，檢查兩者輸出相等，再與 TensorRT 最終接受的框、分數、標籤比對。使用一張參考圖、不同提示詞／方向。Grounding DINO 的 two-stage top-k 對數值擾動敏感，低信心候選 query 不保證逐項一致，原始 tensor 最大差異仍保留於報告。這些檢查不代表完整商品偵測／辨識準確率評估。

## Wingdzero TRT 驗證

相關單元測試 22 項通過，既有 ROS/HTTP 回歸驗證通過。`tests/check_grounding_wingdzero_trt.py` 用實際藥品圖片比對完整提示詞、較短補齊提示詞與旋轉圖片；NMS 後偵測框配對 IoU 最小約 0.9989，信心分數最大誤差約 0.00265。`tests/check_grounded_trt_pipeline.py` 已驗證新 Grounding DINO TRT → DINOv3 ViT-S/16 TRT → 圖庫辨識、物件預覽、較短提示詞更新與超長提示詞拒絕。結果存於 `artifacts/tensorrt/verification_grounding_wingdzero.json` 和 `integration_grounding_wingdzero_vits16.json`。此檢查不是完整資料集精度評估。
