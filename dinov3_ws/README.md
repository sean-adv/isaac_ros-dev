# Grounding DINO → DINOv3 物體辨識

預設流程已改為：**RGB + 文字提示 → Grounding DINO 物體框 → 每框裁切 → DINOv3 特徵 → 所有商品參考圖庫 → 商品名稱或 unknown**。

## 啟動

```bash
cd /workspaces/isaac_ros-dev/dinov3_ws
./run_camera.sh
# 另一個終端；沿用現有 DINOv3 TensorRT engine
./run.sh --model vitb16
# 或使用原本預設的 ViT-H+
./run.sh
```

首次啟動會從 Hugging Face 下載公開的 `IDEA-Research/grounding-dino-tiny` 至 `dinov3_ws/artifacts/grounding_dino`。Grounding DINO 使用 FP32 PyTorch 與原生算子，DINOv3 使用既有 TensorRT／PyTorch 後端；無需安裝 GroundingDINO 原始碼或編譯自訂 CUDA extension。模型載入失敗會在網頁顯示原因。可用 `grounding_model_id` 指定本機模型目錄，並以 `grounding_cache_dir` 覆寫快取位置。

開啟 <http://localhost:8090>：

1. 每個 case 放同一商品的多角度裁切照片，例如 `medicine_A`、`medicine_B`。所有非空 case 都參與辨識；「檢視此圖庫」僅選擇預覽圖片。
2. 在文字提示輸入 `medicine bottle. medicine box. bottle. box.`，按「套用設定」。英文類別用句點分隔；提示詞不得超過模型的 256 token 上限。
3. 使用 ROS 相機或上傳目標圖。每個框顯示商品名稱、偵測文字、`det` 偵測分數及 `sim` 圖庫相似度。
4. 調整偵測門檻、文字門檻、辨識相似度與前兩名最小差距。設定立即生效並重新處理靜態目標，僅保存至當前程序；永久設定請修改 `config/dinov3_cases.yaml`。

## 辨識方式與限制

Grounding DINO 的框會映射回原始 RGB 座標、裁齊邊界，經跨文字類別的 NMS 移除重複框。NMS 保留較高分的框，不取外接聯集；最多辨識 `grounding_max_detections` 個框。

參考圖片與每個 ROI 使用相同的等比例縮放：長邊設為 `object_max_side`，短邊不足 32 pixels 或 feature stride 整數倍時複製邊緣補齊。DINOv3 產生正規化 patch 特徵，取平均後再次 L2 正規化，作為整個物體的描述向量。每張參考圖保留一個向量，各 case 取其最高 cosine similarity；以最高分與不同 case 的第二高分之差進行接受判定。這是無需訓練的初版視覺檢索，門檻需以真實商品驗證集校準。

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
| `grounding_prompt` | `medicine bottle. medicine box. bottle. box.` | 文字提示 |
| `grounding_box_threshold` | 0.30 | 候選框分數門檻 |
| `grounding_text_threshold` | 0.25 | 文字片語門檻 |
| `grounding_nms_iou` | 0.50 | 重複框抑制 IoU |
| `grounding_max_detections` | 20 | 每幀最大 ROI 數量 |
| `object_max_side` | 224 | 參考圖與 ROI 的共同長邊尺寸 |
| `recognition_threshold` | 0.75 | 接受商品身分的 cosine 門檻 |
| `recognition_margin` | 0.05 | 不同商品前兩名最小分差 |

`reference_max_side`、`target_padding`、`cosine_threshold`、形態學與框合併參數僅影響舊 heatmap 流程。新流程的參考圖網格預覽依 `object_max_side` 顯示。

## API 與 ROS

`POST /api/grounding` 接受部分或全部設定：

```json
{"prompt":"medicine bottle. medicine box.","box_threshold":0.3,"text_threshold":0.25,"recognition_threshold":0.75,"margin":0.05}
```

`GET /api/status` 增加 `pipeline`、`grounding`、`grounding_model_id`、`object_max_side`。
`GET /api/detections` 的每個物體包含 `bbox`（原圖半開 XYXY）、`area`（矩形面積）、`label`、`candidate`（最接近商品）、`score`（DINOv3 相似度或 null）、`margin`、`reason`、`detection_label`、`detection_score`、`pose`、`pose_error`。
有可用參考圖時另含 `reference`（case、name）。`GET /api/object_visualizations` 回傳逐物體尺寸、視圖清單與 `revision`；影像使用 `/debug/objects/<revision>/<index>/<view>.jpg`。兩個端點皆可加上 `?reference_case=<標籤>`，未指定則使用自動模式。manifest 另回傳 `reference_cases`、`reference_case`、`ready`；物體資料另含 `comparison_case`、`comparison_score`、`comparison_reference`、`comparison_reference_count`。不存在或無圖片的標籤回傳 400。

快取保留最近三幀的 ROI 特徵與當時的參考圖庫，標籤比較在請求時產生；每幀最多保存自動模式及兩個指定標籤的影像。設定／來源／圖庫變更會清空；已失效 revision 回傳 410，頁面會重新取得最新結果。切換標籤時會清除舊標籤卡片，快速切換所產生的過期請求不會覆蓋新的選擇。

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
