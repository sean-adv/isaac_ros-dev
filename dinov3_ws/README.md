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

## 官方匯出與離線提示

來源是 [OmDet 官方儲存庫](https://github.com/om-ai-lab/OmDet)，沿用官方
`OmDetV2TurboInfer`、ONNX opset 17 與四個固定尺寸輸入：

| 輸入 | 尺寸 | 意義 |
| --- | --- | --- |
| img_tensor | 1×3×640×640 | RGB，直接 resize，均值／標準差正規化 |
| label_feats | C×1×512 | CLIP 的類別 EOT 特徵，含 text_projection |
| task_feats | 77×1×512 | 任務句的 token 特徵，補齊至 77 |
| task_mask | 1×77 | 有效 token 為 1，padding 為 0 |

類別名稱與任務句分開編碼；使用官方修改版 CLIP，並載入 OmDet checkpoint 中的
`language_backbone` 權重。**不要使用其他 CLIP 的一般 pooled embedding 代替任務 token 特徵。**
ONNX 不含文字編碼與後處理。TensorRT 11 引擎預設使用 FP16（NVIDIA ModelOpt AutoCast，FP32 I/O）。
偵測解碼器、LayerNormalization、Softmax、ReduceMean／Sum／Max 及校驗判定的敏感節點保留 FP32，
卷積與矩陣乘法採 FP16 張量、明確 FP32 累加，再轉回 FP16，避免 GPU 低精度累加造成分數偏差；
後處理對類別 logits 套用 sigmoid，
將 normalized cxcywh 還原為原圖 XYXY，再做 NMS 與重複物體抑制。

`config/omdet_prompts.json` 定義離線 profiles：`prompt` 是網頁／ROS 的選擇鍵；
`labels` 是有序類別列表；`task` 是獨立任務句。預設包含 medicine bottle、medicine box、bottle、box。
執行時只載入 `prompts.npz`，不載入 tokenizer、CLIP、PyTorch 偵測器或下載任何資源。
未建置的提示回傳錯誤；OmDet 不使用 Grounding DINO 的 token 文字門檻。
舊 `grounding_*` 參數、HTTP 路徑及 IPC 名稱保留相容，`grounding_text_threshold` 不影響 OmDet 分數。
網頁只提供已建置提示的選單。

新增提示時編輯 JSON，再重建。此版本採固定輸入尺寸，所有 profile 的類別數須相同；
若類別數改變，必須一起重建 ONNX 和 engine。類別或任務句超過 CLIP 77 tokens 會拒絕，不靜默截斷。

```bash
# 本工作區已下載官方原始碼及權重；新工作區先 clone 官方 repo 至 OmDet。
./dinov3_ws/build_omdet.sh --download    # 官方權重 → 離線提示 → ONNX → TensorRT
# 完全離線重建（使用已有 resources）
./dinov3_ws/build_omdet.sh
# 只從已有 ONNX 建置 engine
./dinov3_ws/build_omdet.sh --stage engine --precision fp16
# 建置獨立 FP32 基準（omdet_tiny_fp32.engine）：
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
不是感測器精度量測。沒有追蹤器，3D detection 的持續 ID 留空。

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

舊 Grounding DINO 重建腳本與模型產物保留作歷史資料；目前啟動入口使用 OmDet。
