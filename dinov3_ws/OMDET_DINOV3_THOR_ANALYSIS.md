# OMDet／DINOv3 推論延遲與 FP16／FP32 精度分析：Jetson Thor

> 歷史量測報告：下文「目前」指分析當時的 OmDet／ViT-S 部署。現行 SAM3／ViT-B 的使用與建置請見 [BUILD_GUIDE.md](BUILD_GUIDE.md)；舊 OmDet 程式已移除。

分析日期：2026-10-12（UTC）。分析對象是目前工作區部署的引擎；補測使用現有模型，未重建引擎、變更精度或調整電源／時脈。

## 1. 結論

- **目前偵測階段的主要成本在 OMDet**：混合 FP16 完整偵測中位數 **75.95 ms**，FP32 為 **111.45 ms**，本次加速 **1.47×**、延遲降低 **31.8%**。GPU 常駐輸入的 TensorRT 執行時間分別為 **30.48／53.78 ms**，加速 **1.76×**。
- **DINOv3 ViT-S/16 本體較快**：FP32 I/O 的 GPU 常駐 adapter 中位數 **1.26 ms**；已準備成 224×224 的單一 ROI 完整編碼 **1.84 ms**。本次高解析度商品框若包含 CPU 裁切與縮放，則為 **26.76 ms／ROI**；因此只報模型執行時間會低估辨識流程成本。
- **OMDet 的「FP16」是 FP16／FP32 混合精度**：視覺端的低精度權重、FP32 解碼器、FP32 敏感運算及累加邊界共存。不能解讀為全模型 FP16。
- **DINOv3 已確認 FP16 部署設定、FP32 I/O 與 FP32 特徵後處理**；現有引擎未保存逐層詳細型別，無法完整核實各 kernel 的權重、activation、累加精度及占比。
- **Thor 的 2070 TFLOPS 是稀疏 FP4 峰值**。目前模型未部署 FP4，也沒有已驗證的 2:4 稀疏性；評估 FP16 運算時應參照約 **258.5 TFLOPS dense FP16 的推導峰值**，而且峰值無法直接換算端到端 FPS。

## 2. 目前部署與量測条件

| 項目 | OMDet | DINOv3 |
| --- | --- | --- |
| 模型 | `omlab/OmDet-Turbo_tiny_SWIN_T` | `facebook/dinov3-vits16-pretrain-lvd1689m` |
| 用途 | 全圖物體框偵測 | 每個 ROI 的 patch 特徵與圖庫辨識 |
| 預設引擎 | `artifacts/omdet/omdet_tiny.engine` | `artifacts/tensorrt11/dinov3_vits16_fp16_224.engine` |
| 計算設定 | ModelOpt AutoCast 混合 FP16／FP32；TF32 關閉 | FP16 TRTMC；FP32 I/O；TF32 建置狀態未記錄 |
| 影像輸入 | batch=1，3×640×640，直接 resize | batch=1，3×224×224，等比例縮放並補邊 |
| 輸出 | boxes、logits 各 `1×1×900×4`，4 類 | hidden state `1×201×384`；pooler `1×384` |
| 實際辨識特徵 | 偵測框後裁切 ROI | 去掉 CLS＋4 register tokens，得到 `196×384` patches |
| TensorRT 融合後 layer 數 | 混合 FP16：443；FP32：478 | 102 |
| 引擎大小 | 混合 FP16：約 201.17 MiB；FP32：約 255.47 MiB | 約 42.75 MiB |

引擎大小包含序列化權重及其他資料，不等於執行時記憶體用量；融合 layer 數也不等於原始模型層數。

本機 `/proc/device-tree/model` 是 **NVIDIA Jetson AGX Thor Developer Kit**，`nvidia-smi` 顯示 NVIDIA Thor、driver **595.78**、driver CUDA 相容版本 **13.2**，MIG 關閉。補測環境為 TensorRT **11.1.0.106**、PyTorch **2.14.1+cu130**、PyTorch CUDA **13.0**；driver 支援版本與 PyTorch runtime 版本是不同欄位。

測試圖片：`dinov3_cases_data/stomach/1790733215386907348_2.png`，**1424×2428**，偵測出一個商品框；門檻 0.3、NMS IoU 0.5、最多 20 個框。每項測試暖機 **10 次**、量測 **60 次**，PyTorch CPU threads=8，採單程序、依序執行。圖片磁碟解碼、模型載入均在量測外；wall-clock 項目於每次樣本前後同步 CUDA。

本次未鎖定 GPU／CPU 時脈；環境找不到 `nvpmodel`，未確認電源模式，也未取得逐樣本時脈、溫度與功率。這是**目前系統狀態下的暖機延遲**，不是 MAXN 或固定時脈下的硬體上限。桌面程序仍在運行。

## 3. 推論延遲對照

### 3.1 本次 OMDet 補測

| 量測邊界 | 混合 FP16 中位數 ms | 混合 FP16 P95 ms | FP32 中位數 ms | FP32 P95 ms |
| --- | ---: | ---: | ---: | ---: |
| CPU 影像前處理：RGB、resize、FP32 正規化、NCHW | 16.62 | 17.31 | 16.82 | 17.46 |
| GPU 常駐輸入 `infer()` wall time | 31.63 | 57.10 | 56.84 | 82.65 |
| 已绑定 buffer 的 TensorRT CUDA-event 執行 | 30.48 | 53.29 | 53.78 | 81.27 |
| FP32 後處理：sigmoid、top-k、框還原、NMS | 0.93 | 1.39 | 1.56 | 3.73 |
| 完整 `detect()`：前處理＋CPU 輸入 copy＋推論＋後處理 | **75.95** | **96.81** | **111.45** | **131.16** |

CUDA-event 項目只在 TensorRT `execute_async_v3()` 前後記錄事件，不含 CPU 前處理及前置輸入 copy。`infer()` 項目仍包含 GPU buffer copy、binding、同步與輸出 clone；兩者均使用 GPU 常駐影像輸入。完整 `detect()` 則遵循目前程式，將 CPU 影像張量複製到 GPU。

各列是**獨立測試的統計值**，不能將中位數直接相加，再把差值全部歸因於傳輸。完整偵測與 GPU 常駐測試的記憶體路徑、CPU 工作、時脈狀態不同；目前沒有在同一影格逐段量測 copy 成本，尚不能定量判定其占比。

以完整偵測中位數的倒數作為單工作串行速率參考：混合 FP16 約 **13.17 次／秒**，FP32 約 **8.97 次／秒**。這是延遲倒數，**不是實測相機結果 FPS**。

### 3.2 本次 DINOv3 補測：每個 ROI

| 量測邊界 | 中位數 ms | P95 ms | 包含內容 |
| --- | ---: | ---: | --- |
| GPU 常駐 `TRTBackbone.__call__()` wall time | **1.26** | 3.31 | binding、輸出配置、TRT、同步 |
| Adapter 外包 CUDA events | 1.14 | 1.61 | 包含 adapter 造成的 stream 間隔；不是純 TRT kernel 時間 |
| 已準備 224×224 ROI 的 `Encoder.encode()` | **1.84** | 6.35 | NumPy→張量、GPU copy、FP32 正規化、TRT、patch L2 正規化與有效值檢查 |
| 高解析度框裁切＋`prepare_object()`＋編碼 | **26.76** | **34.06** | CPU 裁切、等比例縮放、補邊及上述編碼 |

本次框約 **1360×2411**，接近整張高解析度商品圖；26.76 ms 不宜套用到所有相機 ROI。小框的裁切與 resize 成本通常不同。上述編碼均**不含圖庫相似度比對、追蹤、深度姿態、疊圖、JPEG、ROS／HTTP 發布**。

相機 profiler 中的 `dinov3_trt_ms` 是該影格**所有 ROI adapter 呼叫的累加**，不能當成單個 ROI 延遲。先前 `camera_single.json` 的 4.05 ms 對應中位數約 3 個物件，且整個報告使用 Grounding DINO；它只能作為歷史場景，不能代表目前 OMDet 串接流程。

### 3.3 既有實測与本次量測的關係

| 證據 | FP32 完整偵測中位數 | 混合 FP16 完整偵測中位數 | 樣本／性質 |
| --- | ---: | ---: | --- |
| `artifacts/omdet/validation_fp16.json` | 92.26 ms | 72.77 ms | 暖機 5 次、20 次樣本；加速 1.27× |
| 本次 `current_precision_latency/report.json` | 111.45 ms | 75.95 ms | 暖機 10 次、60 次樣本；加速 1.47× |

兩次使用同一圖片及相同 OMDet 混合 FP16 引擎 SHA256；量測時系統負載、同步方式與電源／時脈條件未被完整固定，不能把差異視為模型退化。兩份報告都支持混合 FP16 比 FP32 快，但不支持固定加速倍率。

目前 OMDet `integration_single_224.json`／`integration_split_224.json` 是上傳圖片的功能整合快照；single 處理約 0.461 秒、split 約 1.87 秒，沒有穩態暖機與多樣本統計，因此不拿它們當作暖機推論延遲。`artifacts/performance/camera_*.json` 的 status 明確標示 Grounding DINO；`grounding_*_stages.json` 缺少足以核對目前引擎的 hash／版本證據，也不併入本次 OMDet 結論。

## 4. 模型內部 FP16／FP32 精度

此處的「量化資訊」主要是**浮點降精度／混合精度**。FP16 轉換與 INT8／FP8／FP4 的 scale、校準及 Q/DQ 量化不是同一件事。OMDet AutoCast 使用實圖決定低精度邊界，不代表建立了 INT8 量化引擎。

### 4.1 OMDet：由 metadata、建置程式與實際 ONNX 交叉核實

| 模組／資料 | 精度 | 可確認的依據 |
| --- | --- | --- |
| 原始 ONNX 浮點 initializer | FP32 | 493 個，51,619,334 個元素 |
| 混合模型視覺 backbone／neck | FP16 與 FP32 共存 | 實際 ONNX 的兩個模組均有 FLOAT16／FLOAT 張量；非全部轉 FP16 |
| Conv／MatMul／Gemm 的低精度路徑 | FP16 儲存／邊界，顯式 FP32 運算邊界，回轉 FP16 | `enforce_fp32_accumulation()`；123 個改寫節點 |
| 偵測 decoder | FP32 浮點張量 | `^/decoder/.*` 排除轉換；實際 decoder 浮點輸出記錄為 FLOAT，沒有 FLOAT16 |
| LayerNormalization | FP32 | 53 個 ONNX 節點，排除 AutoCast |
| Softmax | FP32 | 25 個 ONNX 節點，排除 AutoCast |
| ReduceSum／ReduceMax | FP32 | 7／1 個 ONNX 節點，排除 AutoCast |
| ReduceMean | 列入 FP32 保護規則 | 目前 ONNX 中無獨立 ReduceMean 節點，不能聲稱有實際節點占比 |
| 影像、label/task features、boxes、logits I/O | FP32 | 實際 engine binding 為 FLOAT |
| task_mask | INT32 | `1×77`；是遮罩型別，非 INT8 量化 |
| 執行時 CLIP 文字編碼 | 不執行 | 已离線編碼，載入 FP32 `prompts.npz` |
| sigmoid、框座標還原、NMS 前浮點資料 | FP32 | `postprocess()` 的 `.float()` 與 FP32 engine 輸出 |
| TF32 | 關閉 | 兩個 OMDet metadata 及 build 的 `clear_flag(TF32)` |
| INT8／FP8／FP4 Q/DQ | 未部署於此 ONNX | QuantizeLinear／DequantizeLinear 節點均為 0；浮點 initializer 僅 FLOAT／FLOAT16 |

混合 ONNX 的 initializer 統計：

| 型別 | initializer 數量 | 元素數 | 浮點元素比例 | 原始 initializer 資料大小 |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 213 | 40,359,914 | 約 77.19% | 約 76.98 MiB |
| FP32 | 300 | 11,928,860 | 約 22.81% | 約 45.50 MiB |
| 合計 | 513 | 52,288,774 | 100% | 約 122.49 MiB |

initializer 包含權重與常數，轉換後亦可能新增／複製資料；因此不是唯一模型參數量。元素比例也**不是 layer 比例、FLOPs 比例或耗時比例**。混合模型 Cast 目標包含 181 個 FLOAT16、362 個 FLOAT，這是 ONNX 圖節點統計；TensorRT 可融合，不能視為 543 個獨立執行的 Cast kernel。

FP32 cast-around-GEMM 是 NVIDIA 文件支持的累加控制方式：TensorRT 可將轉換與 GEMM 融合為低精度輸入、FP32 累加。但實際 Conv／GEMM tactic 是否如此融合，不能只由 ONNX 的 Cast 數證明；本次未取得逐 kernel 的累加精度。[NVIDIA TensorRT 精度控制文件](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/precision-control.html)

### 4.2 DINOv3：已確認項目與證據限制

| 模組／資料 | 目前可確認資訊 | 證據程度 |
| --- | --- | --- |
| 部署精度 | `FP16 TRTMC (FP32 I/O)` | 部署 metadata；未保存完整 build 版本／逐層型別 |
| pixel_values、last_hidden_state、pooler_output | FP32 | 實際 engine binding 驗證 |
| RGB 正規化 | FP32 | `patches.preprocess_image()` |
| Patch embedding、QKV／output projection、MLP | 目前 TRTMC FP16 建置路徑採 FP16 工作 dtype／權重 | 轉換器原始碼依據；不是部署引擎逐 kernel 證據 |
| RoPE | 目前原始碼先在 CPU 以 FP32 計算三角常數，按工作 dtype 入圖 | 同上，不能認定部署 kernel 全程 FP32 |
| LayerNorm | 可見融合 layer 名稱有 Cast 與正規化計算；精確 I/O／累加型別未保存 | 無法由名稱定量判定；不可照搬 OMDet 的 FP32 保護規則 |
| Attention Softmax、QK／PV 累加 | 目前轉換器採工作 dtype 路徑；實際融合與累加精度未知 | inspector 可見 `_gemm_mha_v2`，未提供 datatype／tactic 詳情 |
| Patch 特徵 L2 正規化 | FP32 | `patch_features()` 的 `tokens.float()` |
| 圖庫特徵與相似度資料 | FP32 張量 | 特徵輸出、`compute_similarity_maps()`；實際 matmul 是否用 TF32 需查 runtime 設定 |
| INT8／FP8／FP4 | 沒有目前部署或驗證證據 | 現有 metadata 標示 FP16；未作低位元 kernel 全面鑑識 |

參考轉換器：[DINOv3 builder](../TensorRT-Model-Connect/families/dinov3/model.py)、[graph helpers](../TensorRT-Model-Connect/families/dinov3/graph_ops.py)。這是**目前工作區的建置策略**；metadata 未記錄轉換器 commit，故無法保證與舊部署引擎逐項相同。

三個 engine 的 inspector 都只有 `ProfilingVerbosity.LAYER_NAMES_ONLY`。本次已保存 layer 名稱，但不能從現有 engine 還原完整逐層 datatype、權重型別、FP16／FP32 kernel 耗時占比。若要完成這一層的稽核，須另建 `DETAILED` profiling 引擎並保存原始碼版本、build log／tactics，再配合逐層 profiling；本次未為分析重建部署模型。

### 4.3 已有精度驗證

- OMDet：四張測試影像（含一張旋轉圖），FP16／FP32 有效偵測類別及數量一致；最小框 IoU 約 **0.99937**，最大分數差 **0.02243**。設定驗收界線為 IoU≥0.98、分數差≤0.03；測試包含未用於 AutoCast 的影像。這是小樣本回歸，沒有完整資料集 mAP 結果。[既有驗證](artifacts/omdet/validation_fp16.json)
- DINOv3：224×224 測試圖的 patch 特徵對 PyTorch 參考結果，平均 cosine **0.99999046**、最小 **0.99985886**。這是特徵一致性驗證，沒有完整商品辨識準確率資料。[既有驗證](artifacts/tensorrt11/dinov3-adapter-validation.json)

## 5. 對照 Jetson Thor T5000 運算能力

AGX Thor Developer Kit 使用 T5000。本機已確認 Developer Kit 型號；官方規格為 Blackwell GPU、2560 CUDA cores、第五代 Tensor Cores、128 GB LPDDR5X、**273 GB/s** 記憶體頻寬、最高 GPU **1.57 GHz**、14 核 CPU，模組功率範圍 **40–130 W**。這些是平台規格，不是本次實際運行功率與時脈。[NVIDIA 官方 Thor 規格](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/)

| 計算格式 | T5000 峰值／依據 | 對目前模型的意義 |
| --- | ---: | --- |
| FP4 sparse | 2070 TFLOPS，官方公布 | 未部署 FP4；不能拿來換算目前延遲 |
| FP4 dense | 1035 TFLOPS，官方公布 | 需轉換與重新驗證模型 |
| FP8 sparse | 1035 TFLOPS，官方公布 | 目前無 FP8 或符合稀疏条件的證據 |
| FP8 dense | 517 TFLOPS，官方公布 | 有潛在加速空間，需校準與正確性驗證 |
| FP16 sparse | 517 TFLOPS，官方公布 | 現有模型未證實支援的結構化稀疏性 |
| **FP16 dense** | **約 258.5 TFLOPS，依 sparse÷2 推導** | 與目前低精度 Conv／GEMM 較相符，但不是模型達成的算力 |
| 嚴格 FP32／TF32 | 本次可取得的官方產品表未列出對應峰值 | OMDet 關閉 TF32；不得用 FP16 或 FP4 峰值當成其 FP32 算力 |

各精度官方峰值來源為 [NVIDIA Thor 技術文章的 T5000 表格](https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/)。dense FP16 一列是推導值，假設同條件下支援的結構化稀疏運算相對 dense 為 2×；不是這兩個模型的實測吞吐。

FP16 operands 配合 FP32 accumulation 可運用 Tensor Cores，**不能把 FP32 accumulation 一律當成全 FP32 CUDA-core 路徑**；相反地，FP32 decoder、正規化、資料搬移及小算子也不會自動取得 FP16 GEMM 的峰值。

合理的延遲下界模型為：

```text
GPU 單一階段延遲 ≥ max(該階段 FLOPs / 對應精度有效算力,
                       該階段 DRAM 傳輸量 / 有效頻寬)
端到端延遲另加 CPU、copy、同步、排程及輸出處理
```

這裡需要分階段或逐 kernel 的 FLOPs、精度、資料流量與有效算力；目前沒有完整計算圖成本及 profiler counters，**無法計算 GPU 算力利用率、精確 memory-bound／compute-bound 比例，或用峰值推算可靠 FPS**。引擎檔案大小也不是每次推論的 DRAM 流量。

## 6. 流程容量與優化優先順序

```text
全圖 RGB → OMDet 前處理 → 混合精度 TensorRT → FP32 解碼／NMS
         → N 個 ROI 裁切／縮放 → DINOv3（目前 batch=1，逐 ROI）
         → FP32 patch 正規化／圖庫比對 → 追蹤／姿態／疊圖／發布
```

對本次高解析度輸入，粗略規劃可用 `75.95 + N×26.76 ms` 作為**偵測＋裁切編碼**的中位數相加估算；N=1 時約 **102.71 ms**，約 **9.74 次／秒**的延遲倒數。它不是串接實測，也未包含圖庫比對、深度、疊圖、發布，且不同 ROI 尺寸不可沿用同一係數。若 ROI 已準備完成，編碼係數約 1.84 ms，但 CPU 影像準備成本仍須另計。

30 FPS 的每幀預算是 **33.33 ms**；即使 OMDet GPU-event 中位數約 30.48 ms，完整偵測 75.95 ms 及 P95 96.81 ms 已超出預算。現有流程尚沒有每幀 30 FPS 的證據，使用最新影格避免堆積也不等於每幀均完成辨識。

1. **先查核前處理、CPU→GPU copy 與同步成本**：維持官方 resize／正規化語意，使用同影格逐階段量測，再評估 pinned memory、buffer 重用或 GPU 前處理。不能把舊 Grounding DINO 的優化幅度套到 OMDet。
2. **減少高解析度 ROI 裁切／縮放成本**：本次 26.76 ms 與已準備 ROI 的 1.84 ms 差距大，顯示影像準備值得優先分析；需以實際相機框大小再量測。
3. **評估明確非預設 CUDA stream**：本次 TensorRT 提示 default stream 可能帶来額外同步；兩個 adapter 目前亦同步後才返回。修改須配合 buffer 生命週期，重新驗證一致性與延遲。
4. **取得完整 engine 精度／耗時證據後，再調整 OMDet decoder**：目前 FP32 decoder 可合理解釋為潛在瓶頸，但沒有逐層耗時，尚不能稱為最大耗時模組。保留框 IoU／類別／分數驗證門檻。
5. **多 ROI 時評估 batch 引擎及圖庫比對**：目前固定 batch=1，不能只改 batch 數；需重建相應 shape 的引擎，並量測整體辨識，不只模型。
6. **FP8／FP4 與結構化稀疏性列為後續實驗**：Thor 支援不代表模型已使用；須有精度策略、校準資料、商品辨識結果、逐層執行與端到端速度證據。

## 7. 證據與重現

本次新增的完整樣本、引擎 hash、I/O、ONNX 統計與 layer 名稱位於：

- [補測 JSON](artifacts/performance/current_precision_latency/report.json)
- [OMDet 混合精度 layer 名稱](artifacts/performance/current_precision_latency/omdet_tiny_layers.json)
- [OMDet FP32 layer 名稱](artifacts/performance/current_precision_latency/omdet_tiny_fp32_layers.json)
- [DINOv3 layer 名稱](artifacts/performance/current_precision_latency/dinov3_vits16_fp16_224_layers.json)
- [OMDet metadata](artifacts/omdet/omdet_tiny.json)、[DINOv3 metadata](artifacts/tensorrt11/dinov3_vits16_fp16_224.json)
- [OMDet 建置程式](build_omdet.py)、[OMDet adapter](src/dinov3_cases/dinov3_cases/omdet_trt.py)、[DINOv3 adapter](src/dinov3_cases/dinov3_cases/trt_backend.py)

此為舊 OmDet 部署的歷史量測；對應的一次性量測腳本已移除，現有 JSON 報告仍保留。當前 SAM3 的測試與量測入口見 [測試說明](tests/README.md)。

當時的量測腳本驗證了 OMDet engine／prompt hash、固定 I/O 與輸出有限值；沒有重新執行完整資料集準確率測試，也沒有重測目前 OMDet 相機端到端流程。DINOv3 的 FP32 引擎基準未部署於目前預設路徑，本次沒有虛構其 FP32 延遲或 FP16 加速倍率。
