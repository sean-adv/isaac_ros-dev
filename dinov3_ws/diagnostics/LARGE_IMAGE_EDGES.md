# 大圖外框高 cosine：位置相關特徵的控制實驗

本次使用本機 DINOv3 ViT-H/16+、六張 punateng reference、長邊 640、每圖均勻抽樣且總上限 8192，與目前正式流程一致。未改動正式 inference 或使用者原圖。新增兩個離線診斷腳本。

**已證實：在目前模型與 reference bank 下，即使 target 是完全均勻的灰圖，也會產生外圈較高的 raw patch cosine。這個空間圖樣可由 RoPE 位置排列控制，並非必須存在背景混合 patch 才會發生。**

**尚未證實：使用者實際大圖的主要原因與各因素貢獻。工作區沒有已確認的原始 camera target，不能把合成控制圖當成實際大圖的復現。**

## 目前程式路徑

- `patches.preprocess_image`：只正規化，不 resize、切圖或 padding。
- `patches.patch_features`：排除 CLS + 4 registers，對最後一層特徵做 L2 normalize。
- `node.run_worker`：reference 縮到設定長邊，target 保留原解析度；建立 bank 後做 max cosine matching。
- `patches.similarity_pixels`：bilinear 插值，未覆蓋的不足一個 patch 的底／右條帶填 -1。
- `node.render_heatmap`：固定 0–1 色階，沒有每張圖 min-max 拉伸。

因此目前不存在切圖接縫或 padding 引入的外框。控制實驗直接計算原始 patch 分數，跳過插值與熱圖顯示。

## 1. 無物體、無 RGB 邊界，也會高分

target 包含純灰、純黑（640×384、1280×768、1920×1152）與固定種子的 1280×768 隨機噪聲。以下為 TensorRT 原始 patch 平均 cosine。外圈為最外一圈；內部排除外三圈。

| target | 全 bank 外圈 | 全 bank 內部 | ≥95% alpha 前景 bank 外圈 | 前景 bank 內部 |
|---|---:|---:|---:|---:|
| 灰 640 | 0.6577 | 0.4961 | 0.6359 | 0.4961 |
| 灰 1280 | 0.6674 | 0.4920 | 0.6314 | 0.4918 |
| 灰 1920 | 0.6707 | 0.4843 | 0.6252 | 0.4833 |
| 黑 1280 | 0.6796 | 0.5077 | 0.6356 | 0.5072 |
| 噪聲 1280 | 0.7030 | 0.5362 | 0.6882 | 0.5361 |

這些控制圖沒有藥盒。即使 reference 只保留前景，仍出現外圈／內部差異。不能只根據高 cosine 宣稱語意辨識成功。

灰 1280 的外圈最佳 reference 匹配有 **96.8%** 落在 reference 外側兩圈。再從前景 bank 移除 reference 外兩圈，外圈平均降到 0.5742，但內部仍為 0.4897，差異未消失。此額外刪除也會改變 bank 大小，因此不能將下降全歸因於邊界位置；它只是匹配來源診斷，不是已驗證的正式修復。

解析度增加時，全 bank 的外圈／內部差距由 0.1616 增至 0.1864；640 尺度已存在，不能聲稱只有大圖才會發生。

## 2. TensorRT 對照

七個 target 均以原生 PyTorch BF16 / SDPA 重新建立 reference bank 並重新推論。

- 外圈 TRT／PyTorch 對應 feature cosine 均值約 0.99973–0.99984。
- 全 bank 外圈分數平均絕對差約 0.00153–0.00195。
- 兩個後端均有明顯外圈高分。

因此這些控制案例的主要現象不能歸因於 TRT 匯出或算術誤差。本次沒有聲稱所有真實輸入均已驗證，也未比較 FP32 或 Meta 原始實作。

## 3. 位置編碼因果對照

使用 PyTorch、1280×768 純灰圖、固定且未改變的 reference bank：

1. 正常 RoPE：全 bank 外圈 0.66747、內部 0.49282；前景 bank 外圈 0.63344、內部 0.49266。
2. 將 target 的 RoPE cos/sin 在 patch 網格平移半個高度與寬度（24、40 patches），像素完全不變。結果高分圖樣跟著位置排列移動：與原圖平移後相比，全 bank 相關係數 **0.999976**、平均絕對差 **0.000552**；前景 bank 相關係數 **0.999963**。
3. 只在 target 端將 RoPE 改為 cos=1、sin=0，保持 reference 特徵不變。分數在整張灰圖幾乎相同：**0.30235592–0.30235618**。

這是目前模型中「位置相關特徵 → 空间分數圖樣」的直接證據。位置排列平移是座標的重排，不是正常相機移動；純色輸入使內容完全相同，因而可以隔離位置影響。它不量化真實場景的語意誤匹配比例。

RoPE 經 attention 影響最終 embedding。alpha ≥95% 只保證該 patch 的像素大多為前景，不會移除 embedding 的位置／上下文成分。因此前景 bank 仍能提供位置相關匹配，max matching 再從所有候選中選出高分。

**停用 RoPE 不是修復建議**：這是偏離訓練設定的診斷介入，真實物體辨識效能尚未測量。

參考：[Meta 官方 RoPE 實作](https://github.com/facebookresearch/dinov3/blob/main/dinov3/layers/rope_position_encoding.py)。本次實際執行的是本機 Transformers 實作；它以影像網格大小計算標準化 patch 座標。本報告的因果結論來自本機實驗，而非引用 issue 推斷。

## 4. Padding 控制的限制

把同一片灰色內容放到四周多 64 pixels 的同色畫布中，原本外圈位置的平均分數從 0.66747 降到 0.59029，但新畫布外圈仍是 0.66747。前景 bank 的原圈則從 0.63344 降到 0.58928。

這支持邊界位置的作用，但不代表 padding 能消除問題；它同時改變 token 數與標準化座標，且這裡只測均勻灰圖。

## 下一個必要驗證

取得使用者出問題的原始大圖及實際 reference 分類／模型，用同一 bank 比較 native、長邊 640、patch 對齊原圖及 reflect／edge padding 後裁回原座標。腳本已支援這些對照。需同時量測外框高分和真實物體分數，才能決定 padding、reference 篩選或匹配分數校正是否適合正式使用。尚未驗證的方案不加入服務。

## 重現與產物

在工作區根目錄：

```bash
PYTHONDONTWRITEBYTECODE=1 dinov3_venv/bin/python dinov3_ws/diagnostics/large_image_edges.py
PYTHONDONTWRITEBYTECODE=1 dinov3_venv/bin/python dinov3_ws/diagnostics/position_ablation.py
```

實際原圖（原圖加 128 後須在 TRT 2560×2560 profile 內）：

```bash
PYTHONDONTWRITEBYTECODE=1 dinov3_venv/bin/python dinov3_ws/diagnostics/large_image_edges.py \
  --target /absolute/path/target.png --out /tmp/dino_actual_edges
```

- `large_edge_results/report.json`：七個 target、兩個 backend、三組 bank 的統計與匹配來源。
- `large_edge_results/*.npy`：原始 patch 分數。
- `position_results/report.json`：位置消融、同色 padding 與平移誤差。
- `position_results/position_ablation.png`：前景 bank 的固定 0–1 色階對照圖。
- 兩個脚本均已實際跑完；位置平移誤差與停用 RoPE 後空間分數一致性的 assertions 通過。
