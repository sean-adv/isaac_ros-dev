# 去背後外框高相似度：離線診斷

使用目前預設 TensorRT DINOv3 ViT-H/16+，reference 長邊 640、cosine 閾值 0.70。
未修改推論服務、模型、參考圖或使用者既有修改。

## 已確認的程式行為

- `image_manager.remove_background` 清除透明像素的 RGB，背景變黑；不存在隱藏 RGB 未清除的問題。
- `node.decode_image` 轉成 RGB，丟棄 alpha。
- 建 reference bank 時未用 mask 篩選 patch；透明背景與混合邊界一併加入。
- `compute_similarity_maps` 對所有 reference patches 取最大 cosine。
- 熱圖使用固定 0–1 色階；bilinear interpolation 不會產生高於輸入 patch 最大值的新峰值，但會改變高分區域的輪廓。

## 實驗範圍

現有 `punateng` 六張 RGBA 去背圖，逐張作 target、其餘五張作 reference，避免同圖自匹配。
這些診斷 target 也縮到長邊 640，並非使用者原始 camera target；不能據此斷言原始畫面的原因。
提供 `--target` 時保留 target 原生解析度。

alpha 與 RGB 對齊縮放後，按每個完整 16×16 patch 的平均 alpha 分組：
前景 >= 0.95、背景 <= 0.05、其餘混合邊界。六張共 4520 patches，
其中前景 3911、純背景 35、混合邊界 574。部分被去背的窄區域位在混合 patch 中，
所以單純比較前景／純背景兩張 heatmap 不足以找出全部邊緣匹配來源。

以下按 target patch 數量加權彙總，均為未插值的原始 patch 分數：

| Target 區域 | 全 reference 平均 cosine | 僅 reference 前景平均 cosine | 全 reference >= 0.70 | 僅前景 >= 0.70 |
| --- | ---: | ---: | ---: | ---: |
| 外側一圈 patch（682 個） | 0.849 | 0.722 | 97.5% | 57.8% |
| alpha 純背景 patch（35 個） | 0.907 | 0.582 | 100% | 8.6% |
| alpha 前景 patch（3911 個） | 0.743 | 0.743 | 72.3% | 72.3% |

外側一圈的最佳 reference 匹配來源：混合邊界 70.1%、前景 24.8%、純背景 5.1%。
純背景 target 的最佳匹配來源：純背景 60%、混合邊界 40%、前景 0%。
前景 target 的最佳匹配來源：前景 97.6%、混合邊界 2.4%、純背景 0%。

## 結論與限制

現有資料支持的直接原因是 reference 背景／混合邊界仍參與 max matching；
其中外框主要由混合邊界 patch 提供高分。僅篩掉純背景、保留混合邊界，可能仍看到紅框。
前景 bank 在純背景區留下的高分比例只有 8.6%，但外框包含真實物體區域，
不可將其 57.8% 的高分直接判為模型錯誤。

原始 patch 網格已有高分，插值不是高分的起點。實驗尚不能分離 DINO 空間偏差、
self-attention 上下文與相似輪廓特徵，也未證實 semantic false positive。
等量抽樣前景／背景 heatmap 一併輸出供參考；前景只抽到 14–35 個候選，
單次抽樣不適合用來估計穩定的語意效能。

建議下一步保留 reference alpha，在建 bank 前先篩選高前景覆蓋率 patch，
同時保留混合邊界 bank 作診斷。用原始 target 重跑，確認實際畫面的高分來源再調整正式流程。

## 重現

在工作區根目錄執行：

```bash
PYTHONDONTWRITEBYTECODE=1 dinov3_venv/bin/python dinov3_ws/diagnostics/reference_background.py
```

原始 target：

```bash
PYTHONDONTWRITEBYTECODE=1 dinov3_venv/bin/python dinov3_ws/diagnostics/reference_background.py \
  --target /absolute/path/target.png --out /tmp/dinov3_target_diagnostic
```

`--model` 預設為 `vith16plus`，若實際服務使用 `vitb16`，應指定相同模型。
輸出 JSON 報告、各組 raw `.npy` 分數，以及固定色階、nearest-neighbor 放大的比較圖。
內建 assertions 檢查全 bank 分數等於分組分數的最大值，及 bilinear 不超出 patch 分數範圍。
