# Qwen3-VL-Embedding-8B 試用

辨識預設改為 `SAM3 W8A8 → Qwen3-VL-Embedding-8B → cosine 圖庫比對`。
SAM3 偵測、KCF、ROS topic 與圖庫資料夾沿用原流程。

```bash
./dinov3_ws/run.sh
```

設定在 `config/dinov3_cases.yaml`：

```yaml
recognition_backend: qwen
embedding_model: Qwen/Qwen3-VL-Embedding-8B
```

可指定本機模型目錄：

```bash
./dinov3_ws/run.sh -p embedding_model:=/absolute/path/to/model
```

切回原本 DINOv3 TensorRT：

```bash
./dinov3_ws/run.sh -p recognition_backend:=dinov3
```

Qwen 使用內建 Transformers `Qwen3VLModel` backbone，採用官方的影像對話格式、最後一個有效 token pooling 與 L2 normalization。沒有載入語言生成 head，也不執行下載的 Python 程式碼。公用模型固定 revision `2c4565515e0f265c6511776e7193b22c0968ddc7`；權重快取在 `artifacts/huggingface/hub`。首次執行會下載約 16.3 GB 權重。

圖庫與物件 ROI 都等比例縮放、右下邊緣填補至 448×448，再用 BF16 / SDPA 編碼為 4096 維全域向量。沿用每類取最佳參考圖的 cosine 分數與分類 margin。原本的辨識門檻 `0.85` / margin `0.05` 暫時保留，需用獨立相機影像重新校準。

全域向量沒有 DINOv3 的局部 patch 相似度熱圖；診斷頁預設顯示物件裁切圖，仍可看輸入與最佳參考圖。參考圖頁的格線表示 vision token 網格，並非局部相似度。原 ROS overlay topic 名稱保留供現有訂閱端使用。

## 實測與重現

在 NVIDIA Thor 上完成 16 張圖庫編碼與 4 類自我檢索，每個類別都可找回對應圖庫影像。穩態每張圖的編碼約 138–149 ms，Qwen 編碼測試的 GPU tensor 記憶體峰值約 15.26 GiB；此數字不包含 SAM3 子程序及完整應用程式。圖庫初始化約 7.9 秒，模型載入約 17 秒。

這是圖庫自我檢索的功能測試，不能據此判斷未見影像的辨識準確率。多個物件目前依序編碼，延遲會隨物件數增加；目前沒有做 Qwen TensorRT 或量化轉換。

```bash
PYTHONPATH=dinov3_ws/src/dinov3_cases \
  dinov3_trt11_venv/bin/python dinov3_ws/tests/check_qwen_pipeline.py
```

輸出報告：`artifacts/qwen3_vl_embedding_validation.json`。

另已短暫啟動 ROS / HTTP 服務接收現有相機，驗證 SAM3 TRT10 子程序、KCF、Qwen、448×448 參考圖 API 與全域向量診斷輸出。當次兩個物件約 329 ms / frame（約 3 FPS），均因低於沿用的 0.85 門檻標成 unknown；沒有據此自動降低門檻。此測試關閉深度計算，結果保存於 `artifacts/qwen3_vl_node_validation.json`，測試服務已停止。

官方參考：

- https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B
- https://github.com/QwenLM/Qwen3-VL-Embedding/blob/main/src/models/qwen3_vl_embedding.py
