# SAM3 + Cat2Real

獨立的 ROS 2／HTTP 商品辨識專案，模型與 Python 環境均位於此目錄。

- SAM3：官方 Torch-TensorRT FP16 vision + 官方 Transformers PyTorch FP16 bbox head，支援 1–4 個文字類別。
- 追蹤：DeepStream 9.1 NvDCF，原生 GPU 子程序；安裝：`sudo bash native/install_nvdcf.sh`。
- Cat2Real：`zhanganyi88/Cat2Real-DINOv3-384`，官方 Torch-TensorRT FP32，384×384、1024 維 pooler 向量。
- 圖庫：`data/gallery/`；模型：`artifacts/`；Python：`.venv/`。
- 即時影像：共用 960×540 H.264 編碼，提供網頁 WebRTC 與 ROS 封包 topic；優先使用 NVENC。詳見 [影像介面](src/dinov3_cases/README.md#共用-h264-即時影像)。

```bash
# 在本目錄執行；現有環境與模型已遷入
bash run.sh

# 新環境／模型
bash setup_env.sh
.venv/bin/python build_models.py --download

# 只驗證既有模型
.venv/bin/python build_models.py --validate-only
```

開啟 <http://127.0.0.1:8090>。ROS Jazzy、CUDA driver、相機驅動與 ROS 訊息套件需由系統提供；模型建置與推論不依賴相鄰的模型專案。SAM3 首次下載需要 Hugging Face 官方存取權；已有本機模型可離線執行。

套件清單：[requirements.txt](requirements.txt)、[requirements-build.txt](requirements-build.txt)、[Thor pins](constraints-thor.txt)。完整操作與排程說明見 [BUILD_GUIDE.md](BUILD_GUIDE.md)。

程式責任與資料流程見 [程式導覽](src/dinov3_cases/README.md)，檢查指令見 [測試說明](tests/README.md)。模型建置統一使用 `build_models.py`。
