# 維護狀態與介面邊界

[回程式導覽](../README.md) · [目前功能地圖](../../../../docs/CURRENT_FUNCTIONALITY.md) · [本輪清理與驗證](../../../../docs/CLEANUP_REPORT.md)

## 現用流程

部署入口為 `node.py:main`；SAM3、NvDCF、Cat2Real、深度量測與 HTTP
共用節點狀態。`config.py` 定義程式預設值與啟動驗證，部署覆寫在
[config/dinov3_cases.yaml](../../../config/dinov3_cases.yaml)。網頁設定不寫回 YAML。

同步模式直接呼叫 `recognize_objects`；非同步模式由 `async_nvdcf.py` 負責
背景 SAM3／DINO、分類沿用及 seed ownership 綁定。診斷快照保存 seed-frame
裁切與完整舊圖庫版本，按需繪製。深度佇列使用既有 `node.lock`。

`recognition_margin`／HTTP `margin` 仍是目前 UI／API 的設定契約，會接受與驗證，
但不傳入分類函式。分類先依 DINO threshold 選候選，再選 Lab 分數最高的參考圖。
辨識輸出的 `margin` 是不同候選分類的 Lab gap，僅供診斷，與設定值不同。
本輪保留這些對外欄位與 root host wrapper，外部使用情況未確認。

## 模型編碼介面

- 正式介面為 `prepare_image()`、`encode_with_descriptor()` 與 `encode_regions()`。
- `encode_object()` 用於圖庫與 `profile_gpu_pipeline.py` 的單張對照；
  使用 Cat2Real 置中白色補邊與 OpenCV 雙線性縮放、pooler descriptor。
- 候選框辨識直接使用批次 `encode_regions()`；保留批次切分、順序、pooler、
  預覽與無效描述向量測試。不提供逐張後備或 patch 平均描述向量。
- `preprocess_image()`、`patch_features()` 與相似度運算同時服務正式 encoder 與測試。
- `tests/encoder_stub.py` 是共用 CPU 替身；小型 Transformers／合成資料測試不代表
  部署模型數值驗證。正式固定 384 輸入與 FP32 pooler 行為保持原樣。

## 深度與影像輸出

`depth_bbox.py` 使用單平面 RANSAC 與平面內最小面積矩形長邊方向；CPU Open3D
及 CUDA fitter 都是現用路徑。測試涵蓋傾斜平面、離群點、方向歧義、擬合回退、
ROS quaternion 與繪圖一致性。

影像輸出只有 ROS Image／MJPEG；結構化 ROS 發布、疊圖、Image 發布與 JPEG
轉換在背景執行。詳細生命週期見 [runtime.md](runtime.md)。

## 已淘汰流程與開發入口

資源監控、多平面姿態輔助模組、H.264／WebRTC 舊傳輸、`TrackedObjectRecognizer`、
`prepare_object()`、`Encoder.encode()` 與 `run_dino_image()` 均已在本輪開始前移除，
不再屬於保留需求。原有深度／追蹤／分類／診斷測試仍保留。

原生建置統一使用 `native/build.sh [all|gpu_geometry|nvdcf_bridge]`；
系統依賴使用 `native/install_deps.sh`，完整準備使用 `setup_all.sh`。
移除只轉送指令的三支舊 native 相容腳本。Python 安裝直接使用 `requirements.txt`，
開發測試額外使用 `requirements-dev.txt`，不再經過空的 build requirements 包裝。

## 驗證邊界

本輪開始時完整 pytest：470 passed，無失敗／skip／xfail。
本輪完整 pytest 為 472 passed；另行執行的非同步完整模型 smoke 在清理前後均未通過，
兩次失敗位於再次交接的不同 pending 斷言，仍保留原檢查並記錄為待釐清。
詳細結果與未驗證項目集中於 [清理報告](../../../../docs/CLEANUP_REPORT.md)，
執行方式見 [測試說明](../../../tests/README.md)。根 REPOSITORY_AUDIT 保留較早工作的
歷史證據，其舊 API、測試數字與影片架構不能作目前功能依據。
