# 維護狀態與相容性

[回程式導覽](../README.md)

## 目前使用的流程

部署入口為 `node.py:main`；SAM3、NvDCF、Cat2Real、深度量測與 HTTP
共同使用節點狀態。`config.py` 保存程式預設值與啟動驗證；部署覆寫值仍在
[config/dinov3_cases.yaml](../../../config/dinov3_cases.yaml)。輸出 worker 與非同步追蹤的交接順序維持原有實作。

`recognition_margin`／HTTP `margin` 是舊版設定的相容參數，仍接受與驗證，
但僅保留於 ROS／HTTP 設定邊界，不再傳入分類函式。分類先依 DINO threshold 選候選，再選 Lab 分數最高的參考圖。
辨識輸出的 `margin` 是不同候選類別的 Lab 分數差，僅供診斷，與設定值不同。

## 現用姿態估算

`depth_bbox.py` 使用單平面 RANSAC，搭配平面內最小面積矩形長邊估算方向；
CPU Open3D 與 CUDA plane fitter 均保留。現用測試涵蓋傾斜平面、離群點、
方向歧義、擬合失敗回退、ROS quaternion 與繪圖一致性。

## 已移除的功能

2026-11-03 依使用者要求移除系統資源監控：刪除後端採樣器、前端腳本與
兩份專屬測試。此功能原先未接上部署節點與首頁；模型耗時計時維持原有功能。

2026-11-03 確認只使用現有 RANSAC 後，移除未接入主流程的
`surface_pose.py` 多平面輔助模組與 `test_surface_pose.py` 專屬測試。
現用 `test_depth_bbox.py`、`test_gpu_acceleration.py` 等回歸測試保留。

## 歷史回歸基準（移除遺留功能前）

2026-11-03 本輪整理前，完整 pytest 為 **480 通過、12 失敗**。
兩項 `test_resource_web.py` 失敗涉及未註冊的資源 API；十項
`test_surface_pose.py` 失敗涉及舊多平面契約與目前深度實作不一致。
這些是整理前已存在的失敗；不得透過排除或略過測試宣稱完整回歸通過。

執行方式與 GPU／相機整合檢查見 [測試說明](../../../tests/README.md)。

整理後完整 pytest 為 **488 通過、12 失敗**；新增的 8 項啟動設定測試通過，
失敗名稱與整理前一致，未新增 skip／xfail。前端 Node.js 測試因本環境沒有
`node` 而未執行；本輪未啟動真實模型／相機做端到端驗證。

## 移除資源監控後的驗證結果

2026-11-03 移除資源監控後，完整 pytest：**477 通過、10 項既有失敗**，失敗皆位於 `test_surface_pose.py`。移除的專屬測試原有 11 項通過、2 項失敗；沒有新增失敗或 skip／xfail。

## 最新驗證結果

2026-11-03 移除未使用的多平面模組與專屬測試後，完整 pytest：**474 通過、0 失敗**（14.37 秒）。現用 RANSAC／CUDA、深度框、追蹤、ROS 與 HTTP 測試均保留，未新增 skip／xfail。多平面專屬測試移除前為 3 通過、10 失敗。


## 單次辨識與同步清理（2026-11-03）

- `node.py` 改為直接呼叫 `recognize_objects`，移除每次工作都重新建立的
  `TrackedObjectRecognizer` 及無法在部署流程重用的跨呼叫快取／重新分類分支。
  `async_nvdcf.py` 仍負責追蹤期間的分類沿用、背景辨識與 seed ownership 綁定；
  診斷快照保留 seed-frame 框的獨立副本與延後繪圖。
- 分類、批次分類與單次辨識不再接收舊輸入 `margin`。ROS `recognition_margin`
  與 HTTP `margin` 仍接受並驗證，診斷輸出的 Lab gap 保留。
- 深度佇列改由既有 `node.lock` 保護，移除沒有等待者的 Condition 與通知。
- 取消追蹤 4 個 `__pycache__/*.pyc`，保留本機快取並沿用既有忽略規則。
- 更新整合檢查的辨識攔截入口與呼叫參數；移除舊快取類別的 9 項測試，改以
  2 項單次辨識測試驗證裁切順序、空集合、失敗後重試。節點交接、同影格交接、
  部分／全部失追與診斷沿用測試保留，並補強 HTTP margin 的相容與邊界驗證。

最終完整 pytest：**467 通過、0 失敗**（14.36 秒），未新增 skip／xfail；
74 個套件／測試 Python 檔案語法檢查通過。測試數由 474 減少 7 項，來自上述
9 項舊快取測試與 2 項替代測試。未啟動真實模型／相機做端到端驗證；本環境
沒有 Node.js，未執行前端測試。
