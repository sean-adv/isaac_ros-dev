# SAM3 + Cat2Real ROS 2 workspace

此 repository 的維護範圍以 `dinov3_ws/` 與根目錄容器／啟動設定為主。辨識流程使用 SAM3、DeepStream NvDCF、Cat2Real DINOv3 與 RGB-D 3D 框，提供 ROS 2 和 HTTP／MJPEG 介面。

- [應用操作與環境設定](dinov3_ws/README.md)
- [模型與原生程式建置](dinov3_ws/BUILD_GUIDE.md)
- [模組責任、資料流程與介面](dinov3_ws/src/dinov3_cases/README.md)
- [回歸測試與效能工具](dinov3_ws/tests/README.md)
- [Host Docker／Compose 操作](DOCKER_HOST.md)
- [Repository audit、整理結果與部署缺口](REPOSITORY_AUDIT.md)

## 主要入口

在已設定 NVIDIA runtime 的 Jetson host 執行 `./start.sh` 進入容器。容器內從 repository 根目錄執行：

```bash
# 新環境安裝；不需對整支腳本加 sudo
bash dinov3_ws/setup_all.sh
# 下載與建置模型需要 Hugging Face SAM3 存取權
dinov3_ws/.venv/bin/python dinov3_ws/build_models.py --download
# 以下兩支入口在不同終端機執行
bash run_camera.sh
bash dinov3_ws/run.sh
```

開啟 <http://127.0.0.1:8090>。相機驅動須可由 ROS 找到，或指定 `ORBBEC_SETUP=/path/to/install/setup.bash`。部署參數在 `dinov3_ws/config/`；相機 launch 由外部 `orbbec_camera` 套件提供。

根目錄 `run.sh` 保留既有 host `docker exec` 相容入口；其預設 container 為 `money_arm_medicine`，workspace 為 `/workspaces/medicine_recog/isaac_ros-dev`，與 `start.sh` 的 Compose 設定不同。新 Compose 環境請進入容器後使用 `dinov3_ws/run.sh`；既有 host 使用者可透過 `DINOV3_CONTAINER`、`DINOV3_CONTAINER_WORKSPACE`、`DINOV3_CONTAINER_USER` 指定目標。

模型、TensorRT metadata、圖庫及 `.venv` 不包含在 Git 中。單獨 clone 不等於完成部署；所需系統資源與版本限制見上述建置與 audit 文件。相鄰的外部 repository、舊環境、`copy/`、`trash/` 與資料目錄保留原樣，不應直接打包整個 workspace。
