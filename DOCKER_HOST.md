# 在 host 啟動 Isaac ROS Jazzy

目前的 workspace 掛載：

- Host：`/home/ubuntu/workspaces/isaac_ros-dev`
- 容器：`/workspaces/isaac_ros-dev`

這些檔案已透過共享 workspace 寫到 host，不需要另外複製。在 **Jetson host 的終端機** 執行：

```bash
cd /home/ubuntu/workspaces/isaac_ros-dev
./start.sh
```

需要 Docker Engine、Docker Compose v2（支援 `up --wait`）及已設定的 NVIDIA Container Toolkit / `nvidia` runtime。
使用 `dinov3_ws/native/Dockerfile` 建置映像 `isaac-ros-jazzy-nvdcf:9.1`，沿用基底映像的
`/usr/local/bin/scripts/workspace-entrypoint.sh`，以 host UID/GID 建立 `admin` 使用者。
映像必須包含此 entrypoint。

```bash
./start.sh --detach  # 背景啟動
./start.sh           # 啟動或重用容器，進入 Bash
./start.sh --logs    # 查看日誌
./start.sh --config  # 檢查最終設定
./start.sh --down    # 停止並移除容器；workspace 保留
```

退出 Bash 後容器仍在背景執行。腳本預設將自身所在的 host 目錄掛到
`/workspaces/isaac_ros-dev`；若檔案移到別處，請指定原 workspace：

```bash
HOST_WORKSPACE=/home/ubuntu/workspaces/isaac_ros-dev /path/to/start.sh
```

Compose 使用 host network、host IPC、NVIDIA runtime 和 privileged 模式，供 ROS 2 discovery、GPU 及相機存取。
USB 掛載必須存在；X11 socket、Xauthority、Jetson 版本資訊、tegrastats、VPI、multimedia/SIPL API
與 jtop socket 則由腳本依 host 實際存在的路徑加入掛載。X11 使用既有的 host 授權檔，不自動修改 `xhost`。
不掛載 host 的 AWS / SSH 憑證或覆蓋容器的 `.bashrc`。

若需不同 ROS domain：

```bash
ROS_DOMAIN_ID=10 ./start.sh
```

容器內啟動既有相機及 pipeline：

```bash
cd /workspaces/isaac_ros-dev
./dinov3_ws/run_camera.sh
# 另一個 host 終端機執行 ./start.sh 進入同一個容器，再執行：
./dinov3_ws/run.sh
```

本設定會使用獨立的 Compose project `isaac-ros-jazzy`，不會自動停止既有開發容器。
啟動相機前請先停止其他容器中正在占用同一相機的程序。

## 容器 CUDA／TensorRT 初始化檢查

GPU driver libraries 由 host 的 NVIDIA Container Toolkit／runtime 注入。
Compose 保留 `runtime: nvidia`、`NVIDIA_VISIBLE_DEVICES: all` 與
`NVIDIA_DRIVER_CAPABILITIES: all`。建置脚本不再覆寫 CUDA driver 路徑。
Host 必須先正確設定 NVIDIA runtime；若映像內的 nvgpu／compatibility libraries
覆蓋了 host driver，應修正映像或 runtime 掛載，而非在模型建置時切換 driver。

在容器內以 `admin` 執行以下診斷，再開始建置：

```bash
cd /workspaces/isaac_ros-dev
dinov3_ws/.venv/bin/python dinov3_ws/tests/check_gpu_environment.py
```

診斷會列出實際載入的 `libcuda` 路徑、`cuInit` 狀態，並測試 PyTorch CUDA allocation
與 TensorRT Builder。`nvidia-smi` 成功本身不代表上述檢查會通過。

## GPU 裝置群組與套用變更

Compose 的啟動命令會讀取 `/dev/dri/renderD*`、`/dev/dri/card*` 和 `/dev/nvmap`
的實際 GID，建立缺少的群組，再透過映像已提供的免密碼 sudo 將 `admin`
加入群組。GID 為 0 的裝置會跳過。這些設定寫入容器的使用者資料庫，
使後續 `docker compose exec --user admin` 與 VS Code 終端取得相同群組。
完成後 PID 1 變成 `sleep`，健康檢查才通過。
此流程不需額外的 GPU 群組或 driver environment shell 檔案。

在 host 執行以下指令套用 Compose 變更；`up` 會重建設定已變更的服務：

```bash
cd /home/ubuntu/workspaces/isaac_ros-dev
./start.sh --detach
./start.sh
```

重新連線 VS Code／終端，再以 `id` 和 GPU 診斷確認權限與 driver 初始化。
如果 root 成功、admin 失敗，請比對 `id` 與 `ls -ln /dev/dri /dev/nvmap`。
若兩者都失敗，先检查 host driver、NVIDIA runtime 與容器映像相容性。
