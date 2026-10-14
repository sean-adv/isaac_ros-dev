#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'HELP'
用法：./start.sh [--detach|--down|--logs|--config|--help]
  無參數      啟動容器，並進入 admin 的 Bash；離開 shell 後容器繼續運行
  --detach    只啟動容器
  --down      停止並移除本 Compose 的容器
  --logs      查看容器日誌
  --config    顯示包含 host 可用掛載的 Compose 設定，不啟動容器
環境變數：HOST_WORKSPACE、HOST_USER_UID、HOST_USER_GID、DISPLAY、XAUTHORITY、ROS_DOMAIN_ID
請在 Linux / Jetson host 執行，需要 Docker Compose v2 和 NVIDIA runtime。
HELP
}

mode="${1:-shell}"
if (( $# > 1 )); then usage >&2; exit 2; fi
case "$mode" in
  --help|-h) usage; exit 0 ;;
  shell|--detach|--down|--logs|--config) ;;
  *) usage >&2; exit 2 ;;
esac

if [[ -f /.dockerenv || -f /run/.containerenv ]]; then
  echo '請在 host 執行此腳本，而非目前的開發容器。' >&2
  exit 1
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export HOST_WORKSPACE="${HOST_WORKSPACE:-$script_dir}"
if [[ ! -d "$HOST_WORKSPACE" || "$HOST_WORKSPACE" != /* ]]; then
  echo "HOST_WORKSPACE 必須是已存在的 host 絕對路徑：$HOST_WORKSPACE" >&2
  exit 1
fi
export HOST_USER_UID="${HOST_USER_UID:-${SUDO_UID:-$(id -u)}}"
export HOST_USER_GID="${HOST_USER_GID:-${SUDO_GID:-$(id -g)}}"
if [[ ! "$HOST_USER_UID" =~ ^[0-9]+$ || ! "$HOST_USER_GID" =~ ^[0-9]+$ || "$HOST_USER_UID" -eq 0 ]]; then
  echo '請以一般 host 使用者執行，或指定非 root 的 HOST_USER_UID 與 HOST_USER_GID。' >&2
  exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
  echo '找不到 Docker Compose v2；請先在 host 安裝 Docker Compose plugin。' >&2
  exit 1
fi

# Optional host resources must not become empty directories when absent.
override_file="$(mktemp "${TMPDIR:-/tmp}/isaac-ros-compose.XXXXXX.yaml")"
trap 'rm -f -- "$override_file"' EXIT
printf 'services:\n  isaac-ros:\n    volumes: []\n' > "$override_file"
mount_count=0
yaml_quote() {
  local value="${1//\'/\'\'}"
  printf "'%s'" "$value"
}
add_mount() {
  local source="$1" target="$2" read_only="${3:-true}"
  [[ -e "$source" ]] || return 0
  if (( mount_count == 0 )); then
    printf 'services:\n  isaac-ros:\n    volumes:\n' > "$override_file"
  fi
  {
    printf '      - type: bind\n        source: '
    yaml_quote "$source"
    printf '\n        target: '
    yaml_quote "$target"
    printf '\n        read_only: %s\n        bind:\n          create_host_path: false\n' "$read_only"
  } >> "$override_file"
  mount_count=$((mount_count + 1))
}

add_mount /tmp/.X11-unix /tmp/.X11-unix
host_home="${HOME}"
if [[ -n "${SUDO_USER:-}" ]]; then
  host_home="$(getent passwd "$SUDO_USER" | cut -d: -f6)"
fi
add_mount "${XAUTHORITY:-$host_home/.Xauthority}" /home/admin/.Xauthority
add_mount /etc/nv_tegra_release /etc/nv_tegra_release
add_mount /usr/bin/tegrastats /usr/bin/tegrastats
add_mount /usr/share/vpi3 /usr/share/vpi3
add_mount /usr/src/jetson_multimedia_api /usr/src/jetson_multimedia_api
add_mount /usr/src/jetson_sipl_api /usr/src/jetson_sipl_api
add_mount /run/jtop.sock /run/jtop.sock false

compose=(docker compose --project-name isaac-ros-jazzy --project-directory "$script_dir"
  -f "$script_dir/compose.yaml" -f "$override_file")
case "$mode" in
  --config) "${compose[@]}" config ;;
  --down) "${compose[@]}" down ;;
  --logs) "${compose[@]}" logs --follow isaac-ros ;;
  shell|--detach)
    "${compose[@]}" up -d --wait --wait-timeout 90 isaac-ros
    if [[ "$mode" == shell ]]; then
      tty_args=()
      if [[ ! -t 0 || ! -t 1 ]]; then tty_args=(-T); fi
      "${compose[@]}" exec "${tty_args[@]}" --user admin isaac-ros bash
    fi
    ;;
esac
