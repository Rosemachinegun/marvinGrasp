#!/usr/bin/env bash
# Ubuntu 22.04 + NVIDIA GPU; never launches the robot application.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME=marvin-grasp
SKIP_SYSTEM=0
CHECK_ONLY=0
ROBOT_SETUP="${MARVIN_ROBOT_SETUP:-}"
usage() {
  cat <<'HELP'
用法：bash setup_env.sh [选项]
默认安装系统依赖、ROS2 Humble、Conda 环境、CUDA 和项目依赖，编译 PointNet2。
要求：Ubuntu 22.04 x86_64、可用的 NVIDIA 驱动、网络；系统安装需要 sudo。
  --env-name NAME     环境名称（默认 marvin-grasp；重跑会更新同名环境）
  --robot-setup FILE  已编译机器人工作空间的 install/setup.bash
  --skip-system      跳过 apt 和 ROS 安装（需已准备好系统依赖）
  --check            仅检查已有环境，不安装或生成文件
  -h, --help         显示帮助
安装成功后：source .marvin-env/activate.sh
日志保存在 .marvin-env/；不自动启动应用或操作机器人。
HELP
}
fail() { printf '\n[错误] %s\n' "$*" >&2; exit 1; }
while (($#)); do
  case "$1" in
    --env-name|--robot-setup)
      (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || fail "$1 缺少参数"
      if [[ "$1" == --env-name ]]; then ENV_NAME="$2"; else ROBOT_SETUP="$2"; fi
      shift 2 ;;
    --skip-system) SKIP_SYSTEM=1; shift ;;
    --check) CHECK_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "未知参数：$1（使用 --help 查看用法）" ;;
  esac
done
[[ "$ENV_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ && "$ENV_NAME" != base ]] || fail "请使用非 base 的环境名（字母、数字、下划线或短横线）"
[[ -z "$ROBOT_SETUP" || -f "$ROBOT_SETUP" ]] || fail "机器人工作空间入口不存在：$ROBOT_SETUP"
if [[ -n "$ROBOT_SETUP" ]]; then ROBOT_SETUP="$(realpath -- "$ROBOT_SETUP")"; fi
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || fail "当前脚本支持 Linux x86_64"
# shellcheck disable=SC1091
source /etc/os-release
[[ "$ID" == ubuntu && "$VERSION_ID" == 22.04 ]] || fail "需要 Ubuntu 22.04（ROS2 Humble / Python 3.10）"
cd "$ROOT"
export PYTHONNOUSERSITE=1
if (( ! CHECK_ONLY )); then
  mkdir -p "$ROOT/.marvin-env"
  LOG="$ROOT/.marvin-env/setup-$(date +%Y%m%d-%H%M%S)-$$.log"
  exec > >(tee -a "$LOG") 2>&1
fi
trap 'printf "\n[失败] 第 %s 行执行失败；修复后可重跑。日志：%s\n" "$LINENO" "${LOG:-未生成（检查模式）}" >&2' ERR
for file in environment-flowpose.yml perception/sam3/pyproject.toml \
  perception/flowpose/networks/pts_encoder/pointnet2_utils/pointnet2/setup.py \
  daimon_gripper/dm_gripper_py/setup.py daimon_gripper/dm_gripper_tac_py/setup.py; do
  [[ -f "$file" ]] || fail "缺少源码：$file，请准备完整项目（含夹爪 SDK）"
done
command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null || fail "NVIDIA 驱动不可用，请先安装驱动并重启，再运行脚本"
TMP_DIR=""
trap 'if [[ -n "$TMP_DIR" ]]; then rm -rf -- "$TMP_DIR"; fi' EXIT
if (( ! CHECK_ONLY && ! SKIP_SYSTEM )); then
  echo '[1/6] 安装系统依赖和 ROS2 Humble（可能需要输入 sudo 密码）'
  SUDO=()
  if (( EUID != 0 )); then
    command -v sudo >/dev/null || fail '需要 sudo 安装系统依赖'
    SUDO=(sudo)
    sudo -v
  fi
  "${SUDO[@]}" apt-get update
  "${SUDO[@]}" apt-get install -y --no-remove curl ca-certificates git build-essential \
    cmake ninja-build pkg-config python3 python3-pip software-properties-common \
    libgl1 libglib2.0-0 libusb-1.0-0-dev libudev-dev libportaudio2 portaudio19-dev \
    libxcb-xinerama0 libxkbcommon-x11-0 ffmpeg
  if [[ ! -f /opt/ros/humble/setup.bash ]]; then
    "${SUDO[@]}" add-apt-repository -y universe
    TMP_DIR="$(mktemp -d)"
    curl -fL --retry 3 https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest -o "$TMP_DIR/ros-release.json"
    ROS_APT_VERSION="$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tag_name"])' "$TMP_DIR/ros-release.json")"
    [[ "$ROS_APT_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail '无法解析 ROS apt 源版本'
    curl -fL --retry 3 "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_VERSION}/ros2-apt-source_${ROS_APT_VERSION}.jammy_all.deb" -o "$TMP_DIR/ros2-apt-source.deb"
    "${SUDO[@]}" dpkg -i "$TMP_DIR/ros2-apt-source.deb"
    "${SUDO[@]}" apt-get update
  fi
  "${SUDO[@]}" apt-get install -y --no-remove ros-humble-ros-base ros-humble-tf2-ros \
    ros-humble-geometry-msgs ros-humble-sensor-msgs ros-humble-trajectory-msgs \
    ros-humble-visualization-msgs ros-humble-xacro python3-colcon-common-extensions
  if [[ ! -f /etc/udev/rules.d/99-realsense-libusb.rules && ! -f /lib/udev/rules.d/99-realsense-libusb.rules ]]; then
    [[ -n "$TMP_DIR" ]] || TMP_DIR="$(mktemp -d)"
    curl -fL --retry 3 https://raw.githubusercontent.com/IntelRealSense/librealsense/v2.55.1/config/99-realsense-libusb.rules -o "$TMP_DIR/99-realsense-libusb.rules"
    "${SUDO[@]}" install -m 644 "$TMP_DIR/99-realsense-libusb.rules" /etc/udev/rules.d/99-realsense-libusb.rules
    "${SUDO[@]}" udevadm control --reload-rules
    echo '[提示] 已安装 RealSense USB 权限规则，请在安装完成后重新插拔相机'
  fi
fi
[[ -f /opt/ros/humble/setup.bash ]] || fail '缺少 ROS2 Humble；请去掉 --skip-system 后安装'
echo '[2/6] 查找或安装 Conda'
CONDA_BIN="${CONDA_EXE:-}"
if [[ ! -x "$CONDA_BIN" ]]; then CONDA_BIN="$(type -P conda || true)"; fi
if [[ ! -x "$CONDA_BIN" ]]; then
  for candidate in "$HOME/miniforge3/bin/conda" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda"; do
    if [[ -x "$candidate" ]]; then CONDA_BIN="$candidate"; break; fi
  done
fi
if [[ ! -x "$CONDA_BIN" ]]; then
  (( ! CHECK_ONLY )) || fail '找不到 Conda，请先运行安装'
  [[ ! -e "$HOME/miniforge3" ]] || fail '~/miniforge3 已存在但不可用，请修复或手动安装 Conda'
  [[ -n "$TMP_DIR" ]] || TMP_DIR="$(mktemp -d)"
  INSTALLER=Miniforge3-Linux-x86_64.sh
  for suffix in '' .sha256; do
    curl -fL --retry 3 "https://github.com/conda-forge/miniforge/releases/latest/download/${INSTALLER}${suffix}" -o "$TMP_DIR/${INSTALLER}${suffix}"
  done
  (cd "$TMP_DIR" && sha256sum -c "$INSTALLER.sha256")
  bash "$TMP_DIR/$INSTALLER" -b -p "$HOME/miniforge3"
  CONDA_BIN="$HOME/miniforge3/bin/conda"
fi
CONDA_BASE="$("$CONDA_BIN" info --base)"
# Conda and ROS setup scripts may reference unset variables.
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
set -u
if (( ! CHECK_ONLY )); then
  echo "[3/6] 创建或更新环境 $ENV_NAME（首次下载较大，请耐心等待）"
  env -u PYTHONPATH "$CONDA_BIN" env update --name "$ENV_NAME" --file "$ROOT/environment-flowpose.yml"
fi
set +u
conda activate "$ENV_NAME"
source /opt/ros/humble/setup.bash
if [[ -n "$ROBOT_SETUP" ]]; then source "$ROBOT_SETUP"; fi
set -u
export PYTHONNOUSERSITE=1
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-2}"
python -c 'import sys; assert sys.version_info[:2] == (3, 10), "需要 Python 3.10"'
if (( ! CHECK_ONLY )); then
  echo '[4/6] 检查 CUDA 并编译 PointNet2'
  python - <<'PY'
import torch
assert torch.cuda.is_available(), 'PyTorch 无法访问 GPU，请检查 NVIDIA 驱动'
assert torch.version.cuda == '12.1', f'期望 CUDA 12.1，实际为 {torch.version.cuda}'
print('GPU:', torch.cuda.get_device_name(0))
PY
  "$CUDA_HOME/bin/nvcc" --version
  # setup.py uses relative src paths, so build from its own directory.
  (cd "$ROOT/perception/flowpose/networks/pts_encoder/pointnet2_utils/pointnet2" && \
    python -m pip install --no-build-isolation --no-deps .)
fi
echo '[5/6] 验证依赖、ROS Python 和 CUDA 扩展'
# ROS apt packages have their own dependency metadata; validate the Conda
# environment separately, then validate ROS imports with its paths loaded.
env -u PYTHONPATH python -m pip check
python "$ROOT/scripts/check_env.py"
if (( ! CHECK_ONLY )); then
  echo '[6/6] 生成环境激活入口'
  # %q preserves spaces and shell metacharacters in local paths.
  {
    echo '# Generated by setup_env.sh; source this file from Bash.'
    echo 'if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then echo "请使用 source 激活此脚本" >&2; exit 1; fi'
    echo '_marvin_activate() {'
    echo '  local had_nounset=0 status=0'
    echo '  [[ $- != *u* ]] || had_nounset=1'
    echo '  set +u'
    printf '  source %q &&\n' "$CONDA_BASE/etc/profile.d/conda.sh"
    printf '  conda activate %q &&\n' "$ENV_NAME"
    echo '  source /opt/ros/humble/setup.bash &&'
    if [[ -n "$ROBOT_SETUP" ]]; then printf '  source %q &&\n' "$ROBOT_SETUP"; fi
    echo '  export PYTHONNOUSERSITE=1 CUDA_HOME="$CONDA_PREFIX" || status=$?'
    echo '  if (( had_nounset )); then set -u; fi'
    echo '  return "$status"'
    echo '}'
    echo 'if _marvin_activate; then unset -f _marvin_activate; else unset -f _marvin_activate; return 1; fi'
  } > "$ROOT/.marvin-env/activate.sh"
  printf '\n软件依赖安装并验证完成。新终端执行：\n  source %q\n' "$ROOT/.marvin-env/activate.sh"
fi
printf '\n启动前仍需准备模型权重、RealSense 权限/连接及机器人控制工作空间，详见 env.md。\n'
