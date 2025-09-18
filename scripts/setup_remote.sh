#!/usr/bin/env bash

# I use this script to setup a remote machine to use LeRobot and other tools
# It installs different dependencies that can be missing
# The list can be redundant in some cases and not complete in others
# But it covers most of the cases I faced when worked with LeRobot
# It can take around 10 minutes to install all the dependencies
# Just run `bash setup_remote.sh` to install all the dependencies

set -euo pipefail

# 1) Apt deps
# Some dependencies can be redundant but I included all that caused me issues in the past
SUDO=""
if [ "${EUID:-$(id -u)}" -ne 0 ]; then SUDO="sudo"; fi

export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update
$SUDO apt-get install -y \
  curl htop git-lfs gcc g++ make build-essential python3-dev \
  linux-libc-dev "linux-headers-$(uname -r)" || true
$SUDO apt-get install -y \
  libgl1-mesa-glx libgl1-mesa-dev libglfw3 libglfw3-dev libglew-dev xorg-dev \
  libsvtav1-dev
git lfs install || true



# 2) Ensure conda is available and usable in this script
if command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
else
  echo "Conda not found, installing Miniconda to \$HOME/miniconda3..."
  tmpd="$(mktemp -d)"
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$tmpd/miniconda.sh"
  bash "$tmpd/miniconda.sh" -b -p "$HOME/miniconda3"
  rm -rf "$tmpd"
  CONDA_BIN="$HOME/miniconda3/bin/conda"
fi

# 3) Create env if missing
if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -Fxq "lerobot"; then
  "$CONDA_BIN" create -y -n lerobot -c conda-forge python=3.11 ffmpeg=7.1.1
fi
conda init

# 4) install Python packages
"$CONDA_BIN" install -y -n lerobot -c nvidia cuda-toolkit=12.4
# remove Conda toolchain packages so builds use system gcc + system headers
"$CONDA_BIN" remove -y -n lerobot \
  gcc_linux-64 gxx_linux-64 sysroot_linux-64 \
  libgcc-devel_linux-64 libstdcxx-devel_linux-64 || true

"$CONDA_BIN" run -n lerobot pip install -U \
  'lerobot[smolvla]' \
  'huggingface_hub[cli]' \
  gpustat \
  wandb

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

conda init
eval "$("$CONDA_BIN" shell.bash hook)"
conda activate lerobot

# 5) Optional logins
wandb login
huggingface-cli login

echo "Setup done. Next time: 'conda activate lerobot'"
