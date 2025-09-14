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
  curl htop git-lfs gcc build-essential python3-dev linux-libc-dev \
  libgl1-mesa-glx libgl1-mesa-dev libglfw3 libglfw3-dev libglew-dev xorg-dev \
  libsvtav1-dev
git lfs install || true

# 2) Ensure conda is available and usable in this script
if ! command -v conda >/dev/null 2>&1; then
  echo "Conda not found, installing Miniconda to \$HOME/miniconda3..."
  tmpd="$(mktemp -d)"
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$tmpd/miniconda.sh"
  bash "$tmpd/miniconda.sh" -b -p "$HOME/miniconda3"
  rm -rf "$tmpd"
  eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
  conda init bash || true   # persist for future terminals
else
  eval "$(conda shell.bash hook)"   # no restart needed
  conda init bash || true           # persist for future terminals
fi

# 3) Create env if missing
if ! conda env list | awk '{print $1}' | grep -Fxq "lerobot"; then
  conda create -y -n lerobot -c conda-forge python=3.11 ffmpeg=7.1.1
fi

# 4) Activate and install Python packages
conda activate lerobot
conda install -y -c nvidia cuda-toolkit=12.4
pip install -U 'lerobot[smolvla]' 'huggingface_hub[cli]' gpustat wandb
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 5) Optional logins you should be registered in Wandb and Hugging Face to use them
if [ -n "${WANDB_API_KEY:-}" ]; then
  wandb login --relogin "$WANDB_API_KEY"
else
  wandb login
fi

if [ -n "${HF_TOKEN:-}" ]; then
  huggingface-cli login --token "$HF_TOKEN"
else
  huggingface-cli login
fi

echo "Setup done. Next time: 'conda activate lerobot'"
