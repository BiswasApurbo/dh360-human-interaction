#!/usr/bin/env bash
# ==============================================================================
# DH360 Human Interaction — Raspberry Pi 5 Automated Setup Script
# ==============================================================================
# Target: Raspberry Pi 5 (8 GB RAM / 64-bit Raspberry Pi OS Bookworm ARM64)
# ==============================================================================

set -euo pipefail

echo "=========================================================="
echo " Starting DH360 Raspberry Pi 5 Environment Setup"
echo "=========================================================="

# 1. Update APT & install headless system libraries
echo "--> Installing required system packages via APT..."
sudo apt-get update
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    v4l-utils \
    curl \
    git

# 2. Setup Virtual Environment
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$INSTALL_DIR"

echo "--> Initializing Python 3 virtual environment at .venv..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

# 3. Install DH360 dependencies
echo "--> Installing Python package dependencies..."
if command -v uv &> /dev/null; then
    uv pip install -e .
else
    pip install -e .
fi

# 4. Download and verify ONNX Model Weights
echo "--> Downloading OpenCV Zoo ONNX model weights..."
python3 scripts/download_models.py --all

# 5. Prepare environment configuration template
if [ ! -f .env ]; then
    echo "--> Creating default .env from template..."
    cp .env.example .env
    echo "    Generated .env (Please edit .env to set your DH360_HI_API__OPERATOR_TOKEN)"
fi

echo "=========================================================="
echo " Setup Complete!"
echo " To run DH360 on Raspberry Pi 5:"
echo "   source .venv/bin/activate"
echo "   dh360-hi serve --config-file config/default.toml"
echo "=========================================================="
