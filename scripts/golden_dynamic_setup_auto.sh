#!/bin/bash

# -----------------------------------
# golden_dynamic_setup_full_v2.sh
# One-shot dynamic installer for PyTorch GPU stack + Unsloth + TRL
# Handles Transformers circular dependencies
# -----------------------------------

set -e

# ----------------------------
# Parse flags
# ----------------------------
DRY_RUN_MODE=0
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN_MODE=1
fi

# ----------------------------
# Remove old venv and create new
# ----------------------------
if [[ -d "golden-venv" ]]; then
    echo "🔹 Removing existing golden-venv..."
    rm -rf golden-venv
fi

echo "🔹 Creating new virtual environment..."
python3 -m venv golden-venv
source golden-venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# ----------------------------
# Detect GPU and CUDA
# ----------------------------
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 || echo "CPU")
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep release | sed 's/.*release //' | sed 's/,//' || echo "none")

echo "🔹 Detected GPU: $GPU_NAME"
echo "🔹 CUDA Version: $CUDA_VERSION"

# ----------------------------
# Dry run
# ----------------------------
if [[ $DRY_RUN_MODE -eq 1 ]]; then
    echo "🔹 DRY RUN MODE"
    echo "Planned packages to install (dynamic golden set):"
    echo "  torch + torchvision + torchaudio (matching CUDA)"
    echo "  bitsandbytes"
    echo "  xformers"
    echo "  transformers==4.56.2 (compatible with Unsloth + TRL)"
    echo "  unsloth==2025.10.8"
    echo "  unsloth-zoo==2025.10.9"
    echo "  peft, trl, accelerate, datasets"
    exit 0
fi

# ----------------------------
# Determine PyTorch + CUDA wheels
# ----------------------------
TORCH_PACKAGE="torch"
TORCH_VISION_PACKAGE="torchvision"
TORCHAUDIO_PACKAGE="torchaudio"
TORCH_INDEX_URL=""

if [[ "$CUDA_VERSION" != "none" ]]; then
    TORCH_PACKAGE="torch==2.8.0+cu${CUDA_VERSION//./}"
    TORCH_VISION_PACKAGE="torchvision==0.15.2+cu${CUDA_VERSION//./}"
    TORCHAUDIO_PACKAGE="torchaudio==2.8.2+cu${CUDA_VERSION//./}"
    TORCH_INDEX_URL="-f https://download.pytorch.org/whl/cu${CUDA_VERSION//./}/torch_stable.html"
fi

# ----------------------------
# Install PyTorch + dependencies
# ----------------------------
echo "🔹 Installing PyTorch stack..."
set +e
pip install $TORCH_PACKAGE $TORCH_VISION_PACKAGE $TORCHAUDIO_PACKAGE $TORCH_INDEX_URL
TORCH_STATUS=$?
set -e

if [[ $TORCH_STATUS -ne 0 ]]; then
    echo "⚠️ PyTorch GPU install failed. Falling back to CPU version..."
    pip install torch torchvision torchaudio
fi

# ----------------------------
# Install other core ML packages
# ----------------------------
echo "🔹 Installing ML packages..."
pip install bitsandbytes xformers peft trl accelerate datasets
# ----------------------------
# Install Flash Attention 2
# ----------------------------
echo "🔹 Installing Flash Attention 2..."
set +e
pip install flash-attn --no-build-isolation
FLASH_STATUS=$?
set -e
if [[ $FLASH_STATUS -ne 0 ]]; then
    echo "⚠️ Flash Attention 2 install failed. Training will use xformers fallback."
fi


# ----------------------------
# Uninstall stale Transformers
# ----------------------------
echo "🔹 Removing any existing Transformers versions..."
pip uninstall -y transformers

# ----------------------------
# Install compatible Transformers + Unsloth + unsloth-zoo
# ----------------------------
TRANSFORMERS_VERSION="4.56.2"
UNSLOTH_VERSION="2025.10.8"
UNSLOTH_ZOO_VERSION="2025.10.9"
TRL_VERSION="0.23.0"

echo "🔹 Installing Transformers $TRANSFORMERS_VERSION + Unsloth + unsloth-zoo + TRL..."
pip install transformers==$TRANSFORMERS_VERSION
pip install unsloth==$UNSLOTH_VERSION unsloth-zoo==$UNSLOTH_ZOO_VERSION trl==$TRL_VERSION --force-reinstall

# ----------------------------
# Initialize xFormers submodules
# ----------------------------
XFORMERS_DIR=$(python -c "import xformers; import os; print(os.path.dirname(xformers.__file__))")
if [[ -d "$XFORMERS_DIR/third_party" ]]; then
    echo "🔹 Initializing xFormers submodules..."
    cd "$XFORMERS_DIR"
    git submodule update --init --recursive >/dev/null 2>&1 || echo "⚠️ Failed to init submodules"
    cd -
fi

# ----------------------------
# Register Jupyter kernel
# ----------------------------
pip install ipykernel
python -m ipykernel install --user --name=golden-venv --display-name "Python (golden-venv)"

# ----------------------------
# Verification
# ----------------------------
echo "🔹 Verifying installation..."
python - <<EOF
import sys
import torch

print(f"Python: {sys.version}")
print(f"Torch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

for pkg in ["bitsandbytes", "xformers", "transformers", "unsloth", "unsloth-zoo", "peft", "trl", "accelerate", "datasets"]:
    try:
        mod = __import__(pkg)
        print(f"{pkg}: {mod.__version__}")
    except Exception as e:
        print(f"⚠️ {pkg} not installed or import failed! ({e})")
EOF

echo "✅ Golden environment setup complete!"
echo "🔹 Use 'source golden-venv/bin/activate' to activate the venv."
echo "🔹 Restart your notebook kernel to apply Unsloth patches before training."
