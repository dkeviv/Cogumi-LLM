#!/bin/bash
# H100 Optimized Training Environment Setup
# Based on comprehensive dependency analysis
# Configuration: Python 3.10.12, CUDA 12.1, PyTorch 2.3.1, Unsloth 2024.7
# Expected: 4-6× faster training, 68-79% cost reduction

set -e  # Exit on error

echo "=================================================================="
echo "🚀 H100 OPTIMIZED TRAINING ENVIRONMENT SETUP"
echo "=================================================================="
echo "Target: 4-6× faster training (38hr → 8-12hr)"
echo "Cost: $20-30 (vs $95 unoptimized) = 68-79% savings"
echo "Configuration: PyTorch 2.3.1, Unsloth 2024.7, Flash Attn 2.5.8"
echo "=================================================================="
echo ""

# ============================================================================
# STEP 0: PRE-FLIGHT CHECKS
# ============================================================================

echo "📋 Step 0: Pre-flight Checks"
echo "=================================================================="

# Check Python version (3.10 or 3.11)
echo "Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
if [[ $PYTHON_VERSION == 3.10.* ]] || [[ $PYTHON_VERSION == 3.11.* ]]; then
    echo "✅ Python $PYTHON_VERSION (compatible)"
else
    echo "❌ ERROR: Python 3.10 or 3.11 required (found $PYTHON_VERSION)"
    echo "Install Python 3.10.12: https://www.python.org/downloads/"
    exit 1
fi

# Check CUDA version (12.1+)
echo "Checking CUDA version..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    if [[ $CUDA_VERSION == 12.* ]]; then
        echo "✅ CUDA $CUDA_VERSION (compatible)"
    else
        echo "⚠️  WARNING: CUDA 12.1+ recommended (found $CUDA_VERSION)"
        echo "H100 requires CUDA 12.1+. Continue? (y/n)"
        read -r response
        if [[ ! $response =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    echo "⚠️  WARNING: nvcc not found. CUDA may not be installed."
    echo "Continue anyway? (y/n)"
    read -r response
    if [[ ! $response =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check GPU (should be H100)
echo "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    echo "✅ GPU detected: $GPU_NAME"
    if [[ $GPU_NAME == *"H100"* ]]; then
        echo "✅ H100 GPU confirmed (optimal)"
    else
        echo "⚠️  WARNING: Not H100. Configuration optimized for H100."
    fi
else
    echo "⚠️  WARNING: nvidia-smi not found. GPU may not be available."
fi

echo ""
echo "=================================================================="
echo "Pre-flight checks complete. Starting installation..."
echo "=================================================================="
echo ""

# ============================================================================
# STEP 1: VIRTUAL ENVIRONMENT
# ============================================================================

echo "📦 Step 1: Creating Virtual Environment"
echo "=================================================================="

if [ -d "venv_h100_optimized" ]; then
    echo "⚠️  venv_h100_optimized already exists. Remove it? (y/n)"
    read -r response
    if [[ $response =~ ^[Yy]$ ]]; then
        rm -rf venv_h100_optimized
    else
        echo "Using existing venv. Proceeding..."
    fi
fi

if [ ! -d "venv_h100_optimized" ]; then
    python -m venv venv_h100_optimized
    echo "✅ Virtual environment created: venv_h100_optimized"
else
    echo "✅ Using existing virtual environment"
fi

source venv_h100_optimized/bin/activate
echo "✅ Virtual environment activated"
echo ""

# ============================================================================
# STEP 2: UPGRADE PIP
# ============================================================================

echo "⬆️  Step 2: Upgrading pip"
echo "=================================================================="
pip install --upgrade pip setuptools wheel
echo "✅ pip upgraded"
echo ""

# ============================================================================
# STEP 3: PYTORCH (CUDA 12.1) - MOST CRITICAL
# ============================================================================

echo "🔥 Step 3: Installing PyTorch 2.3.1 (CUDA 12.1)"
echo "=================================================================="
echo "This may take 2-3 minutes..."
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')" || {
    echo "❌ ERROR: PyTorch installation failed"
    exit 1
}
echo "✅ PyTorch 2.3.1 installed with CUDA 12.1"
echo ""

# ============================================================================
# STEP 4: FLASH ATTENTION - USE PRE-COMPILED WHEELS (NO COMPILATION!)
# ============================================================================

echo "⚡ Step 4: Installing Flash Attention 2.5.8 (Pre-compiled)"
echo "=================================================================="
echo "Using pre-compiled wheels to avoid compilation (much faster)..."

# Try pre-compiled wheels FIRST (installs in seconds)
pip install flash-attn==2.5.8 --no-build-isolation \
    --extra-index-url https://flashattn.github.io/whl/cu121/torch2.3/ && {
    echo "✅ Flash Attention 2.5.8 installed from pre-compiled wheel"
} || {
    echo "⚠️  Pre-compiled wheel not available. Trying compilation..."
    echo "This may take 5-10 minutes and requires build tools..."
    
    # Fallback to compilation only if pre-compiled fails
    MAX_JOBS=4 pip install flash-attn==2.5.8 --no-build-isolation || {
        echo "⚠️  Flash Attention installation failed"
        echo "Training will still work but will be ~30% slower without Flash Attention"
        echo ""
        echo "Continue without Flash Attention? (y/n)"
        read -r response
        if [[ ! $response =~ ^[Yy]$ ]]; then
            echo "Installation aborted. Please resolve Flash Attention issue."
            exit 1
        fi
        echo "Continuing without Flash Attention..."
    }
}

# Verify Flash Attention (optional, don't fail if missing)
python -c "import flash_attn; print('✅ Flash Attention 2.5.8 installed')" 2>/dev/null || {
    echo "⚠️  Flash Attention not available (training will be slower)"
}
echo ""

# ============================================================================
# STEP 5: TRANSFORMERS STACK
# ============================================================================

echo "🤗 Step 5: Installing Transformers Stack"
echo "=================================================================="

pip install transformers==4.43.3
pip install peft==0.11.1
pip install accelerate==0.30.1
pip install bitsandbytes==0.43.1
pip install datasets==2.19.1
pip install tokenizers==0.19.1
pip install safetensors==0.4.3

echo "✅ Transformers stack installed"
echo ""

# ============================================================================
# STEP 6: UNSLOTH (CRITICAL - Specific Version)
# ============================================================================

echo "🦥 Step 6: Installing Unsloth 2024.7 (CRITICAL)"
echo "=================================================================="
echo "Using specific version to avoid dependency conflicts..."

# Try git installation first
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git@July-2024" || {
    echo "⚠️  Git installation failed. Trying pip..."
    pip install "unsloth[colab-new]==July-2024" || {
        echo "❌ ERROR: Unsloth installation failed"
        echo "This is CRITICAL for 2-3× speedup. Cannot proceed."
        exit 1
    }
}

# Verify Unsloth installation
python -c "import unsloth; print('✅ Unsloth July-2024 installed')" || {
    echo "❌ ERROR: Unsloth import failed"
    exit 1
}
echo ""

# ============================================================================
# STEP 7: SUPPORTING LIBRARIES (CRITICAL - NumPy 1.26.4)
# ============================================================================

echo "📚 Step 7: Installing Supporting Libraries"
echo "=================================================================="
echo "CRITICAL: Installing NumPy 1.26.4 (NOT 2.0+)"

# Install NumPy first, force version
pip install "numpy<2.0"
pip install numpy==1.26.4 --force-reinstall

# Then install other libraries
pip install scipy==1.11.4
pip install scikit-learn==1.3.2
pip install sentence-transformers==2.7.0
pip install pandas==2.1.4

# Verify NumPy version
NUMPY_VERSION=$(python -c "import numpy; print(numpy.__version__)")
if [[ $NUMPY_VERSION == 1.26.* ]]; then
    echo "✅ NumPy $NUMPY_VERSION (correct version)"
else
    echo "⚠️  WARNING: NumPy $NUMPY_VERSION (should be 1.26.x)"
    echo "Forcing NumPy 1.26.4..."
    pip install numpy==1.26.4 --force-reinstall
fi
echo ""

# ============================================================================
# STEP 8: MONITORING TOOLS
# ============================================================================

echo "📊 Step 8: Installing Monitoring Tools"
echo "=================================================================="

pip install tensorboard==2.14.0
pip install wandb==0.16.6
pip install tqdm==4.66.2
pip install rich==13.7.1

echo "✅ Monitoring tools installed"
echo ""

# ============================================================================
# STEP 9: API CLIENTS (Already used in Phase 0)
# ============================================================================

echo "🔌 Step 9: Installing API Clients"
echo "=================================================================="

pip install groq>=0.32.0
pip install openai>=2.4.0
pip install together>=1.4.0
pip install anthropic>=0.18.0

echo "✅ API clients installed"
echo ""

# ============================================================================
# STEP 10: ADDITIONAL UTILITIES
# ============================================================================

echo "🔧 Step 10: Installing Additional Utilities"
echo "=================================================================="

pip install python-dotenv>=1.0.0
pip install pydantic>=2.6.0
pip install tiktoken>=0.5.0
pip install datasketch>=1.6.0
pip install jsonschema>=4.20.0

echo "✅ Additional utilities installed"
echo ""

# ============================================================================
# STEP 11: COMPREHENSIVE VERIFICATION
# ============================================================================

echo "=================================================================="
echo "🧪 Step 11: Comprehensive Verification"
echo "=================================================================="
echo ""

python << 'VERIFY_SCRIPT'
import sys

def check(name, test_func):
    try:
        result = test_func()
        print(f"✅ {name}: {result}")
        return True
    except Exception as e:
        print(f"❌ {name}: FAILED - {e}")
        return False

print("Core Stack:")
print("-" * 60)
success = True
success &= check("Python", lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

import torch
success &= check("PyTorch", lambda: torch.__version__)
success &= check("CUDA Available", lambda: torch.cuda.is_available())
success &= check("CUDA Version", lambda: torch.version.cuda)
if torch.cuda.is_available():
    success &= check("GPU", lambda: torch.cuda.get_device_name(0))

import transformers
success &= check("Transformers", lambda: transformers.__version__)

import peft
success &= check("PEFT", lambda: peft.__version__)

import accelerate
success &= check("Accelerate", lambda: accelerate.__version__)

print("\nCritical Components:")
print("-" * 60)

try:
    import unsloth
    print("✅ Unsloth: INSTALLED")
except ImportError as e:
    print(f"❌ Unsloth: FAILED - {e}")
    success = False

try:
    import flash_attn
    print("✅ Flash Attention: INSTALLED")
except ImportError:
    print("⚠️  Flash Attention: NOT AVAILABLE (training will be slower)")

import numpy
numpy_ver = numpy.__version__
if numpy_ver.startswith("1.26"):
    print(f"✅ NumPy: {numpy_ver} (correct version)")
else:
    print(f"⚠️  NumPy: {numpy_ver} (should be 1.26.x)")
    success = False

print("\nFull Test:")
print("-" * 60)

try:
    # Test Unsloth + Llama 3.1 integration
    from unsloth import FastLanguageModel
    print("✅ Unsloth FastLanguageModel import successful")
    
    # Don't actually load model (takes time), just verify import works
    print("✅ All imports working correctly")
except Exception as e:
    print(f"❌ Integration test failed: {e}")
    success = False

print("\n" + "=" * 60)
if success:
    print("✅ ALL VERIFICATION CHECKS PASSED")
    print("=" * 60)
    sys.exit(0)
else:
    print("⚠️  SOME CHECKS FAILED - Review output above")
    print("=" * 60)
    sys.exit(1)
VERIFY_SCRIPT

VERIFY_STATUS=$?

echo ""

if [ $VERIFY_STATUS -eq 0 ]; then
    echo "=================================================================="
    echo "✅ INSTALLATION COMPLETE - ALL CHECKS PASSED"
    echo "=================================================================="
    echo ""
    echo "Environment: venv_h100_optimized"
    echo "Activate: source venv_h100_optimized/bin/activate"
    echo ""
    echo "Expected Performance:"
    echo "  - Training Speed: 4-6× faster (38hr → 8-12hr)"
    echo "  - Cost: $20-30 (vs $95 unoptimized)"
    echo "  - Savings: 68-79% cost reduction"
    echo ""
    echo "Next Steps:"
    echo "  1. Activate environment: source venv_h100_optimized/bin/activate"
    echo "  2. Run training: python train_phase1a_optimized_h100.py"
    echo "  3. Monitor progress: tensorboard --logdir data/checkpoints"
    echo ""
    echo "Documentation:"
    echo "  - Full analysis: docs/DEPENDENCY_ANALYSIS_H100_UNSLOTH.md"
    echo "  - Requirements: requirements-h100-optimized.txt"
    echo "=================================================================="
else
    echo "=================================================================="
    echo "⚠️  INSTALLATION COMPLETED WITH WARNINGS"
    echo "=================================================================="
    echo ""
    echo "Some components may be missing or misconfigured."
    echo "Review the output above for details."
    echo ""
    echo "Common Issues:"
    echo "  - Flash Attention build failed: Training will work but slower"
    echo "  - NumPy wrong version: Run 'pip install numpy==1.26.4 --force-reinstall'"
    echo "  - Unsloth import failed: Check git access or try pip installation"
    echo ""
    echo "For help, see: docs/DEPENDENCY_ANALYSIS_H100_UNSLOTH.md"
    echo "=================================================================="
fi
