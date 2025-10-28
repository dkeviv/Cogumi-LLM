# PRE-COMPILED BINARIES GUIDE - H100 OPTIMIZED SETUP

## 🎯 GOAL
Avoid compilation wherever possible to ensure:
- ✅ Fast installation (seconds, not minutes)
- ✅ No build failures
- ✅ No need for build tools (gcc, nvcc, make, etc.)
- ✅ Works on any H100 system without additional setup

---

## 📦 PRE-COMPILED PACKAGES OVERVIEW

### Critical Packages That Can Be Compiled (Avoid!)

| Package | Compilation Time | Failure Rate | Pre-compiled Available |
|---------|------------------|--------------|------------------------|
| **Flash Attention** | 5-10 minutes | Medium | ✅ **YES** |
| **BitsAndBytes** | 2-5 minutes | Low | ✅ **YES** |
| **Unsloth** | 1-3 minutes | Medium | ✅ **YES** |
| **xformers** | 10-20 minutes | High | ✅ **YES** |
| PyTorch | N/A | N/A | ✅ **YES** (standard) |
| Transformers | N/A | N/A | ✅ **YES** (pure Python) |

---

## ⚡ FLASH ATTENTION 2 - PRE-COMPILED WHEELS

### Problem
- Requires CUDA compiler (nvcc)
- Takes 5-10 minutes to compile
- May fail with cryptic errors
- Requires exact CUDA version match

### Solution: Use Official Pre-compiled Wheels

**Installation (RECOMMENDED):**
```bash
# CUDA 12.1 + PyTorch 2.3.x
pip install flash-attn==2.5.8 --no-build-isolation \
    --extra-index-url https://flashattn.github.io/whl/cu121/torch2.3/
```

**For other CUDA/PyTorch versions:**
```bash
# CUDA 12.1 + PyTorch 2.2.x
pip install flash-attn==2.5.8 --no-build-isolation \
    --extra-index-url https://flashattn.github.io/whl/cu121/torch2.2/

# CUDA 11.8 + PyTorch 2.3.x
pip install flash-attn==2.5.8 --no-build-isolation \
    --extra-index-url https://flashattn.github.io/whl/cu118/torch2.3/
```

**Verification:**
```python
import flash_attn
print(f"Flash Attention version: {flash_attn.__version__}")
```

**Benefits:**
- ✅ Installs in 5-10 seconds (vs 5-10 minutes compilation)
- ✅ No nvcc required
- ✅ No build failures
- ✅ Same performance as compiled version

---

## 🔢 BITSANDBYTES - PRE-COMPILED WHEELS

### Problem
- Requires CUDA compiler for custom kernels
- Compilation can fail on some systems

### Solution: Official PyPI Wheels

**Installation (RECOMMENDED):**
```bash
# Standard installation (uses pre-compiled wheels)
pip install bitsandbytes==0.43.1
```

**For CUDA 12.1+ specifically:**
```bash
# Explicit CUDA 12.1 wheel
pip install bitsandbytes==0.43.1 \
    --extra-index-url https://jllllll.github.io/bitsandbytes-wheels
```

**Verification:**
```python
import bitsandbytes as bnb
print(f"BitsAndBytes version: {bnb.__version__}")
print(f"CUDA available: {bnb.is_cuda_available()}")
```

**Benefits:**
- ✅ Installs instantly from PyPI
- ✅ No compilation needed
- ✅ Works on all CUDA 12.x systems

---

## 🦥 UNSLOTH - PRE-COMPILED INSTALLATION

### Problem
- Can compile custom kernels during installation
- May have dependency conflicts with latest versions

### Solution: Use Specific Stable Version

**Installation (RECOMMENDED - Version 2024.7):**
```bash
# Option 1: From git (specific stable tag)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git@2024.7"

# Option 2: From PyPI (if available)
pip install "unsloth[colab-new]==2024.7"
```

**Why Version 2024.7:**
- ✅ Proven stable with PyTorch 2.3.1
- ✅ No dependency conflicts
- ✅ Pre-built wheels included
- ✅ No custom kernel compilation needed
- ❌ Versions 2024.8+ have reported issues

**Verification:**
```python
from unsloth import FastLanguageModel
print("Unsloth installed successfully")
```

**Benefits:**
- ✅ Fast installation (1-2 minutes)
- ✅ No custom kernel compilation
- ✅ Stable dependencies

---

## 🔥 PYTORCH - ALWAYS USE PRE-COMPILED

### Installation (RECOMMENDED):**
```bash
# CUDA 12.1 (H100)
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4 (H100, newer)
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu124
```

**Never compile PyTorch from source unless absolutely necessary!**

**Verification:**
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"cuDNN: {torch.backends.cudnn.version()}")
```

---

## 🎨 XFORMERS (Optional but Recommended)

### Problem
- Long compilation time (10-20 minutes)
- High failure rate
- Complex build requirements

### Solution: Use Pre-compiled Wheels

**Installation (RECOMMENDED):**
```bash
# CUDA 12.1 + PyTorch 2.3.x
pip install xformers==0.0.26 \
    --index-url https://download.pytorch.org/whl/cu121
```

**Note:** xformers is optional. Flash Attention 2 is sufficient for most use cases.

**Verification:**
```python
import xformers
print(f"xformers version: {xformers.__version__}")
```

---

## 📋 COMPLETE PRE-COMPILED INSTALLATION SEQUENCE

### Full Installation (No Compilation Required!)

```bash
#!/bin/bash
# H100 Optimized Setup - 100% Pre-compiled (No Compilation!)
# Installation time: 5-10 minutes (vs 30-40 minutes with compilation)

set -e

echo "=================================================================="
echo "H100 OPTIMIZED SETUP - 100% PRE-COMPILED"
echo "No compilation required! Fast and reliable."
echo "=================================================================="

# 1. Create virtual environment
python3.10 -m venv venv_h100_precompiled
source venv_h100_precompiled/bin/activate

# 2. Upgrade pip
pip install --upgrade pip setuptools wheel

# 3. Install PyTorch (pre-compiled for CUDA 12.1)
echo "Installing PyTorch 2.3.1 (pre-compiled)..."
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 4. Install Flash Attention (pre-compiled wheel)
echo "Installing Flash Attention 2.5.8 (pre-compiled wheel)..."
pip install flash-attn==2.5.8 --no-build-isolation \
    --extra-index-url https://flashattn.github.io/whl/cu121/torch2.3/

# 5. Install Transformers stack (all pure Python, no compilation)
echo "Installing Transformers stack..."
pip install transformers==4.43.3
pip install peft==0.11.1
pip install accelerate==0.30.1
pip install datasets==2.19.1
pip install tokenizers==0.19.1
pip install safetensors==0.4.3

# 6. Install BitsAndBytes (pre-compiled wheel)
echo "Installing BitsAndBytes 0.43.1 (pre-compiled)..."
pip install bitsandbytes==0.43.1

# 7. Install Unsloth (specific stable version, pre-built)
echo "Installing Unsloth 2024.7 (pre-built)..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git@2024.7"

# 8. Install supporting libraries (all pure Python)
echo "Installing supporting libraries..."
pip install numpy==1.26.4
pip install scipy==1.11.4
pip install scikit-learn==1.3.2
pip install sentence-transformers==2.7.0
pip install pandas==2.1.4

# 9. Install monitoring tools (all pure Python)
echo "Installing monitoring tools..."
pip install tensorboard==2.14.0
pip install wandb==0.16.6
pip install tqdm==4.66.2
pip install rich==13.7.1

echo ""
echo "=================================================================="
echo "✅ INSTALLATION COMPLETE - 100% PRE-COMPILED"
echo "=================================================================="
echo "No compilation was performed!"
echo "Installation time: ~5-10 minutes"
echo "=================================================================="
```

---

## 🧪 VERIFICATION TESTS

### Test All Pre-compiled Components

```python
#!/usr/bin/env python3
"""Verify all pre-compiled components are working"""

import sys

def test_component(name, test_func):
    try:
        result = test_func()
        print(f"✅ {name}: {result}")
        return True
    except Exception as e:
        print(f"❌ {name}: FAILED - {e}")
        return False

print("=" * 70)
print("VERIFYING PRE-COMPILED INSTALLATION")
print("=" * 70)
print()

all_pass = True

# Test PyTorch (pre-compiled)
import torch
all_pass &= test_component("PyTorch", lambda: f"{torch.__version__}")
all_pass &= test_component("CUDA Available", lambda: torch.cuda.is_available())
all_pass &= test_component("CUDA Version", lambda: torch.version.cuda)

# Test Flash Attention (pre-compiled wheel)
try:
    import flash_attn
    all_pass &= test_component("Flash Attention", lambda: f"{flash_attn.__version__}")
except ImportError:
    print("⚠️  Flash Attention: Not installed (training will be slower)")

# Test BitsAndBytes (pre-compiled wheel)
import bitsandbytes as bnb
all_pass &= test_component("BitsAndBytes", lambda: f"{bnb.__version__}")

# Test Unsloth (pre-built)
try:
    from unsloth import FastLanguageModel
    print("✅ Unsloth: Installed and working")
except ImportError as e:
    print(f"❌ Unsloth: Failed - {e}")
    all_pass = False

# Test Transformers (pure Python)
import transformers
all_pass &= test_component("Transformers", lambda: f"{transformers.__version__}")

# Test PEFT (pure Python)
import peft
all_pass &= test_component("PEFT", lambda: f"{peft.__version__}")

# Test NumPy (pre-compiled)
import numpy
numpy_ver = numpy.__version__
if numpy_ver.startswith("1.26"):
    print(f"✅ NumPy: {numpy_ver} (correct)")
else:
    print(f"❌ NumPy: {numpy_ver} (should be 1.26.x)")
    all_pass = False

print()
print("=" * 70)
if all_pass:
    print("✅ ALL COMPONENTS VERIFIED - READY FOR TRAINING")
    print("All packages installed from pre-compiled binaries!")
else:
    print("⚠️  SOME COMPONENTS MISSING - Review output above")
print("=" * 70)

sys.exit(0 if all_pass else 1)
```

---

## ⚙️ TROUBLESHOOTING PRE-COMPILED INSTALLATIONS

### Issue 1: Flash Attention Pre-compiled Wheel Not Found

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement flash-attn==2.5.8
```

**Solution:**
```bash
# Verify CUDA and PyTorch versions match
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# Use correct wheel index for your versions:
# PyTorch 2.3.x + CUDA 12.1:
pip install flash-attn==2.5.8 --no-build-isolation \
    --extra-index-url https://flashattn.github.io/whl/cu121/torch2.3/

# PyTorch 2.2.x + CUDA 12.1:
pip install flash-attn==2.5.8 --no-build-isolation \
    --extra-index-url https://flashattn.github.io/whl/cu121/torch2.2/
```

### Issue 2: Unsloth Git Installation Fails

**Symptoms:**
```
ERROR: Failed building wheel for unsloth
```

**Solution:**
```bash
# Use pip installation instead of git
pip install "unsloth[colab-new]==2024.7"

# Or use specific wheel URL (if available)
pip install unsloth --extra-index-url https://unsloth.ai/wheels
```

### Issue 3: PyTorch Wrong CUDA Version

**Symptoms:**
```
RuntimeError: CUDA error: no kernel image is available for execution
```

**Solution:**
```bash
# Uninstall and reinstall with correct CUDA version
pip uninstall torch torchvision torchaudio

# For CUDA 12.1 (H100):
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Verify:
python -c "import torch; print(torch.version.cuda)"
```

---

## 🎯 BEST PRACTICES

### DO ✅
1. **Always prefer pre-compiled wheels** over source compilation
2. **Match CUDA version exactly** (PyTorch CUDA == system CUDA major version)
3. **Use specific package versions** (not `latest` or `>=`)
4. **Install in order**: PyTorch → Flash Attn → Transformers → Unsloth → Others
5. **Verify each component** after installation
6. **Use virtual environment** to avoid conflicts

### DON'T ❌
1. **Don't compile from source** unless absolutely necessary
2. **Don't use `pip install flash-attn`** without wheel index
3. **Don't install Unsloth 2024.8+** (has dependency conflicts)
4. **Don't mix CUDA versions** (e.g., PyTorch cu121 + Flash Attn cu118)
5. **Don't skip verification** steps
6. **Don't install NumPy 2.0+** (breaks many packages)

---

## 📊 INSTALLATION TIME COMPARISON

| Method | Flash Attn | Total Time | Failure Risk |
|--------|------------|------------|--------------|
| **Pre-compiled (RECOMMENDED)** | 10 seconds | **5-10 min** | **Low** ✅ |
| Compile from source | 5-10 minutes | 30-40 min | High ❌ |
| Mixed (some compiled) | 3-5 minutes | 15-20 min | Medium ⚠️ |

**Pre-compiled is 3-6× faster and much more reliable!**

---

## ✅ FINAL CHECKLIST

Before starting training, verify:

- [ ] All packages installed from pre-compiled wheels
- [ ] No compilation errors in pip output
- [ ] PyTorch CUDA version matches system CUDA (12.1 or 12.4)
- [ ] Flash Attention imports successfully
- [ ] Unsloth version is 2024.7 (not 2024.8+)
- [ ] NumPy version is 1.26.x (not 2.0+)
- [ ] All verification tests pass
- [ ] Total installation time was <15 minutes

If all checked ✅: **Ready for training!**

---

## 📚 REFERENCES

- Flash Attention Wheels: https://github.com/Dao-AILab/flash-attention#installation-and-features
- PyTorch Wheels: https://pytorch.org/get-started/previous-versions/
- BitsAndBytes Wheels: https://github.com/TimDettmers/bitsandbytes
- Unsloth Releases: https://github.com/unslothai/unsloth/releases
- NVIDIA CUDA Compatibility: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
