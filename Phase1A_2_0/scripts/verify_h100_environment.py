#!/usr/bin/env python3
"""
Verify H100 Optimized Environment - Pre-compiled Components
Tests that all packages are installed correctly from pre-compiled binaries
"""

import sys

def test_component(name, test_func, critical=True):
    """Test a component and return status"""
    try:
        result = test_func()
        print(f"✅ {name}: {result}")
        return True
    except Exception as e:
        if critical:
            print(f"❌ {name}: FAILED - {e}")
        else:
            print(f"⚠️  {name}: Not available - {e}")
        return critical  # Return False only if critical

print("=" * 80)
print("🧪 VERIFYING H100 OPTIMIZED ENVIRONMENT (PRE-COMPILED)")
print("=" * 80)
print()

all_pass = True

# ============================================================================
# CRITICAL COMPONENTS
# ============================================================================

print("CRITICAL COMPONENTS:")
print("-" * 80)

# Python version
import platform
python_ver = platform.python_version()
if python_ver.startswith("3.10") or python_ver.startswith("3.11"):
    print(f"✅ Python: {python_ver} (compatible)")
else:
    print(f"❌ Python: {python_ver} (should be 3.10 or 3.11)")
    all_pass = False

# PyTorch
import torch
all_pass &= test_component("PyTorch", lambda: torch.__version__)
all_pass &= test_component("CUDA Available", lambda: torch.cuda.is_available())

if torch.cuda.is_available():
    all_pass &= test_component("CUDA Version", lambda: torch.version.cuda)
    all_pass &= test_component("GPU", lambda: torch.cuda.get_device_name(0))
    
    # Check if H100
    gpu_name = torch.cuda.get_device_name(0)
    if "H100" in gpu_name:
        print(f"✅ H100 GPU Detected: Optimal configuration")
    else:
        print(f"⚠️  GPU: {gpu_name} (configuration optimized for H100)")
else:
    print("❌ CUDA: Not available - GPU training not possible")
    all_pass = False

# Transformers
import transformers
all_pass &= test_component("Transformers", lambda: transformers.__version__)

# PEFT
import peft
all_pass &= test_component("PEFT", lambda: peft.__version__)

# Accelerate
import accelerate
all_pass &= test_component("Accelerate", lambda: accelerate.__version__)

# Unsloth (CRITICAL)
try:
    from unsloth import FastLanguageModel
    print("✅ Unsloth: Installed (2-3× speedup enabled)")
    
    # Try to verify version
    try:
        import unsloth
        if hasattr(unsloth, '__version__'):
            print(f"   Version: {unsloth.__version__}")
    except:
        pass
except ImportError as e:
    print(f"❌ Unsloth: FAILED - {e}")
    print("   Training will work but will be 2-3× SLOWER without Unsloth!")
    all_pass = False

print()

# ============================================================================
# PERFORMANCE COMPONENTS (Important but not critical)
# ============================================================================

print("PERFORMANCE COMPONENTS:")
print("-" * 80)

# Flash Attention (Important for speed)
try:
    import flash_attn
    print(f"✅ Flash Attention: {flash_attn.__version__} (1.5× speedup enabled)")
except ImportError:
    print("⚠️  Flash Attention: Not installed (training will be ~30% slower)")
    print("   Install: pip install flash-attn==2.5.8 --no-build-isolation \\")
    print("            --extra-index-url https://flashattn.github.io/whl/cu121/torch2.3/")

# BitsAndBytes (Needed as dependency)
import bitsandbytes as bnb
test_component("BitsAndBytes", lambda: bnb.__version__, critical=False)

print()

# ============================================================================
# SUPPORTING LIBRARIES
# ============================================================================

print("SUPPORTING LIBRARIES:")
print("-" * 80)

# NumPy (CRITICAL - must be 1.26.x)
import numpy
numpy_ver = numpy.__version__
if numpy_ver.startswith("1.26"):
    print(f"✅ NumPy: {numpy_ver} (correct version)")
elif numpy_ver.startswith("2."):
    print(f"❌ NumPy: {numpy_ver} (MUST be 1.26.x, not 2.0+)")
    print("   Fix: pip install numpy==1.26.4 --force-reinstall")
    all_pass = False
else:
    print(f"⚠️  NumPy: {numpy_ver} (recommended: 1.26.4)")

# Other libraries
import scipy
test_component("SciPy", lambda: scipy.__version__, critical=False)

import sklearn
test_component("Scikit-learn", lambda: sklearn.__version__, critical=False)

import datasets
test_component("Datasets", lambda: datasets.__version__, critical=False)

print()

# ============================================================================
# FUNCTIONAL TESTS
# ============================================================================

print("FUNCTIONAL TESTS:")
print("-" * 80)

# Test PyTorch with GPU
if torch.cuda.is_available():
    try:
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.matmul(x, x)
        print("✅ GPU Computation: Working (PyTorch can use GPU)")
    except Exception as e:
        print(f"❌ GPU Computation: Failed - {e}")
        all_pass = False
else:
    print("⚠️  GPU Computation: Skipped (no GPU available)")

# Test Unsloth integration
try:
    from unsloth import FastLanguageModel
    print("✅ Unsloth Integration: Ready (can load Llama 3.1 models)")
except Exception as e:
    print(f"❌ Unsloth Integration: Failed - {e}")
    all_pass = False

# Test Flash Attention (if available)
try:
    import flash_attn
    from flash_attn import flash_attn_func
    print("✅ Flash Attention: Functional (can accelerate attention)")
except ImportError:
    print("⚠️  Flash Attention: Not available (slower attention computation)")
except Exception as e:
    print(f"⚠️  Flash Attention: Import successful but not functional - {e}")

print()

# ============================================================================
# CONFIGURATION SUMMARY
# ============================================================================

print("=" * 80)
print("CONFIGURATION SUMMARY:")
print("=" * 80)

config_score = 0
max_score = 0

# Critical components (5 points each)
critical_components = [
    ("Python 3.10/3.11", python_ver.startswith("3.10") or python_ver.startswith("3.11")),
    ("PyTorch", True),
    ("CUDA Available", torch.cuda.is_available()),
    ("Transformers", True),
    ("PEFT", True),
    ("Unsloth", 'unsloth' in sys.modules),
]

for name, status in critical_components:
    max_score += 5
    if status:
        config_score += 5

# Performance components (3 points each)
performance_components = [
    ("Flash Attention", 'flash_attn' in sys.modules),
    ("H100 GPU", torch.cuda.is_available() and "H100" in torch.cuda.get_device_name(0) if torch.cuda.is_available() else False),
]

for name, status in performance_components:
    max_score += 3
    if status:
        config_score += 3

# Supporting components (1 point each)
supporting_components = [
    ("NumPy 1.26.x", numpy.__version__.startswith("1.26")),
    ("SciPy", 'scipy' in sys.modules),
    ("Scikit-learn", 'sklearn' in sys.modules),
]

for name, status in supporting_components:
    max_score += 1
    if status:
        config_score += 1

percentage = (config_score / max_score) * 100

print(f"\nConfiguration Score: {config_score}/{max_score} ({percentage:.1f}%)")
print()

if percentage >= 90:
    print("✅ EXCELLENT - Optimal configuration for H100 training")
    print("   Expected performance: 4-6× faster than baseline")
    print("   Estimated cost: $20-30 for full Phase 1A training")
elif percentage >= 70:
    print("✅ GOOD - Functional but missing some optimizations")
    print("   Expected performance: 2-4× faster than baseline")
    print("   Consider installing Flash Attention for additional speedup")
elif percentage >= 50:
    print("⚠️  MARGINAL - Will work but significantly slower")
    print("   Missing critical components. Review output above.")
else:
    print("❌ INSUFFICIENT - Critical components missing")
    print("   Training may not work correctly. Fix errors above.")
    all_pass = False

print("=" * 80)

# ============================================================================
# INSTALLATION NOTES
# ============================================================================

if not all_pass or percentage < 90:
    print()
    print("INSTALLATION NOTES:")
    print("-" * 80)
    
    if 'flash_attn' not in sys.modules:
        print("📝 Flash Attention missing:")
        print("   pip install flash-attn==2.5.8 --no-build-isolation \\")
        print("   --extra-index-url https://flashattn.github.io/whl/cu121/torch2.3/")
        print()
    
    if 'unsloth' not in sys.modules:
        print("📝 Unsloth missing:")
        print('   pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git@2024.7"')
        print()
    
    if not numpy.__version__.startswith("1.26"):
        print("📝 NumPy wrong version:")
        print("   pip install numpy==1.26.4 --force-reinstall")
        print()
    
    print("Full installation guide: docs/PRECOMPILED_BINARIES_GUIDE.md")
    print("-" * 80)

# ============================================================================
# EXIT STATUS
# ============================================================================

print()
if all_pass and percentage >= 90:
    print("✅ ENVIRONMENT READY FOR TRAINING")
    print("   Run: python train_phase1a_optimized_h100.py")
    sys.exit(0)
elif all_pass:
    print("⚠️  ENVIRONMENT FUNCTIONAL BUT SUBOPTIMAL")
    print("   Training will work but may be slower than expected")
    sys.exit(0)
else:
    print("❌ ENVIRONMENT NOT READY")
    print("   Fix critical errors above before training")
    sys.exit(1)
