#!/usr/bin/env python3
"""
Diagnose GPU availability and model loading issues
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

print("="*80)
print("GPU DIAGNOSTICS")
print("="*80)

# Check CUDA
print(f"\n1. CUDA Available: {torch.cuda.is_available()}")
print(f"   CUDA Version: {torch.version.cuda}")
print(f"   PyTorch Version: {torch.__version__}")

if torch.cuda.is_available():
    print(f"   GPU Count: {torch.cuda.device_count()}")
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("   ❌ CUDA NOT AVAILABLE - This is the problem!")
    print("   Possible causes:")
    print("   - CUDA drivers not installed")
    print("   - PyTorch CPU-only version installed")
    print("   - CUDA_VISIBLE_DEVICES not set correctly")
    exit(1)

# Check environment variables
print(f"\n2. Environment Variables:")
print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set (using all GPUs)')}")
print(f"   CUDA_DEVICE_ORDER: {os.environ.get('CUDA_DEVICE_ORDER', 'Not set')}")

# Test simple GPU operation
print(f"\n3. Testing GPU Computation:")
try:
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    print(f"   ✅ GPU computation works")
    print(f"   Test tensor device: {y.device}")
except Exception as e:
    print(f"   ❌ GPU computation failed: {e}")
    exit(1)

# Test model loading
print(f"\n4. Testing Model Loading:")
try:
    print("   Loading Llama 3.1 8B (this may take a minute)...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False
    )
    
    # Check where model parameters are
    devices = set(str(p.device) for p in model.parameters())
    print(f"   ✅ Model loaded")
    print(f"   Model devices: {devices}")
    
    if 'cuda' in str(next(model.parameters()).device):
        print(f"   ✅ Model is on GPU")
    else:
        print(f"   ❌ Model is NOT on GPU - device_map='auto' failed")
        print(f"   First parameter device: {next(model.parameters()).device}")
        
    # Check memory
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    print(f"   GPU Memory Used: {allocated:.1f} GB")
    
except Exception as e:
    print(f"   ❌ Model loading failed: {e}")
    exit(1)

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
