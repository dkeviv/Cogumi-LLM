# Google Colab Dependency Management Strategy

## Problem Overview

Google Colab comes pre-installed with many packages that conflict with QLoRA training requirements:

### Pre-installed Conflicts
- **torch 2.8.0** (we need 2.4.0)
- **torchvision 0.23.0** (requires torch 2.8.0, not needed)
- **torchaudio 2.8.0** (requires torch 2.8.0, not needed)
- **tensorflow 2.19.0** (conflicts with tensorboard version)
- **opencv 4.12.0** (requires numpy 2.x, we need 1.26.x)
- **latest transformers** (may include vision models)
- **latest Axolotl** (requires torch 2.6.0+, transformers 4.57+)

## Solution: Clean Install Approach

### Step 1: Complete Cleanup (Cell 7)
Remove ALL conflicting packages before installing our versions:

```python
conflicting_packages = [
    'torch', 'torchvision', 'torchaudio',
    'transformers', 'accelerate', 'peft',
    'tensorflow', 'tensorboard',
    'opencv-python', 'opencv-python-headless', 'opencv-contrib-python',
    'timm', 'pillow',
    'axolotl',
]
```

### Step 2: Install Compatible Versions
Install in specific order to avoid dependency resolution conflicts:

```python
# 1. PyTorch ecosystem (specific CUDA version)
torch==2.4.0 (cu118)

# 2. Core ML packages
transformers==4.41.0
accelerate==0.33.0
peft==0.12.0
bitsandbytes==0.43.3

# 3. Data handling
datasets==2.20.0
tokenizers==0.19.1
numpy==1.26.4

# 4. Monitoring
tensorboard==2.17.0
wandb

# 5. Axolotl (specific version)
axolotl v0.4.0 (with --no-deps flag)
```

### Step 3: Manual Dependency Installation
Install Axolotl's critical dependencies manually:
- fire
- pyyaml
- huggingface-hub

## Why This Works

### 1. **Clean Slate**
- Removes all pre-installed packages that conflict
- Prevents pip's dependency resolver from trying to satisfy conflicting requirements
- Ensures no hidden dependencies interfere

### 2. **Version Compatibility Matrix**

| Package | Version | Why This Version |
|---------|---------|------------------|
| torch | 2.4.0 | Last stable before 2.6.0 jump, full QLoRA support |
| transformers | 4.41.0 | Pre-vision models, LLAMA-3.2 support, stable |
| accelerate | 0.33.0 | Compatible with torch 2.4.0 and transformers 4.41 |
| peft | 0.12.0 | Proven QLoRA support, works with all above |
| bitsandbytes | 0.43.3 | NF4 quantization, compatible with torch 2.4.0 |
| Axolotl | v0.4.0 | Pre-strict-versioning, works with our stack |

### 3. **No Vision Dependencies**
- Removed: timm, torchvision, pillow, opencv
- Not needed for LLM training
- Primary source of torch version conflicts

### 4. **Controlled Axolotl Installation**
- Use `--no-deps` flag to prevent automatic dependency installation
- Manually install only required dependencies
- Avoids Axolotl pulling in conflicting package versions

## Verification Process (Cell 28)

After installation, verify:
1. ✅ torch 2.4.x imported
2. ✅ transformers 4.41.x imported
3. ✅ accelerate imported
4. ✅ peft imported
5. ✅ bitsandbytes imported
6. ✅ axolotl imported
7. ✅ AutoModelForCausalLM imported (critical test)

If any fail → Runtime restart → Rerun cell 7

## Common Issues & Solutions

### Issue: "ResolutionImpossible" error
**Cause**: Pre-installed packages still present
**Solution**: 
1. Runtime → Restart runtime
2. Rerun cell 7 (dependencies)

### Issue: torch version mismatch
**Cause**: torchaudio/torchvision reinstalled by another package
**Solution**: Cell 7 removes these at start

### Issue: Axolotl import fails
**Cause**: Missing manual dependencies
**Solution**: Cell 7 installs fire, pyyaml, huggingface-hub

### Issue: Training launch fails
**Cause**: Dependency got reinstalled during another operation
**Solution**: Run verification cell (28), check versions

## Nuclear Option (Cell 29)

If nothing works:
1. Uncomment: `import os; os.kill(os.getpid(), 9)`
2. Runtime restarts completely
3. Rerun cell 7 (dependencies)
4. Re-upload dataset
5. Proceed with training

## Timeline

- **Cleanup**: 30-60 seconds
- **Core packages**: 2-3 minutes
- **Axolotl**: 2-3 minutes
- **Total**: ~5-7 minutes

## Success Criteria

✅ No "ERROR: pip's dependency resolver" messages
✅ All verification checks pass
✅ `accelerate launch` command works
✅ Training starts without import errors

## Why Previous Attempts Failed

1. **Using `-U` flag**: Allowed pip to "upgrade" to latest (conflicting) versions
2. **Not removing pre-installed**: Colab packages stayed, created conflicts
3. **Latest Axolotl**: Auto-installed torch 2.6.0+ dependencies
4. **Vision packages**: torchvision pulled in torch 2.8.0

## This Approach

1. ✅ **Removes everything first**: Clean slate
2. ✅ **Specific versions only**: No upgrades allowed
3. ✅ **Old Axolotl**: Compatible with our stack
4. ✅ **No vision packages**: Eliminated torch conflicts
5. ✅ **Manual dependencies**: Full control over what's installed

---

**Result**: Clean, conflict-free installation that works reliably for QLoRA training on Google Colab.
