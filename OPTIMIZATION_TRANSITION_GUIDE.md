# 🚀 Safe Transition to Optimized Training

## ⚠️ CRITICAL: Follow These Steps Exactly

### Current Status
- **Current step**: 885 / 60,060
- **Current speed**: 8.86 seconds/step
- **Current ETA**: ~148 hours (6 days)

### Target Status
- **Optimized speed**: 5-6 seconds/step
- **Optimized ETA**: ~90 hours (3.75 days)
- **Time saved**: 58 hours (~2.5 days) ⚡

---

## 📋 Step-by-Step Transition (NO DATA LOSS)

### STEP 1: Wait for Checkpoint ⏰

**Current step**: 885  
**Next checkpoint**: 1000 (in ~30 minutes)

**DO NOT STOP TRAINING YET!**

Wait until you see in the logs:
```
Saving model checkpoint to data/checkpoints/llama-3.1-8b-phase1a/checkpoint-1000
```

---

### STEP 2: Verify Checkpoint Exists ✅

Run this in a NEW Colab cell:

```python
import os
import glob

checkpoint_dir = "data/checkpoints/llama-3.1-8b-phase1a"

# Find all checkpoints
checkpoints = sorted([
    d for d in os.listdir(checkpoint_dir) 
    if d.startswith('checkpoint-')
])

print("📂 Available checkpoints:")
for cp in checkpoints:
    cp_path = f"{checkpoint_dir}/{cp}"
    size = sum(os.path.getsize(os.path.join(cp_path, f)) for f in os.listdir(cp_path)) / 1e9
    print(f"  ✅ {cp} ({size:.2f} GB)")

# Verify latest checkpoint
latest = checkpoints[-1] if checkpoints else None
if latest:
    latest_path = f"{checkpoint_dir}/{latest}"
    required_files = ['config.json', 'pytorch_model.bin', 'trainer_state.json']
    
    print(f"\n🔍 Verifying {latest}:")
    for file in required_files:
        exists = os.path.exists(f"{latest_path}/{file}")
        status = "✅" if exists else "❌"
        print(f"  {status} {file}")
    
    print(f"\n🎯 Resume command:")
    print(f"python train_qlora_optimized.py --resume_from_checkpoint {latest_path}")
else:
    print("\n❌ No checkpoints found! Wait for step 1000.")
```

**Expected output:**
```
📂 Available checkpoints:
  ✅ checkpoint-1000 (2.34 GB)

🔍 Verifying checkpoint-1000:
  ✅ config.json
  ✅ pytorch_model.bin
  ✅ trainer_state.json

🎯 Resume command:
python train_qlora_optimized.py --resume_from_checkpoint data/checkpoints/llama-3.1-8b-phase1a/checkpoint-1000
```

---

### STEP 3: Check System Resources 🖥️

Run this to verify you have enough resources:

```python
import os
import psutil
import torch

print("=" * 60)
print("💻 SYSTEM RESOURCES CHECK")
print("=" * 60)

# CPU
cpu_count = os.cpu_count()
cpu_percent = psutil.cpu_percent(interval=1)
print(f"\n🖥️  CPU:")
print(f"  Cores: {cpu_count}")
print(f"  Usage: {cpu_percent}%")
print(f"  Recommended workers: {min(cpu_count, 8)}")

# RAM
ram = psutil.virtual_memory()
print(f"\n💾 RAM:")
print(f"  Total: {ram.total / 1e9:.1f} GB")
print(f"  Available: {ram.available / 1e9:.1f} GB")
print(f"  Usage: {ram.percent}%")

# GPU
if torch.cuda.is_available():
    print(f"\n🎮 GPU:")
    print(f"  Name: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Current memory usage
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved: {reserved:.2f} GB")

print("\n" + "=" * 60)
print("OPTIMIZATION READINESS")
print("=" * 60)

# Recommendations
warnings = []
if cpu_count < 4:
    warnings.append("⚠️  Low CPU count - use 4 workers instead of 8")
if ram.available < 10e9:
    warnings.append("⚠️  Low RAM - might affect 8 workers")
if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):
    warnings.append("⚠️  Not on A100 - keep gradient_checkpointing=True")

if warnings:
    print("\n⚠️  Warnings:")
    for w in warnings:
        print(f"  {w}")
else:
    print("\n✅ All checks passed! Ready for optimized training.")
    print(f"\n🚀 Recommended config:")
    print(f"  - dataloader_num_workers: {min(cpu_count, 8)}")
    print(f"  - per_device_train_batch_size: 6")
    print(f"  - gradient_checkpointing: False")
    print(f"  - dataloader_prefetch_factor: 4")
```

---

### STEP 4: Stop Current Training ⏸️

**ONLY after checkpoint-1000 is saved!**

In Colab:
1. Click **Runtime → Interrupt execution**
2. OR press **Ctrl+M** then **I**

Wait for:
```
KeyboardInterrupt
Training interrupted.
```

---

### STEP 5: Upload Optimized Script 📤

The optimized script is already created at:
`train_qlora_optimized.py`

Upload it to Colab:

```python
# In Colab, run this to create the optimized script
!wget https://raw.githubusercontent.com/dkeviv/Cogumi-LLM/main/train_qlora_optimized.py

# Or copy from your local repo
from google.colab import files
uploaded = files.upload()  # Upload train_qlora_optimized.py
```

---

### STEP 6: Test Load Checkpoint (DRY RUN) 🧪

**CRITICAL SAFETY CHECK** - Verify checkpoint loads before full restart:

```python
# Test loading the checkpoint
from transformers import TrainingArguments, Trainer
import torch

print("🧪 Testing checkpoint loading...")

# Quick dummy trainer to test
test_args = TrainingArguments(
    output_dir="./test",
    per_device_train_batch_size=1,
)

try:
    # This will validate the checkpoint can be loaded
    test_trainer = Trainer(args=test_args)
    state = test_trainer._load_from_checkpoint(
        "data/checkpoints/llama-3.1-8b-phase1a/checkpoint-1000"
    )
    
    print(f"✅ Checkpoint valid!")
    print(f"  Current step: {state.global_step}")
    print(f"  Current epoch: {state.epoch:.4f}")
    print(f"  Loss: {state.log_history[-1].get('loss', 'N/A')}")
    
except Exception as e:
    print(f"❌ ERROR loading checkpoint:")
    print(f"  {e}")
    print("\n⚠️  DO NOT PROCEED! Checkpoint is corrupted.")
```

**Expected output:**
```
✅ Checkpoint valid!
  Current step: 1000
  Current epoch: 0.0498
  Loss: 1.234
```

---

### STEP 7: Start Optimized Training 🚀

**ONLY if Step 6 passed!**

Run the optimized training:

```python
!python train_qlora_optimized.py --resume_from_checkpoint data/checkpoints/llama-3.1-8b-phase1a/checkpoint-1000
```

**Watch for:**
```
🔄 Resuming training from checkpoint: data/checkpoints/llama-3.1-8b-phase1a/checkpoint-1000
✅ Checkpoint found

🚀 OPTIMIZED TRAINING CONFIGURATION
================================================================================
Per-device batch size: 6
Gradient accumulation: 6
Effective batch size: 36
Gradient checkpointing: False
Dataloader workers: 8
Prefetch factor: 4
================================================================================

Loading tokenizer...
Loading model...
...
🚀 STARTING OPTIMIZED TRAINING
Expected speed: 5-6 seconds/step (vs 8.86s/step before)
```

---

### STEP 8: Monitor Performance 📊

After ~50 steps, check the speed:

```python
# Look for this in the training output:
# 1050/60060 [XX:XX<YY:YY, 5.23s/it]  ← Should be 5-6s, not 8.86s!
```

**If speed is still 8-9s/it:**
- GPU might not have enough memory for batch_size=6
- Fall back to batch_size=4 (edit train_qlora_optimized.py)

**If speed is 5-6s/it:**
- ✅ Optimization successful!
- ✅ Expected completion: ~90 hours from step 1000

---

## 🎯 Expected Timeline

| Phase | Steps | Time | Cumulative |
|-------|-------|------|------------|
| Before optimization | 0-1000 | 2.5h | 2.5h |
| After optimization | 1000-60060 | 82h | 84.5h |
| **Total** | **60060** | **~85h** | **~3.5 days** |

Compare to original: **148 hours (6 days)** → Save **63 hours!** ⚡

---

## 🆘 Troubleshooting

### Issue: "CUDA out of memory"

**Solution:** Reduce batch size
```python
# Edit train_qlora_optimized.py, line 58:
per_device_train_batch_size=4,  # Down from 6
gradient_accumulation_steps=8,  # Up from 6
```

### Issue: "Too many open files"

**Solution:** Reduce workers
```python
# Edit train_qlora_optimized.py, line 79:
dataloader_num_workers=4,  # Down from 8
```

### Issue: Speed still slow (8-9s/it)

**Possible causes:**
1. Gradient checkpointing still enabled (check line 52)
2. CPU bottleneck (reduce workers to 4-6)
3. Colab throttling (check GPU utilization: `!nvidia-smi`)

### Issue: Checkpoint doesn't exist

**Solution:** Wait longer, check save_steps=1000 in logs

---

## ✅ Success Criteria

After optimization, you should see:
- ✅ Speed: 5-6 seconds/step (down from 8.86s)
- ✅ GPU utilization: 95-100%
- ✅ Training loss continues smoothly from checkpoint
- ✅ No OOM errors
- ✅ ETA: ~90 hours (down from 148 hours)

---

## 📞 Need Help?

If anything goes wrong:
1. **Don't panic** - your checkpoint is safe
2. Stop training (Ctrl+M → I)
3. Re-run Step 6 (test checkpoint loading)
4. Fall back to original `train_qlora.py` if needed

You can always resume from checkpoint-1000 with the original slower config - **no training progress is lost!**
