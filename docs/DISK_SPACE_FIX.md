# DISK SPACE FIX - VAST.AI H100 TRAINING

**Issue:** Training crashes at ~9,000 steps with error:
```
SafetensorError: Error while serializing: I/O error: No space left on device (os error 28)
```

**Root Cause:** Checkpoint accumulation fills 100GB local volume disk

**Status:** CRITICAL - Must fix immediately before restarting training

---

## 🔍 DIAGNOSIS

### What's Happening:
1. Training saves checkpoints every 1,000 steps
2. Each checkpoint = ~2.5-3GB (LoRA adapters + optimizer state)
3. By step 9,000 → 9 checkpoints × 3GB = **~27GB**
4. But `save_total_limit=3` should delete old checkpoints...
5. **Problem:** Deleting old checkpoints is failing or not happening

### Why It Failed Twice:
- First crash at 9,000 steps → Lost all progress
- Restarted → Crashed again at 9,000 steps
- Pattern suggests systematic issue with checkpoint cleanup

---

## ✅ IMMEDIATE SOLUTION (3 Options)

### Option 1: Aggressive Checkpoint Cleanup (RECOMMENDED)

**Change in training config:**
```yaml
save_total_limit: 2  # Keep ONLY 2 most recent (down from 3)
save_steps: 2000     # Save less frequently (was 1000)
```

**Why this works:**
- 2 checkpoints × 3GB = **6GB max** (safe buffer)
- Saves every 2,000 steps instead of 1,000
- Still have recovery point every ~5 hours of training

**Risks:** Lose more progress if crash (2,000 steps vs 1,000)
**Benefits:** Much safer on disk space

---

### Option 2: Manual Checkpoint Cleanup Script

Add this to your training notebook/script:

```python
import os
import glob
import shutil

def cleanup_old_checkpoints(checkpoint_dir, keep_last_n=2):
    """Manually delete old checkpoints to free space."""
    checkpoints = sorted(glob.glob(f"{checkpoint_dir}/checkpoint-*"))
    
    if len(checkpoints) > keep_last_n:
        to_delete = checkpoints[:-keep_last_n]
        for cp in to_delete:
            print(f"🗑️  Deleting: {cp}")
            shutil.rmtree(cp)
            size_freed = sum(os.path.getsize(f) for f in glob.glob(f"{cp}/**", recursive=True)) / 1e9
            print(f"   ✅ Freed {size_freed:.2f}GB")

# Run this every N steps (e.g., every 2,000 steps)
cleanup_old_checkpoints("/data/Cogumi-LLM/checkpoints", keep_last_n=2)
```

**Add callback to trainer:**
```python
from transformers import TrainerCallback

class CheckpointCleanupCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        """Clean up old checkpoints after each save."""
        if state.global_step % 2000 == 0:  # Every 2,000 steps
            cleanup_old_checkpoints(args.output_dir, keep_last_n=2)

# Add to trainer
trainer.add_callback(CheckpointCleanupCallback())
```

---

### Option 3: Increase Vast.ai Volume Size

**Current:** 100GB local volume  
**Needed:** 150GB local volume (safe buffer)

**Cost Impact:**
- 100GB → 150GB = +50GB
- Storage cost: ~$0.15/hr extra (vs $2.50/hr GPU)
- Total run cost: +$1.20-1.50 for 8-hour run

**Why this works:** More breathing room for checkpoints

**How to change:**
1. Stop current instance
2. Create new instance with 150GB volume
3. Re-upload data and restart training

---

## 🚨 CRITICAL: Check Disk Space BEFORE Restarting

Run this in tmux or terminal:

```bash
# Check current disk usage
df -h /data

# Check checkpoint directory size
du -sh /data/Cogumi-LLM/checkpoints/*

# List all checkpoints with sizes
ls -lh /data/Cogumi-LLM/checkpoints/
```

**Expected output:**
```
Filesystem      Size  Used Avail Use% Mounted on
/dev/sda1       100G   85G   15G  85% /data  ← DANGEROUS!
```

**If "Used" > 80GB:** You're at risk of running out again!

---

## 📋 RECOMMENDED ACTION PLAN

### Step 1: Clean Up Current State (NOW)

```bash
# SSH into Vast.ai instance
cd /data/Cogumi-LLM/checkpoints

# Keep ONLY the last checkpoint (e.g., checkpoint-9000)
LATEST=$(ls -1d checkpoint-* | sort -V | tail -1)
echo "Keeping: $LATEST"

# Delete all others
for dir in checkpoint-*; do
  if [ "$dir" != "$LATEST" ]; then
    echo "Deleting: $dir"
    rm -rf "$dir"
  fi
done

# Check freed space
df -h /data
```

### Step 2: Update Training Config (CRITICAL)

**Edit your training script or notebook:**

```python
training_args = UnslothTrainingArguments(
    output_dir="/data/Cogumi-LLM/checkpoints",
    
    # CRITICAL CHANGES:
    save_total_limit=2,        # Was 3, now 2 (safer)
    save_steps=2000,            # Was 1000, now 2000 (less frequent)
    
    # Keep these:
    logging_steps=10,
    eval_steps=500,
    per_device_train_batch_size=12,
    gradient_accumulation_steps=1,
    num_train_epochs=3,
    # ... rest of config
)
```

### Step 3: Add Monitoring (IMPORTANT)

**Add this cell to your notebook:**

```python
import subprocess
import time

def monitor_disk_space():
    """Monitor disk space every 10 minutes."""
    while True:
        result = subprocess.run(['df', '-h', '/data'], capture_output=True, text=True)
        usage = result.stdout.split('\n')[1].split()[4]  # Get "85%" from output
        
        print(f"🔍 Disk Usage: {usage} - {time.strftime('%H:%M:%S')}")
        
        if int(usage.strip('%')) > 90:
            print("🚨 WARNING: Disk usage >90%! Risk of crash!")
        
        time.sleep(600)  # 10 minutes

# Run in background thread
import threading
monitor_thread = threading.Thread(target=monitor_disk_space, daemon=True)
monitor_thread.start()
```

### Step 4: Resume Training

```bash
# In tmux session
cd /data/Cogumi-LLM
python train.py  # Will auto-resume from latest checkpoint
```

---

## 🔍 VALIDATION CHECKLIST

Before resuming training, verify:

- [ ] Disk usage < 30GB (df -h /data shows <30% used)
- [ ] Only 1-2 checkpoints remain in /data/Cogumi-LLM/checkpoints/
- [ ] `save_total_limit=2` in training config
- [ ] `save_steps=2000` in training config (or add cleanup callback)
- [ ] Monitoring script running in background
- [ ] tmux session active and attached

**Expected Disk Usage During Training:**
- Start: 20-30GB (dataset + model)
- Peak: 35-40GB (with 2 checkpoints)
- **Never exceed:** 50GB (safe threshold)

---

## 📊 WHY THIS HAPPENS

### Unsloth/Transformers Checkpoint Behavior:

1. **Save new checkpoint** (step N) → +3GB
2. **Check save_total_limit=3** → Should keep last 3
3. **Delete oldest checkpoint** → Frees 3GB
4. **BUT:** If deletion fails silently, disk fills up

### Common Deletion Failures:
- File locks from other processes
- Permission issues
- Slow deletion on network storage
- Race conditions during save/delete

### Why save_total_limit=2 helps:
- More aggressive cleanup
- Less room for race conditions
- Guarantees max 6GB instead of 9GB

---

## 🎯 LONG-TERM SOLUTION

### For Future Training Runs:

**Option A: Larger Volume (Safest)**
- Use 150GB volume instead of 100GB
- Cost: +$0.15/hr = +$1.20 total
- Zero risk of disk issues

**Option B: Checkpoint to External Storage**
- Save checkpoints to cloud (GCS, S3, HuggingFace Hub)
- Delete local checkpoints immediately after upload
- Only keep latest checkpoint locally

**Option C: Aggressive Cleanup + Monitoring**
- save_total_limit=2
- save_steps=2000
- Add CheckpointCleanupCallback
- Monitor disk every 10 minutes

---

## 🚀 RECOVERY STRATEGY

### If Training Crashes Again:

1. **Immediately check disk:**
   ```bash
   df -h /data
   du -sh /data/Cogumi-LLM/checkpoints/*
   ```

2. **Find latest valid checkpoint:**
   ```bash
   cd /data/Cogumi-LLM/checkpoints
   ls -lt checkpoint-* | head -1
   ```

3. **Delete all except latest:**
   ```bash
   LATEST=$(ls -1d checkpoint-* | sort -V | tail -1)
   for dir in checkpoint-*; do
     [ "$dir" != "$LATEST" ] && rm -rf "$dir"
   done
   ```

4. **Verify checkpoint integrity:**
   ```bash
   ls -lh $LATEST/
   # Should contain: adapter_model.safetensors, adapter_config.json, trainer_state.json
   ```

5. **Resume training** → Starts from latest checkpoint

---

## 💡 PREVENTION TIPS

### Monitor These Metrics:

1. **Disk Usage:** `df -h /data` → Should stay <50%
2. **Checkpoint Count:** `ls /data/Cogumi-LLM/checkpoints/ | wc -l` → Should be ≤2
3. **Checkpoint Sizes:** `du -sh /data/Cogumi-LLM/checkpoints/checkpoint-*`

### Red Flags:
- ⚠️  Disk usage >60% → Clean up now
- ⚠️  More than 3 checkpoints exist → Manual cleanup needed
- ⚠️  Checkpoint sizes growing >5GB → Model bloat issue

### Green Flags:
- ✅ Disk usage stable around 30-40%
- ✅ Only 2 checkpoints present
- ✅ Old checkpoints being deleted automatically

---

## 📞 DEBUGGING COMMANDS

```bash
# Check what's using disk space
du -sh /data/* | sort -h

# Check checkpoint directory
du -sh /data/Cogumi-LLM/checkpoints/*

# Monitor disk in real-time (every 5 seconds)
watch -n 5 df -h /data

# Check if training process is running
ps aux | grep train

# Check tmux sessions
tmux ls

# Attach to training tmux
tmux attach -t training

# View training logs
tail -f /data/Cogumi-LLM/logs/training.log  # If logging to file

# Emergency: Kill training if frozen
pkill -f train.py
```

---

## ✅ SUCCESS CRITERIA

**Training is healthy when:**
- ✅ Disk usage stays <50% throughout training
- ✅ Only 2 checkpoints exist at any time
- ✅ Old checkpoints deleted within 1 minute of new save
- ✅ Training completes all 28,000 steps without crashes
- ✅ Final model saved successfully to `/data/Cogumi-LLM/checkpoints/final`

---

## 🎯 NEXT STEPS

1. **Clean up now** (Step 1 above)
2. **Update config** (save_total_limit=2, save_steps=2000)
3. **Add monitoring** (disk space tracker)
4. **Resume training** (auto-resumes from last checkpoint)
5. **Watch closely** for first 2 hours to ensure no issues
6. **Commit changes** to prevent future occurrences

**Estimated time to fix:** 15-30 minutes  
**Estimated impact:** Zero data loss (have checkpoint-9000)  
**Cost impact:** $0 (just config changes)  
**Training restart:** Within 1 hour
