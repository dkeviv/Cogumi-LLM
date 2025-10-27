# IMMEDIATE ACTION: Fix Disk Space Crash

**Status:** Training crashed at step 9000 due to disk exhaustion  
**Time to fix:** 15-30 minutes  
**Data loss:** None (have checkpoint-9000)

---

## ⚡ DO THIS NOW (In Order)

### 1. Clean Up Disk Space (5 minutes)

```bash
# SSH into Vast.ai instance
# Check current disk usage
df -h /data

# Navigate to checkpoints
cd /data/Cogumi-LLM/checkpoints

# List all checkpoints with sizes
ls -lh

# Keep ONLY the latest checkpoint
LATEST=$(ls -1d checkpoint-* | sort -V | tail -1)
echo "Keeping: $LATEST"

# Delete all others
for dir in checkpoint-*; do
  if [ "$dir" != "$LATEST" ]; then
    echo "Deleting: $dir"
    rm -rf "$dir"
  fi
done

# Verify cleanup
df -h /data
# Should show <30GB used now
```

### 2. Update Training Config (5 minutes)

**Find your training script/notebook and change:**

```python
# OLD (causes crash):
save_total_limit=3
save_steps=1000

# NEW (prevents crash):
save_total_limit=2
save_steps=2000
```

**Specific locations:**
- If using H100_Training_Clean.ipynb: Cell 19, line 332
- If using train.py: Line ~50-60 in TrainingArguments

### 3. Add Disk Monitoring (5 minutes)

**Add this to your notebook/script:**

```python
import subprocess
import time
import threading

def monitor_disk():
    """Monitor disk space every 10 minutes."""
    while True:
        result = subprocess.run(['df', '-h', '/data'], 
                              capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            usage = lines[1].split()[4]
            used = lines[1].split()[2]
            print(f"💾 Disk: {used} used ({usage}) - {time.strftime('%H:%M:%S')}")
            
            usage_pct = int(usage.strip('%'))
            if usage_pct > 90:
                print("🚨 CRITICAL: Disk >90%!")
            elif usage_pct > 70:
                print("⚠️  Warning: Disk >70%")
        
        time.sleep(600)  # 10 minutes

# Start monitoring in background
monitor_thread = threading.Thread(target=monitor_disk, daemon=True)
monitor_thread.start()
print("✅ Disk monitoring started")
```

### 4. Verify Before Restarting (2 minutes)

**Run these checks:**

```bash
# Check 1: Disk usage
df -h /data
# Should show <30% used

# Check 2: Checkpoint count
ls /data/Cogumi-LLM/checkpoints/checkpoint-* | wc -l
# Should show 1 or 2

# Check 3: Latest checkpoint integrity
ls -lh /data/Cogumi-LLM/checkpoints/checkpoint-9000/
# Should have: adapter_model.safetensors, adapter_config.json, trainer_state.json

# Check 4: Config updated
grep -n "save_total_limit\|save_steps" /data/Cogumi-LLM/train.py
# Should show save_total_limit=2 and save_steps=2000
```

### 5. Restart Training (1 minute)

```bash
# In tmux session (or create new one)
tmux new -s training

# Navigate to project
cd /data/Cogumi-LLM

# Start training (will auto-resume from checkpoint-9000)
python train.py

# Detach from tmux: Ctrl+B, then D
```

---

## ✅ Validation Checklist

After restarting, monitor for 30 minutes:

- [ ] Training resumes from step 9000
- [ ] Disk usage shown every 10 minutes
- [ ] Disk usage stays <40%
- [ ] Step 10,000 checkpoint saves successfully
- [ ] Old checkpoint (9000) deleted automatically
- [ ] Only 1-2 checkpoints exist

**If all checks pass:** ✅ Training is healthy, let it run

**If any check fails:** ⚠️ Stop training and review DISK_SPACE_FIX.md

---

## 🔍 Monitor These Commands

**During training (check every 30 minutes):**

```bash
# Disk usage
df -h /data

# Checkpoint count
ls /data/Cogumi-LLM/checkpoints/ | wc -l

# Training progress (in tmux)
tmux attach -t training
# Then Ctrl+B, D to detach
```

**Expected output:**
```
Disk: 35-45% used (safe range)
Checkpoints: 2 (occasionally 3 during save, then back to 2)
Training: Step increasing every ~3 seconds
```

---

## 🚨 Emergency Commands

**If disk fills up again:**

```bash
# Stop training immediately
pkill -f train.py

# Emergency cleanup (keeps only latest)
cd /data/Cogumi-LLM/checkpoints
LATEST=$(ls -1d checkpoint-* | sort -V | tail -1)
find . -maxdepth 1 -name "checkpoint-*" ! -name "$LATEST" -exec rm -rf {} \;

# Check freed space
df -h /data
```

---

## 📊 Expected Timeline

| Time | Step | Disk Usage | Status |
|------|------|------------|--------|
| Now | 9,000 | 25GB | ✅ Cleaned |
| +2 hrs | 11,000 | 32GB | ✅ Checkpoint saved |
| +4 hrs | 13,000 | 32GB | ✅ Old deleted |
| +6 hrs | 15,000 | 32GB | ✅ Stable |
| +8 hrs | 17,000 | 32GB | ✅ Stable |
| ... | ... | ... | ... |
| +38 hrs | 28,000 | 35GB | ✅ Complete! |

**Red flag:** If disk usage >45GB at any point → Clean up manually

---

## 💡 Why This Works

**Old config problems:**
- save_total_limit=3 → Up to 9GB checkpoints
- save_steps=1000 → 9 saves before deletion cycle
- No monitoring → No early warning

**New config benefits:**
- save_total_limit=2 → Max 6GB checkpoints
- save_steps=2000 → Less frequent saves
- Monitoring → Early warning at 70%
- Manual cleanup → Fresh start

**Cost of changes:**
- Risk: Lose 2,000 steps (5 hours) vs 1,000 steps if crash
- Benefit: No disk space crashes
- Tradeoff: Acceptable (5 hrs << 20 hrs total training time)

---

## 📞 Need Help?

**If this doesn't work:**

1. Check DISK_SPACE_FIX.md for detailed troubleshooting
2. Check ISSUES_LOG.md for this exact issue documentation
3. Option B: Increase volume to 150GB (costs +$1.20 total)
4. Option C: Switch to checkpoint uploading strategy

**Files to reference:**
- `docs/DISK_SPACE_FIX.md` - Full guide
- `docs/ISSUES_LOG.md` - This exact issue
- `docs/H100_QUICK_REFERENCE.md` - Training basics

---

## ✅ Success = Training Completes Without Crashes

**You'll know it worked when:**
- Training runs from step 9,000 → 28,000 (19 hours)
- Disk usage stays stable 30-40%
- Final model saved successfully
- Ready for Phase 1B benchmarking!

**Good luck! 🚀**
