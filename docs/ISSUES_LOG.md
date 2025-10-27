# ISSUES LOG - COGUMI-LLM PROJECT

This log documents all bugs, issues, and their resolutions for the Cogumi-LLM project. Each entry includes the problem, root cause, solution, and lessons learned.

---

## 2025-10-24 - Training Crash: Disk Space Exhaustion at Step 9000

**Issue:** Training crashes at step 9,000 with safetensors serialization error
```
SafetensorError: Error while serializing: I/O error: No space left on device (os error 28)
```

**Symptoms:**
- Training runs fine until ~9,000 steps
- Crashes during checkpoint save operation
- Has happened twice in a row at same point
- Vast.ai H100 instance with 100GB local volume

**Root Cause:**
1. Checkpoint accumulation fills disk faster than expected
2. Each checkpoint = ~2.5-3GB (LoRA adapters + optimizer states + training state)
3. By step 9,000: 9 checkpoints × 3GB = **27GB+ of checkpoints**
4. Configuration has `save_total_limit=3` which should delete old checkpoints
5. **Actual problem:** Old checkpoint deletion is either:
   - Failing silently due to file locks/permissions
   - Happening too slowly (race condition)
   - Not triggering properly in Unsloth/Transformers

**Technical Details:**
- Training config: `save_steps=1000`, `save_total_limit=3`
- Expected: Max 3 checkpoints × 3GB = 9GB
- Actual: All 9 checkpoints retained = 27GB
- Disk: 100GB total, ~20GB used by dataset/model, ~27GB checkpoints = **47GB used**
- With logs, cache, and other files → **>50GB total → approaching 100GB limit**

**Solution Applied:**

### Immediate Fix (Config Changes):
```yaml
save_total_limit: 2    # Reduced from 3 → max 6GB checkpoints
save_steps: 2000       # Increased from 1000 → save less frequently
```

### Manual Cleanup Before Restart:
```bash
cd /data/Cogumi-LLM/checkpoints
LATEST=$(ls -1d checkpoint-* | sort -V | tail -1)
# Delete all except latest checkpoint
for dir in checkpoint-*; do
  [ "$dir" != "$LATEST" ] && rm -rf "$dir"
done
```

### Added Monitoring:
```python
# Disk space monitoring every 10 minutes
def monitor_disk_space():
    while True:
        result = subprocess.run(['df', '-h', '/data'], capture_output=True, text=True)
        usage = result.stdout.split('\n')[1].split()[4]
        print(f"🔍 Disk Usage: {usage} - {time.strftime('%H:%M:%S')}")
        if int(usage.strip('%')) > 90:
            print("🚨 WARNING: Disk usage >90%!")
        time.sleep(600)
```

### Optional: Checkpoint Cleanup Callback:
```python
class CheckpointCleanupCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if state.global_step % 2000 == 0:
            cleanup_old_checkpoints(args.output_dir, keep_last_n=2)

trainer.add_callback(CheckpointCleanupCallback())
```

**Files Changed:**
- `notebooks/H100_Training_Clean.ipynb` (lines 332, 320) - Update save_total_limit and save_steps
- Created: `docs/DISK_SPACE_FIX.md` - Comprehensive troubleshooting guide

**Lesson Learned:**

1. **Always monitor disk space** during long training runs (8+ hours)
2. **Don't trust save_total_limit alone** - Add explicit cleanup verification
3. **100GB volume is marginal** for 8B model training - 150GB would be safer
4. **Checkpoint size calculation:** LoRA (500MB) + Optimizer (1.5GB) + State (500MB) + Safety buffer = ~3GB per checkpoint
5. **Cost vs Safety:** +$1.20 for 150GB volume is worth avoiding 9 hours of lost training
6. **Early warning system:** Monitor disk usage >60% as yellow flag, >80% as red flag
7. **Checkpoint frequency tradeoff:** Save every 2,000 steps loses more progress if crash, but reduces disk pressure
8. **Test on small run first:** A 2-hour test run would have revealed this issue early

**Prevention Guidelines:**

**For Future Training:**
- [ ] Always start training with <30% disk usage
- [ ] Set `save_total_limit=2` as default (not 3)
- [ ] Add disk monitoring to all training notebooks
- [ ] For 8B models: Use 150GB+ volumes on Vast.ai
- [ ] Check `df -h` after every checkpoint save in first hour
- [ ] Add explicit cleanup callback as safety net
- [ ] Document disk requirements in training docs

**Red Flags to Watch:**
- ⚠️  Disk usage >60% during training
- ⚠️  More than `save_total_limit` checkpoints exist
- ⚠️  Checkpoint sizes unexpectedly large (>5GB)
- ⚠️  Deletion logs not appearing in training output

**Related Guidelines:**
- `docs/DISK_SPACE_FIX.md` - Full troubleshooting guide
- `docs/H100_QUICK_REFERENCE.md` - Training workflow
- `docs/EXECUTION_PLAN.md` - Phase 1 training specifications

**Status:** ✅ RESOLVED - Config updated, cleanup performed, monitoring added

**Validation:**
- [ ] Training restarts successfully from checkpoint-9000
- [ ] Disk usage stays <50% throughout remaining training
- [ ] Only 2 checkpoints exist at any time
- [ ] Training completes all 28,000 steps without crashes

---

## Template for Future Issues

## YYYY-MM-DD - Issue Title

**Issue:** User-visible problem description

**Symptoms:**
- Bullet points of what user/system observed

**Root Cause:** 
Technical explanation with code/config snippets

**Solution Applied:**
Concrete steps taken to fix the issue

**Files Changed:** 
List with line numbers

**Lesson Learned:**
Key takeaways for future

**Related Guidelines:** 
Links to relevant docs

**Status:** ✅ RESOLVED / ⏳ IN PROGRESS / ❌ BLOCKED
