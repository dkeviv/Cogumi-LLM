# QUICK START: Parallel Tasks During Training

**Training Status:** Phase 1 running on Colab (36-48 hours)  
**Your Goal:** Be ready for Phase 2 compression the moment training finishes

---

## 🚀 START NOW (Takes 30 minutes)

### Step 1: Setup Phase 2 Environment
```bash
cd /Users/vivekdurairaj/Projects/Cogumi-LLM
./scripts/setup_phase2_compression.sh
```

**What it does:**
- Installs Neural Magic SparseML
- Installs AutoAWQ
- Clones and builds llama.cpp
- Installs compression utilities

**Time:** 15-20 minutes  
**Critical:** Must complete before training finishes

---

### Step 2: Verify Installation
```bash
python -c "import sparseml; print('✅ SparseML installed')"
python -c "import awq; print('✅ AutoAWQ installed')"
python -c "import zstandard; print('✅ Zstd installed')"
ls tools/llama.cpp/main && echo "✅ llama.cpp built"
```

**All passing?** → You're ready for Phase 2!

---

## 📊 MONITOR TRAINING (Check every 6-8 hours)

### Check TensorBoard
1. Open TensorBoard URL from Colab
2. Verify loss is decreasing
3. Screenshot loss curve

### Check Colab Session
1. Is session still connected?
2. Any error messages?
3. Are checkpoints saving? (every 1000 steps)

### GPU Utilization
```python
# Run in Colab cell:
!nvidia-smi
```
**Target:** >80% utilization

---

## 📝 DOCUMENT PROGRESS (15 min every 6h)

Create `docs/PHASE1_TRAINING_LOG.md`:

```markdown
## Training Session

**Start Time:** [timestamp]
**Hardware:** A100 40GB / 80GB
**Dataset:** 640,637 samples

### Hour 0-6
- Loss: [value]
- Speed: [samples/sec]
- Checkpoint: checkpoint-[number]

### Hour 6-12
- Loss: [value]
- ...
```

---

## ⏰ TIMELINE: What to Do When

### **Now (Start of training)**
- ✅ Run `setup_phase2_compression.sh`
- ✅ Verify installations
- ✅ Create training log document

### **Hour 6**
- Check TensorBoard
- Document loss value
- Screenshot curve

### **Hour 12**
- Check session is still connected
- Verify checkpoints are saving
- Update training log

### **Hour 18**
- Check loss convergence
- Monitor GPU utilization
- Take screenshot

### **Hour 24**
- Check if Colab session timed out
- If yes: resume from latest checkpoint
- If no: continue monitoring

### **Hour 30-36**
- Training should be near completion
- Prepare to download checkpoint
- Have benchmark scripts ready

### **Hour 36-48 (Completion)**
- Download best checkpoint
- Run benchmarks
- **START PHASE 2 IMMEDIATELY**

---

## 🎯 PHASE 2 READINESS CHECKLIST

Before training completes, ensure:

- [ ] ✅ SparseML installed
- [ ] ✅ AutoAWQ installed
- [ ] ✅ llama.cpp built
- [ ] ✅ Zstandard installed
- [ ] ✅ Training progress documented
- [ ] ✅ Know which checkpoint is best
- [ ] ✅ Have 512 calibration samples ready
- [ ] ✅ Benchmark scripts prepared

**All checked?** → Ready for Phase 2! 🚀

---

## 🚨 TROUBLESHOOTING

### "Training loss not decreasing"
- Wait 1000 steps to verify
- If still flat: may need to adjust learning rate
- Check: Is data loading properly?

### "Colab session disconnected"
- Find latest checkpoint: `ls -lh data/checkpoints/*/checkpoint-*`
- Modify `train_qlora.py` to resume from checkpoint
- Rerun training cell

### "OOM Error"
- Reduce `per_device_train_batch_size` from 4 to 2
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Restart runtime and rerun

### "Phase 2 install fails"
- Check Python version (need 3.8+)
- Try: `pip install --upgrade pip setuptools wheel`
- Install packages one by one if batch fails

---

## 📚 REFERENCE DOCS

- **Full parallel tasks:** `docs/PARALLEL_TASKS_DURING_TRAINING.md`
- **Execution plan:** `docs/EXECUTION_PLAN.md`
- **Compression strategy:** `docs/ENGLISH_ONLY_COMPRESSION_STRATEGY.md`
- **Current status:** `docs/CURRENT_STATUS.md`

---

## 💡 ONE-LINE SUMMARY

**Setup Phase 2 environment NOW → Monitor training every 6h → Be ready to compress immediately after training finishes!**
