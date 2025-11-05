# Phase 2 Compression - Quick Start Guide

**Status:** Ready to start after Phase 1 training completes  
**Location:** Use Colab (not local)  
**Notebook:** `notebooks/Phase2_Compression_Colab.ipynb`

---

## 🎯 **Prerequisites**

Before starting Phase 2:

- ✅ Phase 1 training complete (loss ~0.5-0.7)
- ✅ Model merged and saved (LoRA → full model)
- ✅ Model uploaded to HuggingFace OR Google Drive
- ✅ Have calibration dataset (512+ samples)

---

## 🚀 **Quick Start (5 Steps)**

### **Step 1: Upload Phase 2 Notebook to Colab**

1. Go to https://colab.research.google.com
2. Click **File → Upload notebook**
3. Upload: `Cogumi-LLM/notebooks/Phase2_Compression_Colab.ipynb`

### **Step 2: Select A100 GPU**

1. **Runtime → Change runtime type**
2. Select: **A100 GPU**
3. Click **Save**

### **Step 3: Upload Your Phase 1 Model**

Choose ONE option:

**Option A: From HuggingFace (Recommended)**
- Upload Phase 1 model to HuggingFace Hub
- Use model ID in notebook: `YOUR_USERNAME/cogumi-llm-phase1a`
- Fastest and most reliable

**Option B: From Google Drive**
- Upload Phase 1 model folder to Drive
- Mount Drive in Colab
- Load from Drive path

**Option C: From Local**
- Compress Phase 1 model: `tar -czf model.tar.gz models/llama-3.1-8b-phase1a-merged/`
- Upload to Colab Files panel (slow: ~30 min for 11GB)

### **Step 4: Run All Cells**

1. **Section 1-2**: Environment setup (10 min)
2. **Section 3-4**: Load model + calibration data (15 min)
3. **Section 5**: Phase 2A - Pruning (5-6 hours)
4. **Section 6**: Phase 2B - Quantization (2-3 hours)
5. **Section 7**: Phase 2C - GGUF Export (1 hour)

**Total time: 8-10 hours**

### **Step 5: Download Compressed Model**

1. Run Section 9: Download cell
2. Save as: `cogumi-llm-480mb.gguf.zst`
3. Decompress locally: `zstd -d cogumi-llm-480mb.gguf.zst`

---

## ⏱️ **Timeline & Checkpoints**

| Phase | Duration | Output | Size | Action |
|-------|----------|--------|------|--------|
| **Setup** | 10 min | Tools installed | - | Verify installations |
| **Load** | 15 min | Model loaded | 11GB | Check model size |
| **2A: Prune** | 5-6h | Pruned model | 3.85GB | Monitor progress |
| **2B: Quantize** | 2-3h | Quantized model | 1.0GB | Verify quality |
| **2C: Export** | 1h | GGUF model | 480MB | Test model |
| **Download** | 5 min | Local file | 480MB | Save to disk |

**Total: 8-10 hours**

---

## 📊 **What Happens in Each Phase**

### **Phase 2A: Neural Magic Pruning**

**What it does:**
- Removes 65% of neurons (vs 60% for multilingual)
- Uses 2:4 structured sparsity patterns
- Optimized for CPU inference (M4 Pro, Apple Silicon)
- **Automatically removes non-English neurons!**

**How:**
```
11GB model
↓
Analyze neuron importance on English calibration data
↓
Remove weakest 65% (mostly non-English pathways)
↓
3.85GB pruned model
```

**Expected quality: 88-90% GPT-4**

---

### **Phase 2B: AWQ Quantization**

**What it does:**
- Compresses weights from 16-bit → 4-bit
- Activation-aware (preserves important weights)
- Group-wise quantization (128 groups)
- Minimal quality degradation

**How:**
```
3.85GB pruned model (16-bit weights)
↓
Analyze activation patterns
↓
Quantize less important weights to 4-bit
↓
Preserve critical weights at higher precision
↓
1.0GB quantized model
```

**Expected quality: 87-89% GPT-4**

---

### **Phase 2C: GGUF Export**

**What it does:**
- Converts to GGUF format (llama.cpp compatible)
- Applies Q5_K_M quantization
- Zstandard lossless compression
- Ready for local deployment

**How:**
```
1.0GB AWQ model
↓
Convert to GGUF format
↓
Apply Q5_K_M quantization
↓
Zstd compression (level 19)
↓
480MB final model
```

**Expected quality: 87-89% GPT-4**

---

## 🎯 **Expected Results**

### **Compression Ratio**

```
Original LLAMA-3.1-8B: 16GB (fp16)
After Phase 1 training: 11GB (English-specialized)
After Phase 2A pruning: 3.85GB (65% pruned)
After Phase 2B quantization: 1.0GB (4-bit)
After Phase 2C GGUF: 480MB (compressed)

Total reduction: 97% (16GB → 480MB)
```

### **Quality Targets**

| Benchmark | Original | Target | Expected |
|-----------|----------|--------|----------|
| **MMLU** | 82% | 78-82% | 79-81% |
| **HumanEval** | 64% | 58-62% | 60-62% |
| **GSM8K** | 90% | 86-88% | 87-89% |
| **Overall** | 100% | 87-89% | **87-89%** |

**Why this works:**
- English-only training → 65% pruning possible
- Pruning removes non-English neurons first
- AWQ preserves critical English pathways
- Net result: Smaller + Better quality for English

---

## 🚨 **Troubleshooting**

### **"Out of Memory" during pruning**

**Solution:**
- Use A100 80GB instead of 40GB
- Or: Reduce calibration samples from 512 → 256
- Or: Use gradient checkpointing

### **"Model quality degraded significantly"**

**Possible causes:**
- Calibration data not representative
- Pruning too aggressive (reduce from 65% → 60%)
- Need more calibration samples (512 → 1024)

**Solution:**
- Re-run with adjusted parameters
- Check calibration data quality

### **"Colab session timed out"**

**Solution:**
- Phase 2A/2B/2C are independent
- Save intermediate models to Drive
- Resume from last completed phase

### **"GGUF export failed"**

**Solution:**
- Check llama.cpp build succeeded
- Verify quantized model format
- Try different GGUF quantization (Q4_K_M instead of Q5_K_M)

---

## 💡 **Pro Tips**

1. **Upload Phase 1 model to HuggingFace first**
   - Faster downloads in Colab
   - Better reliability than Drive
   - Can resume if connection drops

2. **Take checkpoints between phases**
   - Save Phase 2A output before starting 2B
   - Can restart from checkpoint if issues

3. **Monitor GPU utilization**
   - Run `!nvidia-smi` periodically
   - Should be >80% during pruning/quantization
   - If low: may be data loading bottleneck

4. **Test model quality between phases**
   - Quick test after Phase 2A
   - Verify quality before continuing
   - Catch issues early

5. **Prepare calibration data in advance**
   - Upload to Drive before starting
   - Or: Create small subset locally
   - 512 samples = ~5-10MB file

---

## 📚 **Related Documentation**

- **Compression strategy**: `docs/ENGLISH_ONLY_COMPRESSION_STRATEGY.md`
- **Full pipeline**: `docs/EXECUTION_PLAN.md`
- **Current status**: `docs/CURRENT_STATUS.md`
- **Technical specs**: `docs/technical_specification.md`

---

## ✅ **Pre-Flight Checklist**

Before starting Phase 2, verify:

- [ ] Phase 1 training complete (loss <0.8)
- [ ] Model quality benchmarked (>85% GPT-4)
- [ ] Model uploaded to HuggingFace or Drive
- [ ] Calibration dataset ready (512+ samples)
- [ ] Colab Pro+ subscription active
- [ ] A100 GPU selected in Colab
- [ ] Phase 2 notebook uploaded to Colab
- [ ] HuggingFace token ready (if using HF)

**All checked?** → Ready to start Phase 2! 🚀

---

## 🎯 **After Phase 2**

Once compression completes:

1. **Download 480MB model**
2. **Benchmark quality** (MMLU, HumanEval, GSM8K)
3. **Test locally** on MacBook Air M4
4. **If quality good** (>85% GPT-4):
   - ✅ Proceed to Phase 3 (Modifiers)
5. **If quality low** (<85% GPT-4):
   - Debug: Check calibration data
   - May need to reduce pruning (65% → 60%)
   - Re-run Phase 2A with adjusted parameters

---

**Ready to compress!** Open `notebooks/Phase2_Compression_Colab.ipynb` in Colab and start! 🚀
