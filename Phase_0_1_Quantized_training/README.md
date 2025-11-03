# Phase 1A 1.0 - DEPRECATED QLoRA Training (ARCHIVED)

**Status:** ⚠️ **DEPRECATED - DO NOT USE** ⚠️  
**Date Archived:** November 3, 2025  
**Reason:** Catastrophic merge corruption due to training on 4-bit quantized base

---

## ❌ Why This Approach Was Abandoned

**Critical Issue:** Training LoRA adapters on a 4-bit quantized base model caused severe model corruption during the merge step, resulting in:
- 42% drop in tie rate (70% expected → 28% actual)
- 36% drop in code performance
- Completely broken baseline performance

**Root Cause:** The adapter learned to compensate for quantization artifacts in the 4-bit base. When merged, the 4-bit rounding errors corrupted the full-precision output.

**Architecture Error:**
```
❌ WRONG:  Train LoRA on 4-bit base → Merge (rounding errors) → Corrupted model
✅ CORRECT: Train LoRA on full precision base → Merge → Clean model → Quantize (Phase 2)
```

---

## 📁 Archived Files

### 1. `train_qlora_optimized.py`
- **Purpose:** Phase 1A base training using QLoRA 4-bit quantization
- **Base Model:** `unsloth/meta-llama-3.1-8b-instruct-unsloth-bnb-4bit` (quantized)
- **Training:** 3 epochs, 600K examples, A100 40GB
- **Cost:** $220 (120 GPU-hours)
- **Output:** Corrupted merged model
- **Status:** DEPRECATED - caused merge corruption

### 2. `requirements-h100-training.txt`
- **Purpose:** Dependencies for QLoRA training
- **PyTorch:** 2.4.0+cu121
- **CUDA:** 12.1
- **Key Package:** unsloth (auto-installs bitsandbytes for 4-bit)
- **Status:** DEPRECATED - use Phase1A_2_0 requirements instead

---

## ✅ Replacement: Phase 1A 2.0 (Full Precision)

**Location:** `/Phase1A_2_0/`

**Key Differences:**
- ✅ **Base:** Full precision bfloat16 (NOT 4-bit quantized)
- ✅ **PyTorch:** 2.6.0+cu124
- ✅ **CUDA:** 12.4
- ✅ **Flash Attention:** 2.7.4 pre-compiled
- ✅ **Training:** 8-12 hours on H100 80GB
- ✅ **Cost:** $20-30 (vs $220)
- ✅ **Output:** Clean 15GB merged model
- ✅ **Performance:** No corruption, proper baseline

**Requirements:** `/Phase1A_2_0/scripts/requirements-stable-precompiled.txt`

---

## 📊 Performance Comparison

| Metric | Phase 1A 1.0 (QLoRA) | Phase 1A 2.0 (Full Precision) |
|--------|---------------------|-------------------------------|
| **Base Model** | 4-bit quantized | bfloat16 full precision |
| **Training Time** | 90 hours | 8-12 hours |
| **Cost** | $220 | $20-30 |
| **MATH Ties** | 28% (broken) | 70% (expected) |
| **CODE Wins** | 12% (broken) | 48% (expected) |
| **Merge Status** | Corrupted | Clean |
| **Status** | DEPRECATED | ✅ CURRENT |

---

## 🔍 Investigation Timeline (October 2025)

1. **Day 1-3:** Phase 1A 1.0 training completed
2. **Day 4:** Phase 1B training showed catastrophic forgetting (0% wins, 78% losses)
3. **Day 5:** Filtered training data, still failed
4. **Day 6:** Tested Phase 1A baseline → discovered corruption
5. **Day 7:** Investigated merge process → found 4-bit quantization issue
6. **Day 8:** Verified training base → confirmed 4-bit quantized base
7. **Day 9-10:** Pivoted to Phase 1A 2.0 with full precision training
8. **Day 11-13:** Phase 1A 2.0 training successful with clean merge

---

## 🎓 Lessons Learned

1. **NEVER train on quantized base for production models**
   - Quantization should be the LAST step (Phase 2 compression)
   - Training on quantized base causes adapter to learn compensation for artifacts
   
2. **Standard fine-tuning workflow:**
   - Train on full precision base
   - Merge adapter to full precision
   - Quantize merged model (if needed)

3. **Memory vs Correctness:**
   - 4-bit training saved VRAM but destroyed model quality
   - Better to use larger GPU with full precision than save memory with quantization

4. **Unsloth's optimizations:**
   - Pre-quantized base violated standard workflow
   - Worked for Unsloth's internal use but not for merge/export

---

## 🚫 Do Not Use This Approach

This folder is kept for:
- Historical reference
- Documentation of what went wrong
- Preventing future mistakes

**For actual training:** Use `/Phase1A_2_0/` with full precision training.

**Last Updated:** November 3, 2025
