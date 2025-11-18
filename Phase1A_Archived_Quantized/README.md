# Phase 1A - Archived Quantized Training (DEPRECATED)

**Status:** ⚠️ **ARCHIVED - DO NOT USE** ⚠️  
**Date Archived:** November 9, 2025  
**Reason:** Merge corruption due to training on 4-bit quantized base

---

## ⚠️ This Folder Contains DEPRECATED Content

This folder consolidates two failed attempts at Phase 1A training:
1. **Phase1_0_Quantized_training** - Original QLoRA on 4-bit base
2. **Phase1A_Base_Training** - Incomplete reorganization attempt

Both approaches are **ARCHIVED** and should **NOT be used** for any new work.

---

## ❌ Critical Issues

**Problem:** Training LoRA adapters on a 4-bit quantized base model caused severe corruption during merge:
- ✅ **During training**: Adapter worked well (~70% MATH ties, ~48% CODE wins)
- ❌ **After merge**: Catastrophic failure (28% MATH ties, 12% CODE wins)
- 🔧 **Root cause**: INT4→FP16 conversion during merge introduced rounding errors

**Why Adapter Worked But Merge Failed:**
- Adapter learned on INT4 quantized weights (compensated for artifacts)
- Merge dequantized INT4 → FP16 (introduced rounding errors)
- Adapter compensations misaligned with new float weights
- Result: Garbage model output

---

## 📁 Folder Structure

### Main Files
- `README_QUANTIZED_ARCHIVED.md` - Original detailed explanation of failure
- `requirements-h100-training.txt` - Dependencies for quantized training (deprecated)

### `/scripts/`
- `train_qlora_optimized.py` - QLoRA training script (DEPRECATED)
- `merge_lora_adapter.py` - Merge script (caused corruption)
- `merge_adapter_fullprecision.py` - Attempted fix (incomplete)
- `train_phase1a_optimized_h100.py` - H100 training (never fully working)

### `/data/`
- `public_500k_filtered.jsonl` - Original 500K training dataset
- `checkpoints/` - Failed training checkpoints

### `/docs/`
- Various quickstart guides and troubleshooting docs
- **Note:** These docs reference the failed approach

### `/phase1_0_legacy/`
- `benchmark_results_full/` - Benchmark results showing corruption
- `scored/` - Quality-scored training data
- `training_from_benchmark/` - Failure analysis data
- `public_500k_filtered.jsonl` - Training dataset (duplicate)

### `/models/`
- Empty (models were never successfully merged)

---

## ✅ Replacement: Phase1A_2_0 (ACTIVE)

**Location:** `/Phase1A_2_0/`

**Key Differences:**
| Aspect | Phase1A Archived (This Folder) | Phase1A_2_0 (Current) |
|--------|--------------------------------|------------------------|
| **Base Model** | 4-bit quantized (INT4) | Full precision (FP16) ✅ |
| **Training Time** | 90 hours | 8-12 hours |
| **Cost** | $220 | $20-30 |
| **MATH Benchmark** | 28% ties (broken) | 70% ties (expected) |
| **CODE Benchmark** | 12% wins (broken) | 48% wins (expected) |
| **Merge Status** | ❌ Corrupted | ✅ Clean |
| **Model Size** | N/A (failed) | 15GB merged |
| **Performance** | N/A (unusable) | 89-91% GPT-4 |

---

## 🎓 Key Lessons Learned

1. **NEVER train on quantized base for production models**
   - Quantization should be the LAST step (Phase 2 compression)
   - Training on quantized base causes adapter to learn compensation for artifacts

2. **LoRA adapters CAN work on quantized bases during training**
   - Training metrics looked good
   - Validation appeared successful
   - **BUT** merge step introduces fatal corruption

3. **Correct workflow:**
   ```
   ✅ Train on FP16/BF16 full precision
   ✅ Merge in full precision
   ✅ Quantize AFTER merge (if needed)
   
   ❌ Train on INT4 quantized
   ❌ Merge causes corruption
   ❌ Unusable output
   ```

4. **Type mismatch = corruption:**
   - INT4 + Float16 = Rounding errors
   - Float16 + Float16 = Clean merge

---

## 🔍 Investigation Timeline

- **October 17-19, 2025:** Phase 1A 1.0 training on quantized base
- **October 20, 2025:** Phase 1B training showed catastrophic forgetting
- **October 21, 2025:** Discovered Phase 1A baseline corruption
- **October 22-24, 2025:** Root cause analysis (4-bit quantization issue)
- **October 25-27, 2025:** Attempted reorganization (Phase1A_Base_Training)
- **October 28-30, 2025:** Pivoted to Phase1A_2_0 with full precision
- **October 31-November 2, 2025:** Successful Phase1A_2_0 training
- **November 9, 2025:** Consolidated failed attempts into this archive

---

## 📚 For Historical Reference Only

This folder is preserved for:
- Understanding what went wrong
- Documenting failed approaches
- Preventing repeated mistakes
- Benchmark data analysis

**DO NOT USE THIS CODE FOR NEW TRAINING!**

Use **Phase1A_2_0** instead: `/Phase1A_2_0/`

---

## Quick Links

- ✅ **Current Phase1A:** [/Phase1A_2_0/README.md](../Phase1A_2_0/README.md)
- 📖 **Detailed Failure Analysis:** [README_QUANTIZED_ARCHIVED.md](./README_QUANTIZED_ARCHIVED.md)
- 🎯 **Phase1C (Next Step):** [/Phase1C_Targeted_Distillation/](../Phase1C_Targeted_Distillation/)
