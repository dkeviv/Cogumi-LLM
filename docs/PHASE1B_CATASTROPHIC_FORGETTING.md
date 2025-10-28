# Phase 1B.1 Catastrophic Forgetting - Root Cause & Fix

**Date:** October 27, 2025  
**Status:** 🔴 CRITICAL BUG IDENTIFIED → ✅ FIX IMPLEMENTED

---

## 🚨 Problem Summary

**Phase 1B.1 validation showed CATASTROPHIC REGRESSION:**

```
MATH Results:
  Phase 1A baseline: 6% wins, 70% ties, 24% losses
  Phase 1B.1 (BAD):  4% wins, 18% ties, 78% losses  ❌
  
  Change: -2% wins, -52% ties, +54% losses
```

**This is worse than baseline - model FORGOT Phase 1A knowledge!**

---

## 🔍 Root Cause Analysis

### What Went Wrong:

1. **Training data included "TIE" examples (45 out of 73 = 62%)**
   - MATH: 35 ties + 12 losses = 47 total
   - CODE: 10 ties + 16 losses = 26 total

2. **"TIE" means BOTH models were correct:**
   ```json
   {
     "failure_type": "tie",
     "correct_answer": "18",
     "model_response": "Step 1: ... = $18 ✅ CORRECT",
     "judgment": "Both responses are correct... it's a tie"
   }
   ```

3. **Training on ties taught model to CHANGE correct answers:**
   - Model saw: "My answer was $18 (correct)"
   - Training signal: "This needs improvement"
   - Model learned: "I should answer differently"
   - Result: **Catastrophic forgetting of correct behavior**

### Evidence:

- **Ties collapsed:** 70% → 18% (model stopped giving correct answers it knew)
- **Losses exploded:** 24% → 78% (model gave WRONG answers instead)
- **Small improvement in wins:** 6% → 4% (even that went backwards!)

### Why This Happened:

The extraction script (`extract_failures_from_benchmark.py`) treated **any non-win as a failure**, including ties. But ties mean the model was already correct - we shouldn't train on them!

---

## ✅ Solution Implemented

### 1. Created Data Filtering Script

**File:** `scripts/filter_true_losses.py`

**What it does:**
- Removes ALL "tie" examples (model was already correct)
- Keeps ONLY "loss" examples (model was genuinely wrong)
- Creates `*_losses_only.jsonl` files with true failures

**Results:**
```
Original dataset: 73 examples (45 ties + 28 losses)
Filtered dataset: 28 examples (0 ties + 28 losses)

MATH: 47 → 12 (removed 35 ties)
CODE: 26 → 16 (removed 10 ties)
```

### 2. Created New Training Script

**File:** `scripts/run_phase1b_losses_only.sh`

**Key changes:**
- Dataset: `*_losses_only.jsonl` (not `*_failures*.jsonl`)
- Output: `checkpoints/phase1b_losses_only`
- Same base: `checkpoints/phase1a_merged`
- Same params: 2 epochs, 5e-6 LR

### 3. How to Run (Vast.ai)

```bash
# 1. Upload filtered data (already created locally)
# Upload: data/phase1/training_from_benchmark/*_losses_only.jsonl

# 2. Upload new training script
# Upload: scripts/run_phase1b_losses_only.sh

# 3. Run training
bash scripts/run_phase1b_losses_only.sh

# 4. Validate results
bash scripts/validate_phase1b1.sh
```

---

## 📊 Expected Results (After Fix)

### With Proper Data (28 true losses only):

**MATH Expectations:**
- Wins: Should improve modestly (12 targeted fixes)
- **Ties: Should STAY ~70%** (no catastrophic forgetting!)
- Losses: Should decrease or stay stable (not explode to 78%)

**CODE Expectations:**
- Wins: Should improve on 16 targeted losses
- Performance: Better without forgetting baseline knowledge

### Success Criteria:

✅ **MATH ties stay high** (60-70%) - proves no forgetting  
✅ **MATH losses don't increase** (<30%) - proves learning worked  
✅ **CODE shows improvement** - proves targeted training effective  

❌ **If ties still collapse** - indicates deeper issue with training approach

---

## 🎓 Lessons Learned

### 1. **Never Train on "Correct" Examples as Failures**

- Tie = Both models correct → Already good behavior
- Training on ties = Teaching model to unlearn correct answers
- **Only train on genuine losses** where model was wrong

### 2. **Validate Training Data Before Use**

Check distribution:
```bash
grep -o '"failure_type": "[^"]*"' data.jsonl | sort | uniq -c
```

If you see lots of "tie" in training data → **STOP and filter!**

### 3. **Monitor for Catastrophic Forgetting**

Red flags:
- Ties dramatically decrease (70% → 18%)
- Losses dramatically increase (24% → 78%)
- Model performs WORSE than baseline

If this happens → Training data or parameters are wrong!

### 4. **Smaller, Cleaner Data > Larger, Noisy Data**

- 28 true losses > 73 mixed (losses + ties)
- Quality matters more than quantity
- Bad data causes more harm than no data

---

## 📋 Action Items

### Immediate (Vast.ai):

- [x] Filter training data (28 true losses only)
- [ ] Upload filtered data to Vast.ai
- [ ] Upload new training script
- [ ] Run `bash scripts/run_phase1b_losses_only.sh`
- [ ] Validate with `bash scripts/validate_phase1b1.sh`
- [ ] Verify ties stay high (~70%) and losses don't explode

### Future Prevention:

- [ ] Update `extract_failures_from_benchmark.py` to exclude ties by default
- [ ] Add validation step in training scripts to check for ties in data
- [ ] Document this lesson in technical_specification.md
- [ ] Add catastrophic forgetting detection to validation scripts

---

## 🔗 Related Files

**Training Data:**
- Original (BAD): `data/phase1/training_from_benchmark/*_failures*.jsonl` (73 examples, 62% ties)
- Filtered (GOOD): `data/phase1/training_from_benchmark/*_losses_only.jsonl` (28 examples, 0% ties)

**Scripts:**
- Filter script: `scripts/filter_true_losses.py`
- New training script: `scripts/run_phase1b_losses_only.sh`
- Validation script: `scripts/validate_phase1b1.sh`

**Checkpoints:**
- Bad training: `checkpoints/phase1b_from_benchmark/` (DELETE THIS)
- Good training (pending): `checkpoints/phase1b_losses_only/`

---

## 💡 Key Insight

> **Training on "ties" is like telling a student: "You got 100% on the test, but you need to change your answers."**
> 
> **Result: Student second-guesses correct knowledge and performs WORSE.**

This is exactly what happened to Phase 1B.1. The fix ensures we only train on genuine mistakes (losses), not correct answers (ties).

---

**Next Update:** After retraining with filtered data, document results here.
