# Phase 1B: Self-Consistency Training Plan

## 🔍 Problem Identified

**From diagnostic results:**
- **10% consistency** across MATH, CODE, CREATIVITY
- Model generates 10 unique responses for 10 runs (completely different every time)
- This causes **70% tie rate in MATH, 28% in CODE**
- Root cause: Model is too random/non-deterministic at temperature 0.7

## 📋 Solution: Category-Specific Self-Consistency Distillation

### Strategy Overview

Different categories need different approaches:

| Category | Generate Temp | Train Temp | Rationale |
|----------|--------------|------------|-----------|
| **MATH** | 0.0 (greedy) | 0.0 | Math needs exact answers - maximize determinism |
| **CODE** | 0.0 (greedy) | 0.0 | Code must be correct - no room for randomness |
| **REASONING** | 0.1 | 0.0 | Slight diversity, train deterministically |
| **CREATIVITY** | 0.7 | 0.3 | Generate creative, train to reproduce consistently |
| **KNOWLEDGE** | 0.0 | 0.0 | Factual knowledge needs precision |
| **INSTRUCTION** | 0.2 | 0.0 | Follow instructions precisely |

## 🎯 Execution Steps

### Phase 1B.1: Minimal Proof of Concept (FAST VALIDATION) ⚡

**Purpose:** Quick test to validate self-consistency training works before investing in full dataset

**Location:** Vast.ai H100

**Data:**
- MATH: 100 examples (representative sample from GSM8K)
- CODE: 50 examples (representative sample from HumanEval)
- **Total: 150 examples** (was 664)

**Command:**
```bash
cd /workspace/data/Cogumi-LLM
python scripts/self_consistency_distillation.py
```

**What it does:**
1. Loads trained model from `/workspace/data/Cogumi-LLM/checkpoints/final`
2. Generates 100 MATH solutions at temp=0.0 (deterministic)
3. Generates 50 CODE solutions at temp=0.0
4. Saves examples to train model to be deterministic

**Output:** 
- `/workspace/data/Cogumi-LLM/self_distillation/math_distilled.jsonl` (100 examples)
- `/workspace/data/Cogumi-LLM/self_distillation/code_distilled.jsonl` (50 examples)

**Expected time:** 30-45 minutes (was 2-3 hours)

**Cost:** ~$0.50-1 (was $2-3)

### Phase 1B.1: Train Model on Test Data

**Command:**
```bash
cd /workspace/data/Cogumi-LLM
python train_qlora_optimized.py \
  --model_name unsloth/meta-llama-3.1-8b-instruct-bnb-4bit \
  --dataset_path self_distillation/*.jsonl \
  --output_dir checkpoints/self_consistent_test \
  --num_train_epochs 2 \
  --learning_rate 5e-6 \
  --per_device_train_batch_size 4
```

**What it does:**
1. Loads the Phase 1A model
2. Fine-tunes on 150 self-consistent examples
3. Tests if model learns to be more deterministic

**Training config:**
- **Epochs:** 2 
- **Learning rate:** 5e-6
- **Batch size:** 4
- **Total examples:** 150
- **Total steps:** ~75 steps (150 examples × 2 epochs ÷ 4 batch size)

**Expected time:** 30-45 minutes (was 3-4 hours)

**Cost:** ~$1-2 (was $6-8)

### Phase 1B.1: Quick Validation

**Test consistency on 10-20 problems:**
```python
# In notebooks/Benchmark_Diagnostic_v2.ipynb
# Load checkpoints/self_consistent_test
# Run consistency test (Cell 15)
# Check: Did consistency improve 10% → 40-50%?
```

**Decision Point:**
- ✅ **If consistency ≥40%:** Approach validated → Proceed to Phase 1B.2
- ❌ **If consistency <40%:** Iterate on parameters or try different approach

**Expected time:** 15-20 minutes

**Cost:** ~$0.50

---

### Phase 1B.2: Full Expansion (ONLY IF 1B.1 SUCCEEDS) ⭐

**Purpose:** After validating approach, train on model's actual failures for maximum impact

**Step 1: Identify Model Failures (2-3 hrs, $4-6)**
```bash
cd /workspace/data/Cogumi-LLM
python scripts/identify_failures.py
```

**What it does:**
1. Tests model on GSM8K train (2,000 sampled problems)
2. Tests model on MBPP train (374 problems)
3. Identifies problems where:
   - Model gets wrong answer
   - Consistency <30%
4. Outputs ~2,500 failure examples

**Output:**
- `/workspace/data/Cogumi-LLM/failures/math_failures.jsonl` (~2,000)
- `/workspace/data/Cogumi-LLM/failures/code_failures.jsonl` (~500)

**Step 2: Generate Training Data from Failures (1-2 hrs, $2-3)**
```bash
python scripts/self_consistency_distillation.py --input failures/*.jsonl
```

**Step 3: Train on Failures (6-8 hrs, $12-15)**
```bash
python train_qlora_optimized.py \
  --dataset_path self_distillation/failures_*.jsonl \
  --output_dir checkpoints/self_consistent_final \
  --num_train_epochs 2 \
  --learning_rate 5e-6
```

**Step 4: Full Re-Benchmark (2-3 hrs, $4-6)**
```bash
python scripts/run_benchmarks.py \
  --model_path checkpoints/self_consistent_final \
  --output_dir benchmark_results_phase1b
```

**Total Phase 1B.2:**
- **Time:** 11-16 hours
- **Cost:** $22-30
- **Expected:** MATH 45-50% → 65-75%, CODE 60-62% → 70-80%

### Step 4: Analysis & Next Steps (Local Mac)

**Run locally:**
```bash
cd /Users/vivekdurairaj/Projects/Cogumi-LLM/notebooks
# Open Benchmark_Diagnostic_v2.ipynb
# Update paths to benchmark_results_self_consistent
# Run consistency tests on new model
```

**Compare:**
- Before: 10% consistency, 70% MATH ties
- After: Target 70% consistency, <30% MATH ties

## 📊 Success Criteria

### Phase 1B.1 (Minimal Test) - Validation Criteria ⚡
- [ ] **Consistency:** ≥40% (up from 10% - proves approach works)
- [ ] **MATH score:** ≥45% (any improvement validates concept)
- [ ] **Training time:** <1 hour (fast iteration)
- [ ] **No catastrophic forgetting:** Reasoning ≥85%

**If ALL pass:** ✅ Approach validated → Proceed to Phase 1B.2  
**If ANY fail:** 🔄 Iterate on parameters or abandon approach

### Phase 1B.2 (Full Expansion) - Final Success Criteria
- [ ] **Consistency:** ≥60% (up from 10%)
- [ ] **MATH score:** ≥65% (up from 41%)
- [ ] **CODE score:** ≥70% (up from 58%)
- [ ] **MATH ties:** <30% (down from 70%)
- [ ] **CODE ties:** <20% (down from 28%)

### If successful, proceed to Phase 1C (GPT-5 Targeted Distillation)
### If not, iterate with:
- More aggressive temperature scheduling
- Longer training (3-4 epochs)
- Additional categories (reasoning, knowledge)

## 💰 Budget Summary

### Phase 1B.1: Minimal Test (Proof of Concept) ⚡

| Step | Time | Cost |
|------|------|------|
| Data generation (150 examples) | 30-45 min | $0.50-1 |
| Self-consistency training | 30-45 min | $1-2 |
| Quick validation | 15-20 min | $0.50 |
| **Phase 1B.1 Total** | **1.5-2 hrs** | **$2-4** |

**Decision Point:** If validation passes → Proceed to Phase 1B.2

### Phase 1B.2: Full Expansion (Only if 1B.1 succeeds)

| Step | Time | Cost |
|------|------|------|
| Identify failures (2K problems) | 2-3 hrs | $4-6 |
| Generate training data | 1-2 hrs | $2-3 |
| Train on failures (2,500 examples) | 6-8 hrs | $12-15 |
| Full re-benchmark | 2-3 hrs | $4-6 |
| **Phase 1B.2 Total** | **11-16 hrs** | **$22-30** |

### Total Phase 1B Budget

| Scenario | Time | Cost | Expected Result |
|----------|------|------|-----------------|
| **1B.1 only (validation fails)** | 1.5-2 hrs | $2-4 | Stop and iterate |
| **1B.1 + 1B.2 (validation passes)** | 12.5-18 hrs | **$24-34** | MATH 65-75% ✅ |

**Remaining Phase 1 budget:** $505 - $220 (1A) - $34 (1B max) = **$251 for Phase 1C**

## 🚀 Ready to Execute

All scripts are ready:
✅ `scripts/self_consistency_distillation.py` (data generation)
✅ `train_qlora_optimized.py` (training script)
✅ `scripts/run_benchmarks.py` (benchmarking)
✅ `notebooks/Benchmark_Diagnostic_v2.ipynb` (analysis)

**Next command to run on Vast.ai:**
```bash
cd /workspace/data/Cogumi-LLM && python scripts/self_consistency_distillation.py
```

---

**Created:** October 27, 2025
**Phase:** 1B - Self-Consistency Training
**Goal:** Fix 10% consistency → 70%+ consistency, reduce tie rates
