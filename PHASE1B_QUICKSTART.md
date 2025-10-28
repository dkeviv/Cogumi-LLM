# Phase 1B Quick Start - Self-Consistency Training

## 🎯 Goal
Fix the **10% consistency problem** that causes 70% ties in MATH, 28% in CODE.

## 📋 What We Know
**Current State (Phase 1A):**
- MATH: 41% correct, **70% ties** (consistency: 10%)
- CODE: 58% correct, **28% ties** (consistency: 10%)
- Model generates 10 completely different responses for same prompt

**Root Cause:** Model is too random/non-deterministic even with same input

**Solution:** Train on self-consistent examples to "bake in" determinism

## 🚀 Quick Execution (Vast.ai H100)

### Phase 1B.1: Minimal Proof of Concept (FAST TEST) ⚡

**Purpose:** Validate that self-consistency training works before investing in full dataset

**Data:**
- MATH: 100 examples (was 500)
- CODE: 50 examples (was 164)
- **Total: 150 examples** (was 664)

**Why minimal?**
- ✅ Fast validation: 30-45 min training vs 3-4 hours
- ✅ Low cost: $1-2 vs $6-8
- ✅ Proves concept: If this works, expand to 2,500+ examples
- ✅ Quick iteration: Can adjust parameters if needed

### Option 1: One-Command Execution (RECOMMENDED)
```bash
cd /workspace/data/Cogumi-LLM
bash scripts/run_phase1b_self_consistency.sh
```
**Time:** 1-2 hours total (30-45 min training)
**Cost:** $2-4 (was $12-17)

### Option 2: Step-by-Step Execution

**Step 1: Generate Self-Consistent Data (30-45 min, $0.50-1)**
```bash
cd /workspace/data/Cogumi-LLM
python scripts/self_consistency_distillation.py
```
Output: `self_distillation/math_distilled.jsonl` (100), `code_distilled.jsonl` (50)

**Step 2: Train on Consistent Data (30-45 min, $1-2)**
```bash
python train_qlora_optimized.py \
  --model_name unsloth/meta-llama-3.1-8b-instruct-bnb-4bit \
  --dataset_path "self_distillation/*.jsonl" \
  --output_dir checkpoints/self_consistent_test \
  --num_train_epochs 2 \
  --learning_rate 5e-6 \
  --per_device_train_batch_size 4
```

**Step 3: Quick Validation (15-20 min)**
- Run consistency test on 10-20 problems
- Check if consistency improves from 10% → 40-50%
- If YES → Proceed to Phase 1B.2 (expand to 2,500+ examples)
- If NO → Iterate on parameters

## 📊 Expected Results

### Phase 1B.1: Minimal Test (150 examples)

| Metric | Before (1A) | After (1B.1 Target) | Notes |
|--------|-------------|---------------------|-------|
| Training time | - | 30-45 min | Fast validation |
| Training cost | - | $1-2 | Low risk investment |
| Consistency | 10% | 40-50% | Proof of concept |
| MATH Score | 41% | 45-50% | Modest improvement expected |

**Decision Point:**
- ✅ If consistency improves (10% → 40-50%): **Proceed to Phase 1B.2**
- ❌ If no improvement: Iterate on parameters (temp, epochs, lr)

### Phase 1B.2: Full Expansion (if 1B.1 succeeds)

After validating the approach works:
1. Run `scripts/identify_failures.py` to find ~2,500 model failures
2. Train on failures for 6-8 hours ($12-15)
3. Expected final results:

| Metric | Phase 1A | Phase 1B.1 | Phase 1B.2 Target |
|--------|----------|------------|-------------------|
| Consistency | 10% | 40-50% | 60-80% |
| MATH Score | 41% | 45-50% | 65-75% |
| MATH Ties | 70% | 60-65% | <30% |
| CODE Score | 58% | 60-62% | 70-80% |
| CODE Ties | 28% | 25% | <20% |

## ✅ Success Criteria (ALL must pass)

### Phase 1B.1 (Minimal Test) - Validation Criteria

- [ ] **Consistency ≥40%** (up from 10% - proves approach works)
- [ ] **MATH score ≥45%** (any improvement validates concept)
- [ ] **Training completes in <1 hour** (fast iteration)
- [ ] **No catastrophic forgetting** (reasoning score maintains ≥85%)

**If ALL pass:** ✅ Proceed to Phase 1B.2 (expand to 2,500+ examples)  
**If ANY fail:** 🔄 Iterate on parameters or abandon approach

### Phase 1B.2 (Full Expansion) - Final Success Criteria

Only proceed here if Phase 1B.1 validates the approach:

- [ ] **Consistency ≥60%** (major improvement)
- [ ] **MATH score ≥65%** (significant gain)
- [ ] **CODE score ≥70%** (meets target)
- [ ] **MATH ties <30%** (down from 70%)
- [ ] **CODE ties <20%** (down from 28%)

**If ALL pass:** ✅ Proceed to Phase 1C (GPT-5 Targeted Distillation)  
**If ANY fail:** 🔄 Iterate with more data or different approach

## 🔧 Temperature Strategy (Key Innovation)

| Category | Generate Temp | Train Temp | Why |
|----------|--------------|------------|-----|
| MATH | 0.0 | 0.0 | Math needs exact answers |
| CODE | 0.0 | 0.0 | Code must be correct |
| CREATIVITY | 0.7 | 0.3 | Generate creative → train consistently |

**The trick:** Generate at optimal temp, then train model to reproduce those outputs deterministically.

## 📁 Key Files

**Execution:**
- `scripts/run_phase1b_self_consistency.sh` - One-command execution
- `scripts/self_consistency_distillation.py` - Data generation (416 lines)
- `train_qlora_optimized.py` - Training script

**Documentation:**
- `PHASE1B_SELF_CONSISTENCY_PLAN.md` - Full execution plan
- `docs/IMPLEMENTATION_CHECKLIST.md` - Task tracking

**Analysis:**
- `notebooks/Benchmark_Diagnostic_v2.ipynb` - Consistency tests

## 🐛 Troubleshooting

**If data generation fails:**
- Check model checkpoint exists: `checkpoints/final`
- Verify datasets installed: `gsm8k`, `openai_humaneval`

**If consistency doesn't improve:**
- Increase epochs (2 → 3-4)
- Lower training temperature (0.0 → more deterministic)
- Add more categories (reasoning, knowledge)

**If scores drop:**
- Check for catastrophic forgetting
- Lower learning rate (5e-6 → 3e-6)
- Mix in more original training data

## 💰 Budget Tracking

**Phase 1 Total:** $505  
**Phase 1A (Complete):** $220  

**Phase 1B - 2-Stage Approach:**
- **Phase 1B.1 (Minimal Test):** $2-4 (150 examples, 30-45 min) ⚡
- **Phase 1B.2 (Full Expansion):** $12-15 (2,500 examples, 6-8 hrs) - *IF 1B.1 succeeds*

**Total Phase 1B:** $14-19 (was $12-17)  
**Remaining for Phase 1C:** $266-269

---

**Created:** October 27, 2025  
**Status:** Ready to execute  
**Next:** Run `bash scripts/run_phase1b_self_consistency.sh` on Vast.ai
