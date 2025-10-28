# Phase 1B Execution - Ready to Run

## 📊 Summary of Findings

After completing Phase 1A training and running comprehensive diagnostics, we identified a **critical consistency problem**:

### The Problem
- **Consistency:** Only 10% (model generates 10 unique responses for same prompt)
- **Impact:** 
  - MATH: 70% ties (unable to determine correct answer)
  - CODE: 28% ties
  - Makes accurate benchmarking impossible

### Root Cause
Model is **too random/non-deterministic** even with identical inputs. This is why:
- MATH score stuck at 41% (should be 65-75%)
- CODE score at 58% (should be 70-80%)
- High variance in outputs prevents confident predictions

### The Solution
**Self-Consistency Training** - Train model on examples generated at optimal temperatures:
- **MATH:** Generate at temp=0.0 (greedy/deterministic) → Train to be deterministic
- **CODE:** Generate at temp=0.0 → Train to be deterministic
- **CREATIVITY:** Generate at temp=0.7 (diverse) → Train at temp=0.3 (learn patterns)

This "bakes in" consistency so model produces deterministic outputs even at inference.

## 🎯 What's Ready to Execute

### All Scripts Created ✅
1. **Data Generation:** `scripts/self_consistency_distillation.py` (416 lines)
   - Category-specific temperature strategies
   - Self-consistency filtering with majority voting
   - Outputs to `self_distillation/*.jsonl`

2. **Execution Script:** `scripts/run_phase1b_self_consistency.sh`
   - One-command execution for entire Phase 1B
   - Includes verification and validation steps
   - Auto-runs consistency tests

3. **Training Script:** `train_qlora_optimized.py` (already exists)
   - QLoRA fine-tuning on self-consistent data
   - 2 epochs, lr=5e-6, batch size 4

4. **Benchmark Script:** `scripts/run_benchmarks.py` (already exists)
   - Re-run benchmarks on improved model
   - Measure consistency improvement

### All Documentation Created ✅
1. **Full Plan:** `PHASE1B_SELF_CONSISTENCY_PLAN.md`
   - Problem analysis
   - Strategy overview with temperature table
   - Step-by-step execution
   - Budget and timeline
   - Success criteria

2. **Quick Start:** `PHASE1B_QUICKSTART.md`
   - One-page reference
   - Quick execution commands
   - Expected results table
   - Troubleshooting guide

3. **Task Tracking:** `docs/IMPLEMENTATION_CHECKLIST.md` (updated)
   - Phase 1B tasks with success criteria
   - Validation requirements
   - Clear pass/fail thresholds

4. **Status Update:** `docs/CURRENT_STATUS.md` (updated)
   - Current phase: 1B (ready to execute)
   - Phase 1A results documented
   - Diagnostic findings recorded

### Diagnostic Tools Ready ✅
1. **Consistency Test:** `notebooks/Benchmark_Diagnostic_v2.ipynb` Cell 15
   - Runs 10 iterations per problem
   - Measures unique responses
   - Calculates consistency percentage
   - Identifies root cause (10% = very low)

2. **Benchmark Analysis:** Full LLM AutoEval integration
   - MATH (GSM8K), CODE (HumanEval), REASONING (MMLU)
   - Tie detection and reporting
   - Score calculation with error handling

## 🚀 Execution Flow (Vast.ai H100)

### Phase 1B.1: Minimal Proof of Concept (FAST TEST) ⚡

**Purpose:** Quick validation with minimal examples before full investment

```bash
cd /workspace/data/Cogumi-LLM
bash scripts/run_phase1b_self_consistency.sh
```

**What happens:**
1. Generate 150 self-consistent examples (100 MATH + 50 CODE)
2. Train 2 epochs (~75 steps)
3. Quick validation test

**Time:** 1.5-2 hours total  
**Cost:** $2-4  
**Output:** Proof that self-consistency training works

**Success criteria:**
- Consistency improves: 10% → 40-50%
- MATH score improves: 41% → 45-50%
- No catastrophic forgetting

**Decision:**
- ✅ If successful → Proceed to Phase 1B.2 (expand to 2,500+ examples)
- ❌ If not successful → Iterate parameters or abandon approach

---

### Phase 1B.2: Full Expansion (ONLY IF 1B.1 SUCCEEDS)

**Purpose:** Train on model's actual failures for maximum improvement

**Step 1: Identify failures**
```bash
python scripts/identify_failures.py  # 2-3 hours, $4-6
```
Finds ~2,500 problems where model fails or has low consistency

**Step 2: Generate + train on failures**
```bash
python scripts/self_consistency_distillation.py --input failures/*.jsonl
python train_qlora_optimized.py --data self_distillation/failures_*.jsonl  # 6-8 hours, $12-15
```

**Step 3: Full benchmark**
```bash
python scripts/run_benchmarks.py  # 2-3 hours, $4-6
```

**Total Phase 1B.2:** 11-16 hours, $22-30  
**Expected:** MATH 45-50% → 65-75%, CODE 60-62% → 70-80%

## ✅ Success Criteria (ALL Must Pass)

### Phase 1B.1 (Minimal Test) - Fast Validation ⚡

Quick test to prove concept works:
- [ ] **Consistency ≥40%** (up from 10%)
- [ ] **MATH score ≥45%** (up from 41%)
- [ ] **Training completes <1 hour**
- [ ] **No catastrophic forgetting** (reasoning ≥85%)

**Time:** 1.5-2 hours  
**Cost:** $2-4  
**Next:** If pass → Phase 1B.2, If fail → iterate/abandon

### Phase 1B.2 (Full Expansion) - Final Goals

Only attempt after Phase 1B.1 validates approach:
- [ ] **Consistency ≥60%** (major improvement)
- [ ] **MATH score ≥65%** (significant gain)
- [ ] **CODE score ≥70%** (meets target)
- [ ] **MATH ties <30%** (down from 70%)
- [ ] **CODE ties <20%** (down from 28%)

**Time:** 11-16 hours  
**Cost:** $22-30  
**Next:** If pass → Phase 1C, If fail → iterate with more data

## 📁 File Locations

### Execution Scripts
- `/Users/vivekdurairaj/Projects/Cogumi-LLM/scripts/run_phase1b_self_consistency.sh`
- `/Users/vivekdurairaj/Projects/Cogumi-LLM/scripts/self_consistency_distillation.py`
- `/Users/vivekdurairaj/Projects/Cogumi-LLM/train_qlora_optimized.py`

### Documentation
- `/Users/vivekdurairaj/Projects/Cogumi-LLM/PHASE1B_SELF_CONSISTENCY_PLAN.md`
- `/Users/vivekdurairaj/Projects/Cogumi-LLM/PHASE1B_QUICKSTART.md`
- `/Users/vivekdurairaj/Projects/Cogumi-LLM/docs/IMPLEMENTATION_CHECKLIST.md`
- `/Users/vivekdurairaj/Projects/Cogumi-LLM/docs/CURRENT_STATUS.md`

### Diagnostic Tools
- `/Users/vivekdurairaj/Projects/Cogumi-LLM/notebooks/Benchmark_Diagnostic_v2.ipynb`

### Model Checkpoints (Vast.ai)
- Input: `/workspace/data/Cogumi-LLM/checkpoints/final` (Phase 1A)
- Output: `/workspace/data/Cogumi-LLM/checkpoints/self_consistent` (Phase 1B)

## 🔧 Key Innovation: Category-Specific Temperatures

| Category | Generate | Train | Rationale |
|----------|----------|-------|-----------|
| **MATH** | temp=0.0 | temp=0.0 | Exact answers required |
| **CODE** | temp=0.0 | temp=0.0 | Code must be correct |
| **REASONING** | temp=0.1 | temp=0.0 | Slight diversity → deterministic training |
| **CREATIVITY** | temp=0.7 | temp=0.3 | Generate diverse → learn consistent patterns |
| **KNOWLEDGE** | temp=0.0 | temp=0.0 | Factual precision |
| **INSTRUCTION** | temp=0.2 | temp=0.0 | Follow instructions precisely |

**The Insight:** Different tasks need different generation strategies, but ALL need consistent reproduction at inference time.

## 💰 Budget Impact

**Phase 1 Original Budget:** $505  
**Phase 1A Spent:** $220  

**Phase 1B - 2-Stage Approach:**
- **Phase 1B.1 (Minimal Test):** $2-4 (150 examples, <2 hrs) ⚡
- **Phase 1B.2 (Full Expansion):** $22-30 (2,500 examples, 11-16 hrs) - *IF 1B.1 succeeds*

**Total Phase 1B Range:**
- **Best case (1B.1 fails, stop early):** $2-4
- **Full execution (1B.1 + 1B.2):** $24-34

**Phase 1C Remaining:** $251-279 (sufficient for GPT-5 distillation)

## 📈 Expected Improvements

### Phase 1B.1: Minimal Test (150 examples)

| Metric | Phase 1A | Phase 1B.1 Target | Improvement |
|--------|----------|-------------------|-------------|
| Training time | - | 30-45 min | Fast validation |
| Training cost | - | $1-2 | Low risk |
| Consistency | 10% | 40-50% | **+30-40%** |
| MATH Score | 41% | 45-50% | **+4-9%** |

**Purpose:** Prove self-consistency training works before full investment

### Phase 1B.2: Full Expansion (2,500 examples)

Only executed if Phase 1B.1 validates approach:

| Metric | Phase 1A | Phase 1B.1 | Phase 1B.2 Target | Total Gain |
|--------|----------|------------|-------------------|------------|
| Consistency | 10% | 40-50% | 60-80% | **+50-70%** |
| MATH Score | 41% | 45-50% | 65-75% | **+24-34%** |
| MATH Ties | 70% | 60-65% | <30% | **-40%+** |
| CODE Score | 58% | 60-62% | 70-80% | **+12-22%** |
| CODE Ties | 28% | 25% | <20% | **-8%+** |

**Key insight:** Consistency improvement from 10% → 70% unlocks true model performance and enables accurate benchmarking.

## 🎯 Next Steps After Phase 1B

**If successful:**
1. **Phase 1C:** GPT-5 Targeted Distillation
   - Test self-consistent model on 50K examples
   - Identify remaining failure patterns
   - Generate 40K GPT-5 examples
   - Final target: 88-100% GPT-4 baseline

**If needs iteration:**
1. Increase training epochs (2 → 3-4)
2. Add more categories (reasoning, knowledge, instruction)
3. Adjust temperature strategies based on results
4. Re-run with modified parameters

## 🔍 Validation Process

After Phase 1B completes:

1. **Run consistency diagnostic** (notebooks/Benchmark_Diagnostic_v2.ipynb Cell 15)
   - Must show ≥60% consistency
   
2. **Run full benchmarks** (scripts/run_benchmarks.py)
   - MATH ≥65%, CODE ≥70%
   
3. **Check tie rates** (benchmark output)
   - MATH <30%, CODE <20%
   
4. **Compare outputs** (manual spot check)
   - Verify determinism at temp=0.7
   - Check answer quality maintained

5. **Document results** (docs/CURRENT_STATUS.md)
   - Update with actual results
   - Note any deviations from expected
   - Record lessons learned

## ✨ Ready to Execute

Everything is prepared and ready:
✅ All scripts written and tested (syntax-wise)
✅ All documentation complete
✅ Success criteria clearly defined
✅ Budget approved ($12-17)
✅ Timeline estimated (5-7 hours)
✅ Validation process documented

**Next action:** Run on Vast.ai H100:
```bash
cd /workspace/data/Cogumi-LLM
bash scripts/run_phase1b_self_consistency.sh
```

---

**Prepared:** October 27, 2025  
**Phase:** 1B - Self-Consistency Training  
**Status:** 🟢 READY TO EXECUTE  
**Blocker:** None - all dependencies met
