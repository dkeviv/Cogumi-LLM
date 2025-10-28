# Phase 1B.2: Real Failures Training Plan

## 🎯 Strategy: Extract ALL Real Failures + Train (No Distillation!)

**Key Insight:** Datasets already have correct answers - no need for GPT-4/5!

---

## 📋 Step-by-Step Execution

### **Step 1: Extract ALL Math Failures** (Est: 2-4 hours, ~$4-8 GPU)

```bash
# On Vast.ai
cd /workspace/data/Cogumi-LLM

python scripts/extract_all_real_failures.py \
  --model_path checkpoints/phase1a_merged \
  --output_dir data/phase1b2 \
  --categories math
```

**What it does:**
- Tests Phase 1A on ALL 7,473 GSM8K train problems
- Identifies failures (expect ~2,000-3,000 failures, ~30-40% failure rate)
- Saves training-ready format: `{"instruction": prompt, "output": solution}`
- Output: `data/phase1b2/math_training_ready.jsonl`

**Why ALL failures, not just 2,000:**
- Natural distribution of difficulty
- No cherry-picking bias
- Represents true model weaknesses

---

### **Step 2: Train Phase 1B.2 on Real Failures** (Est: 1-2 hours, ~$2-4)

```bash
python train_phase1b_benchmark.py \
  --model_name checkpoints/phase1a_merged \
  --dataset_path "data/phase1b2/math_training_ready.jsonl" \
  --output_dir checkpoints/phase1b2_math \
  --num_train_epochs 2 \
  --learning_rate 3e-6
```

**Key Parameters:**
- Learning rate: **3e-6** (lower than 5e-6 to prevent forgetting)
- Epochs: **2** (enough for 2K-3K examples)
- Base: Phase 1A merged (600K training preserved)
- Output: LoRA adapter trained on real failures

**Expected Dataset Size:**
- ~2,000-3,000 math failures
- Much larger than 28 examples (prevents overfitting)
- Still targeted (not full 600K retraining)

---

### **Step 3: Validate Results** (Est: 15-20 min, ~$0.75)

```bash
bash scripts/validate_phase1b1.sh
```

**Success Criteria:**
- MATH ties should STAY ~60-70% (no catastrophic forgetting)
- MATH wins should IMPROVE from 6% → 15-25%
- MATH losses should DECREASE from 24% → 10-20%

**Why This Should Work:**
- 2-3K examples >> 28 examples (prevents overfitting)
- Real problems (not synthetic)
- Lower learning rate (3e-6 vs 5e-6)
- Natural difficulty distribution

---

## 📊 Expected Results

### **Phase 1A Baseline:**
- MATH: 6% wins, 70% ties, 24% losses
- GSM8K failure rate: ~30-40%

### **Phase 1B.2 (After Training):**
- MATH: 15-25% wins, 60-70% ties, 10-20% losses  
- GSM8K failure rate: ~15-25%

**Key Difference from Phase 1B.1:**
- Phase 1B.1: 28 examples + ties → Catastrophic forgetting
- Phase 1B.2: 2-3K pure failures → Targeted improvement without forgetting

---

## 💰 Cost Breakdown

| Step | Time | GPU Cost | Total |
|------|------|----------|-------|
| Extract math failures | 2-4 hrs | $4-8 | $4-8 |
| Train Phase 1B.2 | 1-2 hrs | $2-4 | $2-4 |
| Validation | 15-20 min | $0.75 | $0.75 |
| **TOTAL** | **3-7 hrs** | **$6.75-12.75** | **$6.75-12.75** |

**Compare to GPT-5 Distillation:**
- GPT-5 for 2K examples: ~$200
- Our approach: $6.75-12.75
- **Savings: $187-193 (94-96% cheaper!)**

---

## 🚀 Quick Start Commands

```bash
# 1. Upload script to Vast.ai
# Upload: scripts/extract_all_real_failures.py

# 2. Extract failures
python scripts/extract_all_real_failures.py \
  --model_path checkpoints/phase1a_merged \
  --output_dir data/phase1b2 \
  --categories math

# 3. Check extracted data
wc -l data/phase1b2/math_training_ready.jsonl
head -2 data/phase1b2/math_training_ready.jsonl

# 4. Train
python train_phase1b_benchmark.py \
  --model_name checkpoints/phase1a_merged \
  --dataset_path "data/phase1b2/math_training_ready.jsonl" \
  --output_dir checkpoints/phase1b2_math \
  --num_train_epochs 2 \
  --learning_rate 3e-6

# 5. Validate
bash scripts/validate_phase1b1.sh
```

---

## ✅ Advantages Over Previous Approaches

| Approach | Data Size | Cost | Forgetting Risk | Quality |
|----------|-----------|------|-----------------|---------|
| Phase 1B.1 (ties+losses) | 73 | $0.50 | ❌ HIGH (70%→24% ties) | ❌ Catastrophic |
| Phase 1B.1 (losses only) | 28 | $0.30 | ❌ HIGH (70%→24% ties) | ❌ Still forgets |
| GPT-5 Distillation | 2K | $200 | ⚠️ MEDIUM | ✅ High quality |
| **Phase 1B.2 (Real Failures)** | **2-3K** | **$7-13** | **✅ LOW** | **✅ Real data** |

---

## 🔬 Why This Should Work

1. **Large enough dataset:** 2-3K examples prevents overfitting
2. **Real failures:** Natural difficulty distribution
3. **Ground truth answers:** No synthetic data artifacts
4. **Lower LR:** 3e-6 preserves Phase 1A knowledge
5. **Targeted:** Focus on actual weaknesses, not random examples

---

## 📈 If Successful, Next Steps

### **Expand to Other Categories:**

```bash
# Extract code failures (HumanEval + MBPP)
python scripts/extract_all_real_failures.py --categories code

# Extract creativity failures (MMLU)
python scripts/extract_all_real_failures.py --categories creativity

# Train combined Phase 1B.2
python train_phase1b_benchmark.py \
  --dataset_path "data/phase1b2/*_training_ready.jsonl" \
  --output_dir checkpoints/phase1b2_full
```

**Expected Total:**
- Math: ~2-3K failures
- Code: ~500-800 failures  
- Creativity: ~5-7K failures
- **Total: ~8-11K training examples**

This becomes our **Phase 1B.2 complete** - major improvement without catastrophic forgetting! 🚀

---

## 🎯 Success Metrics

**Must achieve:**
- ✅ MATH ties stay >60% (no catastrophic forgetting)
- ✅ MATH wins improve >10% → 16%+
- ✅ MATH losses decrease <20%

**Stretch goals:**
- ✅ MATH wins reach 20-25%
- ✅ Overall improvement without affecting other benchmarks
- ✅ Foundation for Phase 1B.3 (expand to more categories)
