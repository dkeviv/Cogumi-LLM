# Phase 1C Targeted Distillation - Quick Start Guide

## 🎯 Overview

Direct training on curated failure cases with authentic LLM critiques.

**Key Context:**
- **Critique Generation:** GitHub Copilot powered by Claude Sonnet 4.5
- **Critical Finding:** Majority of hard failures (4,942) are due to technical issues (JSON parsing errors, truncation, incomplete responses), NOT task difficulty
- **Implication:** Training will improve model's ability to generate well-formed, complete responses

**Expected Results:**
- Pass rate: 63.34% → 88-92% (+25-29 points)
- Training time: 5-7 hours (with early stopping)
- Total cost: $15-20 (training only)
- Timeline: 1 day

---

## 📋 Prerequisites

### 1. Data Files (✅ Already Complete)
```bash
# Combined dataset ready for training
ls -lh Phase1C_Targeted_Distillation/data/phase1c_combined_direct.jsonl
# Expected: 7,426 total cases (2,484 self-critique + 4,942 hard failures)

# Source files (already combined above)
ls -lh Phase1B_Failure_Analysis/data/phase1c_self_critique_train.jsonl
# 2,484 self-corrected cases with authentic Claude Sonnet 4.5 critiques
# Each example includes: instruction, bad output, corrected output, detailed critique

ls -lh Phase1B_Failure_Analysis/data/phase1c_hard_failures.jsonl
# 4,942 hard failure cases (many from JSON parsing errors, not knowledge gaps)
```

### 2. Base Model
```bash
# Phase 1A output (10GB merged model)
ls -lh Phase1A_Base_Training/models/phase1a_merged_10gb/
```

### 3. Python Dependencies
```bash
pip install transformers datasets peft bitsandbytes accelerate torch
```

---

## 🚀 Quick Start: Direct Training

### **One-Command Execution (Recommended)**

```bash
cd Phase1C_Targeted_Distillation

# Direct training on 7,426 combined examples
python scripts/train_phase1c_combined_smart.py \
  --model_name ../Phase1A_Base_Training/models/phase1a_merged_10gb \
  --dataset data/phase1c_combined_direct.jsonl \
  --output_dir data/checkpoints/phase1c_direct \
  --logging_dir data/logs/phase1c_direct \
  --max_epochs 3 \
  --patience 3 \
  --validation_split 0.05 \
  --eval_steps 500 \
  --save_steps 1000 \
  --learning_rate 3e-6 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --lora_r 64 \
  --lora_alpha 16
```

**Resume Support:**
- Training checkpoints allow resuming if interrupted
- Just add: `--resume_from_checkpoint data/checkpoints/phase1c_direct/checkpoint-XXXX`

---

## 📝 Step-by-Step Direct Training

### **Step 1: Verify Data Files**

```bash
# Check combined dataset
wc -l Phase1C_Targeted_Distillation/data/phase1c_combined_direct.jsonl
# Expected: 7,426 total examples

# Preview format
head -1 Phase1C_Targeted_Distillation/data/phase1c_combined_direct.jsonl | jq .
```

### **Step 2: Direct Training with Early Stopping**

```bash
cd Phase1C_Targeted_Distillation

# Train directly on combined dataset (NO bidirectional pairs)
python scripts/train_phase1c_combined_smart.py \
  --model_name ../Phase1A_Base_Training/models/phase1a_merged_10gb \
  --dataset data/phase1c_combined_direct.jsonl \
  --output_dir data/checkpoints/phase1c_direct \
  --logging_dir data/logs/phase1c_direct \
  --max_epochs 3 \
  --patience 3 \
  --validation_split 0.05 \
  --eval_steps 500 \
  --save_steps 1000 \
  --learning_rate 3e-6 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --lora_r 64 \
  --lora_alpha 16
```

**Features:**
- Direct training on 7,426 examples (NO bidirectional pairs generation)
- Early stopping: patience=3 checkpoints (stops when converged)
- Validation monitoring: 5% split (~371 examples)
- Convergence detection: monitors loss, perplexity, gradient norms
- Automatic best checkpoint restoration
- TensorBoard real-time monitoring

**Expected:**
- Duration: 5-7 hours (stops at convergence, not fixed 8-10 hours)
- Cost: ~$15-20 on H100 (saved time from early stopping)
- Improvement: 63.34% → 88-92% pass rate

**Monitor Training:**
```bash
# Real-time TensorBoard monitoring
tensorboard --logdir data/logs/phase1c_direct --port 6006
# Open: http://localhost:6006
# Watch: eval_loss, perplexity, gradient norms
```

**Resume Training:**
```bash
# Find latest checkpoint
ls -lht data/checkpoints/phase1c_direct/checkpoint-*

# Resume with same command + checkpoint path
python scripts/train_phase1c_combined_smart.py \
  --model_name ../Phase1A_Base_Training/models/phase1a_merged_10gb \
  --dataset data/phase1c_combined_direct.jsonl \
  --output_dir data/checkpoints/phase1c_direct \
  --resume_from_checkpoint data/checkpoints/phase1c_direct/checkpoint-XXXX
```

---

### **Step 3: Merge LoRA and Validate**

```bash
# Merge best checkpoint LoRA adapter to base model
python ../src/phase1_base/merge_lora.py \
  --base ../Phase1A_Base_Training/models/phase1a_merged_10gb \
  --adapter data/checkpoints/phase1c_direct/checkpoint-best \
  --output ../Phase1A_Base_Training/models/phase1c_merged_15gb

# Re-evaluate on test set (verify 88-92% pass rate)
python ../Phase1B_Failure_Analysis/step3_llm_evaluation.py \
  --test_dataset "../Phase1B_Failure_Analysis/data/test_dataset_20k.jsonl" \
  --model_path "../Phase1A_Base_Training/models/phase1c_merged_15gb" \
  --output_path "../Phase1B_Failure_Analysis/data/phase1c_evaluation.jsonl"

# Expected: 88-92% pass rate (up from 63.34%)
```
```

---

## 📊 Cost Breakdown

| Component | Duration | Cost | Details |
|-----------|----------|------|---------|
| **Data Preparation** | Complete | $0 | 7,426 examples ready |
| **Training Only** | 5-7h | $15-20 | H100 with early stopping |
| **Total** | 5-7h | **$15-20** | Direct training, no API costs |

**PIVOTED APPROACH (November 8, 2025):** No Claude/GPT generation, no bidirectional pairs. Train directly on existing data for significant cost savings ($165-170 → $15-20).

---

## 🎯 Success Criteria

- ✅ 7,426 total training examples ready (2,484 self-critique + 4,942 hard failures)
- ✅ Data preparation complete (100% field coverage)
- ⏳ Training converges in 5-7 hours (early stopping triggered)
- ⏳ Pass rate: 88-92% (target from direct training)
- ⏳ No catastrophic forgetting on original tasks
- ⏳ Best checkpoint automatically restored

---

## 🐛 Troubleshooting

### **Data Issues**
```bash
# Verify combined dataset exists
ls -lh Phase1C_Targeted_Distillation/data/phase1c_combined_direct.jsonl
# Expected: 7,426 lines

# Check data format
head -1 Phase1C_Targeted_Distillation/data/phase1c_combined_direct.jsonl | jq .
# Should show: instruction, input (bad), output (corrected), meta (critique)
```

### **Training Issues**
```bash
# CUDA out of memory
--per_device_train_batch_size 2  # Reduce batch size
--gradient_accumulation_steps 16  # Increase to maintain effective batch

# Training not converging
--patience 5  # Increase patience (more checkpoints before stopping)
--eval_steps 250  # Evaluate more frequently

# Model loading errors - verify base model path exists
ls -lh ../Phase1A_Base_Training/models/phase1a_merged_10gb/
```

---

## 📁 Output Files

```
Phase1C_Targeted_Distillation/
├── data/
│   ├── checkpoints/
│   │   └── phase1c_direct/
│   │       ├── checkpoint-500/         # Training checkpoints
│   │       ├── checkpoint-1000/
│   │       ├── checkpoint-1500/
│   │       └── checkpoint-best/        # Best model (early stopping)
│   └── logs/
│       └── phase1c_direct/
│           └── events.out.tfevents.*   # TensorBoard logs

Phase1A_Base_Training/models/
└── phase1c_merged_15gb/                # Final merged model (after Step 3)
```

---

## 🚀 Next Steps After Completion

1. ✅ **Validate results** (88-92% pass rate expected)
2. ➡️ **Phase 2:** Extreme compression (15GB → 520MB)
3. ➡️ **Phase 3:** Domain modifiers (code, reasoning, automation)
4. ➡️ **Phase 4:** Router system (13MB router + 3MB escalation)
5. ➡️ **Phase 5:** Deployment & validation

**Final Target:** 668MB system beating GPT-4 on domain tasks

---

## 💡 Tips

- **Monitor TensorBoard** - Watch convergence in real-time
- **Resume support** - Training checkpoints allow continuing after interruptions
- **Test mode** - Use smaller validation split for quicker iterations
- **Early stopping** - Trust the convergence detection, don't overtrain

---

## 📝 Recent Changes (November 8, 2025)

**STRATEGIC PIVOT:** Removed bidirectional pairs approach
- **Before:** Generate Claude examples → Create bidirectional pairs → 14,662 total examples → $165-170 cost
- **After:** Direct training on existing 7,426 examples → $15-20 cost
- **Savings:** $150-165 saved, 2-4 hour generation step eliminated
- **Rationale:** Existing data quality sufficient, simpler pipeline, cost-effective

**CRITICAL DISCOVERY:** Root Cause of Hard Failures
- **Analysis Method:** GitHub Copilot (Claude Sonnet 4.5) generated authentic critiques for all 2,484 self-corrected cases
- **Key Finding:** Majority of 4,942 hard failures stem from technical issues, NOT task difficulty:
  1. **JSON Parsing Errors** - Responses truncated mid-JSON, causing parse failures
  2. **Response Truncation** - Generation stopped before completion
  3. **Incomplete Responses** - Model failed to finish generating answer
  4. **Technical Issues** - Length limits, generation problems, formatting errors
- **Implication:** Training on these examples will improve:
  - JSON formatting and structure
  - Complete response generation
  - Avoiding premature truncation
  - Handling edge cases in output generation
- **Data Quality:** 100% authentic LLM-generated critiques (0 heuristics), comprehensive failure analysis

