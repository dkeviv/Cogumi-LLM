# Phase 1C/1D Combined Training - Quick Start Guide

## 🎯 Overview

Complete workflow for Phase 1C/1D training with Option A (bidirectional pairs) + smart convergence-based early stopping.

**Expected Results:**
- Pass rate: 63.34% → 88-92% (+25-29 points)
- Training time: 5-7 hours (with early stopping)
- Total cost: $200-250
- Timeline: 1-2 days end-to-end

---

## 📋 Prerequisites

### 1. Data Files (Already Available)
```bash
# Verify files exist
ls -lh "./Phase 1B_2_0/data/Phase 1B_2_0/phase1c_hard_failures.jsonl"
# Expected: 4,942 hard failures

ls -lh "./Phase 1B_2_0/data/data/phase1c/phase1c_self_critique_train.jsonl"
# Expected: 2,389 self-critique examples
```

### 2. Base Model
```bash
# Phase 1A output (15GB full precision)
ls -lh Phase1A_2_0/models/phase1a_merged_10gb/
```

### 3. API Keys
```bash
# Option A: OpenAI GPT-4o-mini (RECOMMENDED: cheaper $25)
export OPENAI_API_KEY="your-key-here"

# Option B: Anthropic Claude Sonnet 4.5 (premium $150)
export ANTHROPIC_API_KEY="your-key-here"
```

### 4. Python Dependencies
```bash
pip install anthropic openai rich transformers datasets peft bitsandbytes
```

---

## 🚀 Quick Start: Automated Workflow

### **One-Command Execution (Easiest)**

```bash
# Using OpenAI GPT-4o-mini (recommended: cheaper)
export OPENAI_API_KEY="your-key"
export API_PROVIDER="openai"
export MODEL="gpt-4o-mini"

./src/phase1c_targeted_distillation/run_phase1c_combined_workflow.sh
```

**The script will:**
1. Estimate costs and ask for confirmation
2. Generate improved examples (2-4 hours, ~$25)
3. Create bidirectional pairs (~5 minutes)
4. Combine datasets (~1 minute)
5. Run smart training with early stopping (5-7 hours, ~$15-20)
6. Save results and training summary

**Resume Support:**
- If interrupted, just re-run the script
- Already-generated examples will be skipped
- Training checkpoints allow resuming

---

## 📝 Step-by-Step Manual Execution

### **Step 1: Generate Improved Examples**

**Cost Estimation First:**
```bash
# Estimate costs before generating
python src/phase1c_targeted_distillation/generate_claude_examples.py \
  --input "./Phase 1B_2_0/data/Phase 1B_2_0/phase1c_hard_failures.jsonl" \
  --output "data/phase1c/improved_examples.jsonl" \
  --api_provider openai \
  --model gpt-4o-mini \
  --estimate_only

# Output shows:
# - Total examples: 4,942
# - Estimated cost: ~$25 (OpenAI) or ~$150 (Claude)
```

**Generate Examples:**
```bash
# Using OpenAI GPT-4o-mini (cheaper)
export OPENAI_API_KEY="your-key"

python src/phase1c_targeted_distillation/generate_claude_examples.py \
  --input "./Phase 1B_2_0/data/Phase 1B_2_0/phase1c_hard_failures.jsonl" \
  --output "data/phase1c/improved_examples.jsonl" \
  --api_provider openai \
  --model gpt-4o-mini \
  --batch_size 10 \
  --delay 0.5

# Progress bar shows real-time generation
# Checkpoints after each example (can interrupt/resume)
# Duration: 2-4 hours
# Cost: ~$25
```

**Alternative: Using Claude Sonnet 4.5 (premium)**
```bash
export ANTHROPIC_API_KEY="your-key"

python src/phase1c_targeted_distillation/generate_claude_examples.py \
  --input "./Phase 1B_2_0/data/Phase 1B_2_0/phase1c_hard_failures.jsonl" \
  --output "data/phase1c/improved_examples_claude.jsonl" \
  --api_provider claude \
  --model claude-sonnet-4-20250514 \
  --batch_size 5 \
  --delay 1.0

# Duration: 3-5 hours (slower API)
# Cost: ~$150
```

---

### **Step 2: Create Bidirectional Pairs (Self-Critique)**

```bash
python src/phase1c_targeted_distillation/create_bidirectional_pairs.py \
  --input "./Phase 1B_2_0/data/data/phase1c/phase1c_self_critique_train.jsonl" \
  --output "data/phase1c/self_critique_bidirectional.jsonl" \
  --source_label "self_critique" \
  --validate

# Input: 2,389 examples
# Output: 4,778 pairs (2,389 forward + 2,389 reverse)
# Duration: ~2 minutes
```

**Preview Mode (Optional):**
```bash
# Preview 3 examples before generating
python src/phase1c_targeted_distillation/create_bidirectional_pairs.py \
  --input "./Phase 1B_2_0/data/data/phase1c/phase1c_self_critique_train.jsonl" \
  --output "data/phase1c/self_critique_bidirectional.jsonl" \
  --source_label "self_critique" \
  --preview 3
```

---

### **Step 3: Create Bidirectional Pairs (Claude/GPT Examples)**

```bash
python src/phase1c_targeted_distillation/create_bidirectional_pairs.py \
  --input "data/phase1c/improved_examples.jsonl" \
  --output "data/phase1c/claude_bidirectional.jsonl" \
  --source_label "claude_generation" \
  --validate

# Input: 4,942 examples
# Output: 9,884 pairs (4,942 forward + 4,942 reverse)
# Duration: ~3 minutes
```

---

### **Step 4: Combine Datasets**

```bash
# Combine into unified training file
cat data/phase1c/self_critique_bidirectional.jsonl \
    data/phase1c/claude_bidirectional.jsonl \
    > data/phase1c/combined_training_bidirectional.jsonl

# Verify
wc -l data/phase1c/*.jsonl
# Expected:
#   4,778 self_critique_bidirectional.jsonl
#   9,884 claude_bidirectional.jsonl
#  14,662 combined_training_bidirectional.jsonl
```

---

### **Step 5: Smart Training with Early Stopping**

```bash
python src/phase1c_targeted_distillation/train_phase1c_combined_smart.py \
  --model_name Phase1A_2_0/models/phase1a_merged_10gb \
  --dataset data/phase1c/combined_training_bidirectional.jsonl \
  --output_dir data/checkpoints/phase1c_combined \
  --logging_dir data/logs/phase1c_combined \
  --max_epochs 3 \
  --patience 3 \
  --validation_split 0.05 \
  --eval_steps 500 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --learning_rate 3e-6 \
  --lora_r 64 \
  --lora_alpha 16

# Features:
# - Early stopping: patience=3 checkpoints
# - Validation monitoring: 5% split (733 examples)
# - Convergence detection: loss, perplexity, gradient norms
# - Automatic best checkpoint restoration
# - TensorBoard real-time monitoring

# Expected:
# - Duration: 5-7 hours (stops at convergence)
# - Cost: ~$15-20 on H100
# - Improvement: 63.34% → 88-92% pass rate
```

**Monitor Training (Real-Time):**
```bash
# Open TensorBoard in another terminal
tensorboard --logdir data/logs/phase1c_combined --port 6006

# View in browser: http://localhost:6006
# Watch: eval_loss, perplexity, gradient norms
```

**Resume Training (if interrupted):**
```bash
# Find latest checkpoint
ls -lht data/checkpoints/phase1c_combined/checkpoint-*

# Resume from checkpoint
python src/phase1c_targeted_distillation/train_phase1c_combined_smart.py \
  --model_name Phase1A_2_0/models/phase1a_merged_10gb \
  --dataset data/phase1c/combined_training_bidirectional.jsonl \
  --output_dir data/checkpoints/phase1c_combined \
  --resume_from_checkpoint data/checkpoints/phase1c_combined/checkpoint-XXXX
```

---

### **Step 6: Merge LoRA and Validate**

```bash
# Merge best checkpoint LoRA adapter to base model
python src/phase1_base/merge_lora.py \
  --base Phase1A_2_0/models/phase1a_merged_10gb \
  --adapter data/checkpoints/phase1c_combined/final \
  --output Phase1A_2_0/models/phase1cd_merged_15gb

# Re-evaluate on test set
python Phase1B_2_0/step3_llm_evaluation.py \
  --test_dataset "./Phase 1B_2_0/data/test_dataset_20k.jsonl" \
  --model_path "Phase1A_2_0/models/phase1cd_merged_15gb" \
  --output_path "Phase 1B_2_0/data/phase1cd_evaluation.jsonl"

# Expected: 88-92% pass rate (up from 63.34%)
```

---

## 📊 Cost Breakdown

| Component | Duration | Cost | Details |
|-----------|----------|------|---------|
| **Claude/GPT Generation** | 2-4h | $25 (GPT) / $150 (Claude) | 4,942 examples |
| **Bidirectional Pairs** | ~5min | $0 | Local processing |
| **Smart Training** | 5-7h | $15-20 | H100 with early stopping |
| **Total** | 8-12h | **$40-45 (GPT)** / **$165-170 (Claude)** | End-to-end |

**Recommendation:** Use OpenAI GPT-4o-mini for $125 savings with minimal quality difference.

---

## 🎯 Success Criteria

- ✅ 14,662 total training examples (4,778 self-critique + 9,884 Claude)
- ✅ Training converges in 5-7 hours (early stopping triggered)
- ✅ Pass rate: 88-92% (target from refined pipeline)
- ✅ No catastrophic forgetting on original tasks
- ✅ Best checkpoint automatically restored

---

## 🐛 Troubleshooting

### **Generation Errors**
```bash
# Rate limit errors
--delay 1.0  # Increase delay between API calls

# Out of memory
--batch_size 5  # Reduce batch size
```

### **Training Issues**
```bash
# CUDA out of memory
--per_device_train_batch_size 2  # Reduce batch size
--gradient_accumulation_steps 4  # Increase to maintain effective batch

# Training not converging
--patience 5  # Increase patience (more checkpoints before stopping)
--eval_steps 250  # Evaluate more frequently
```

### **Resume Generation After Error**
```bash
# Script automatically detects existing examples and skips them
# Just re-run the same command - it will continue from where it stopped
python src/phase1c_targeted_distillation/generate_claude_examples.py [same args]
```

---

## 📁 Output Files

```
data/phase1c/
├── improved_examples.jsonl              # Claude/GPT generated (4,942)
├── self_critique_bidirectional.jsonl    # Self-critique pairs (4,778)
├── claude_bidirectional.jsonl           # Claude/GPT pairs (9,884)
└── combined_training_bidirectional.jsonl # Unified dataset (14,662)

data/checkpoints/phase1c_combined/
├── checkpoint-500/                      # Training checkpoints
├── checkpoint-1000/
├── checkpoint-1500/
├── final/                               # Best model (LoRA adapter)
└── training_summary.json                # Convergence metrics

data/logs/phase1c_combined/
└── events.out.tfevents.*                # TensorBoard logs

Phase1A_2_0/models/
└── phase1cd_merged_15gb/                # Final merged model
```

---

## 🚀 Next Steps After Completion

1. ✅ **Validate results** (88-92% pass rate expected)
2. ➡️ **Phase 1E:** Draft model distillation (140MB, 150 tok/s)
3. ➡️ **Phase 1F:** Speculative decoding (3× speedup)
4. ➡️ **Phase 1G:** Mixture of Depths (2× speedup)
5. ➡️ **Phase 1H:** KV cache INT4 (1.5× speedup)

**Final Target:** 135 tok/s desktop, 300 tok/s mobile mode

---

## 💡 Tips

- **Use GPT-4o-mini first** - Test with 100 examples, then scale up
- **Monitor TensorBoard** - Watch convergence in real-time
- **Resume support** - Don't worry about interruptions
- **Test mode** - Use `--max_examples 100` for quick validation
- **Preview pairs** - Use `--preview 3` before full generation
