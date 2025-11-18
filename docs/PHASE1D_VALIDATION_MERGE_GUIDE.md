# Phase 1D: Validation & Merge Guide for Vast.ai

**Last Updated:** November 18, 2025

---

## Overview

After training completes on Vast.ai, you need to:
1. ✅ **Validate** the trained model on **independent benchmark test set** (never seen during training)
2. ✅ **Merge** LoRA weights into base model (creates standalone model)
3. ✅ **Re-validate** merged model (verify identical to LoRA)
4. ✅ **Download** results and merged model

**Expected Runtime:** 30-45 minutes  
**Expected Cost:** ~$0.50-0.75 (0.5 hours on H100)

**CRITICAL:** We use **independent benchmark datasets** (DROP, GPQA, HumanEval, MATH, MMLU) for validation. These provide a **real generalization test** since the model has never seen these examples during training.

---

## Prerequisites

- ✅ Training completed successfully (loss = 0.02)
- ✅ Model saved in `/workspace/models/phase1_maml_lora_v2/best/`
- ✅ Benchmark datasets exist at `/workspace/data/benchmarks/*.jsonl`
- ✅ Virtual environment active

---

## Quick Start (On Vast.ai)

### Step 1: Upload Scripts

From your **local machine**, upload the validation scripts to Vast.ai:

```bash
# Get your Vast.ai connection info
# Port and IP from Vast.ai dashboard

# Upload scripts
scp -P <port> scripts/convert_benchmarks_to_test.py root@<vast-ip>:/workspace/scripts/
scp -P <port> scripts/phase1_validate_maml.py root@<vast-ip>:/workspace/scripts/
scp -P <port> scripts/phase1_merge_lora.py root@<vast-ip>:/workspace/scripts/
scp -P <port> scripts/vastai_validate_and_merge.sh root@<vast-ip>:/workspace/scripts/
```

### Step 2: SSH into Vast.ai

```bash
ssh -p <port> root@<vast-ip>
```

### Step 3: Convert Benchmarks to Test Set

```bash
cd /workspace

# Activate venv
source venv/bin/activate

# Convert benchmarks to validation format
python scripts/convert_benchmarks_to_test.py --samples-per-benchmark 100
```

This creates `data/benchmarks/validation_test.jsonl` with:
- **500 total examples** (100 per benchmark)
- **Domains:** reading comprehension, science Q&A, code generation, math, multitask understanding
- **Difficulty:** Easy (100), Medium (179), Hard (221)
- **Sources:** DROP, GPQA, HumanEval, MATH, MMLU

### Step 4: Run Validation and Merge Workflow

```bash
# Make script executable
chmod +x scripts/vastai_validate_and_merge.sh

# Run complete workflow
bash scripts/vastai_validate_and_merge.sh
```

This will:
- ✅ Validate LoRA model on benchmark test set (never seen during training)
- ✅ Merge LoRA weights into base model
- ✅ Validate merged model
- ✅ Compare results (verify identical)
- ✅ Package everything for download

**Expected Output:**
```
========================================
Phase 1D: Validation & Merge (Vast.ai)
========================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1/4: Validate LoRA Model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Computing metrics on 500 benchmark examples...
✓ LoRA validation complete

LoRA Model Results:
  Loss: 0.0XXX
  Perplexity: X.XX
  Examples: 500 (5 benchmarks)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 2/4: Merge LoRA Weights
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Merging LoRA adapter into base model...
✓ Merge complete

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 4/5: Validate Merged Model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Computing loss on 200 examples...
✓ Merged validation complete

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 5/5: Results Summary & Packaging
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VALIDATION RESULTS SUMMARY
============================================================

LoRA Model:
  Loss:       0.0XXXXX
  Perplexity: X.XXXX
  Examples:   200

Merged Model:
  Loss:       0.0XXXXX
  Perplexity: X.XXXX
  Examples:   200

Comparison:
  Loss Δ:     0.00000XXX
  PPL Δ:      0.00XXXX

✓ Models are IDENTICAL (within tolerance)
  Merge was successful!

============================================================
✓ All validations passed!
```

### Step 4: Download Results

From your **local machine**:

```bash
# Download validation results and merged model
scp -P <port> root@<vast-ip>:/workspace/download_package/*.tar.gz .

# Extract
tar -xzf validation_results.tar.gz
tar -xzf merged_model.tar.gz
```

**Downloaded Files:**
- `validation_results.tar.gz` (~1MB)
  - Test set (200 examples)
  - LoRA validation results
  - Merged validation results
  - Comparison report
- `merged_model.tar.gz` (~7GB)
  - Standalone merged model (no PEFT required)
  - Tokenizer
  - Config files

---

## Manual Step-by-Step (If Workflow Fails)

If the automated workflow fails, you can run steps manually:

### 1. Create Test Set

```bash
python /workspace/scripts/phase1_create_test_set.py \
    --train_file /workspace/data/phase1/answers/training_data_clean.jsonl \
    --output_file /workspace/data/phase1/test_set.jsonl \
    --test_size 200 \
    --stratify_by difficulty,domain \
    --seed 42 \
    --no_backup
```

**Output:**
- Test set: `data/phase1/test_set.jsonl` (200 examples)
- Statistics: `data/phase1/test_split_statistics.json`

### 2. Validate LoRA Model

```bash
python /workspace/scripts/phase1_validate_maml.py \
    --model_path /workspace/models/phase1_maml_lora_v2/best \
    --test_file /workspace/data/phase1/test_set.jsonl \
    --output_dir /workspace/results/phase1_validation \
    --skip_merged
```

**Output:**
- Results: `results/phase1_validation/lora_validation.json`

**Expected Metrics:**
- Loss < 0.10 (generalization on unseen examples)
- Perplexity < 2.0 (average)
- Easy examples: PPL < 1.5
- Hard examples: PPL < 3.0

### 3. Merge LoRA Weights

```bash
python /workspace/scripts/phase1_merge_lora.py \
    --lora_path /workspace/models/phase1_maml_lora_v2/best \
    --output_path /workspace/models/phase1_maml_lora_v2/merged \
    --precision bfloat16 \
    --verify
```

**Output:**
- Merged model: `models/phase1_maml_lora_v2/merged/` (~7GB BF16)

**Verification:**
- Compares sample weights between LoRA and merged
- Should show max difference < 1e-4

### 4. Validate Merged Model

```bash
python /workspace/scripts/phase1_validate_maml.py \
    --model_path /workspace/models/phase1_maml_lora_v2/best \
    --test_file /workspace/data/phase1/test_set.jsonl \
    --output_dir /workspace/results/phase1_validation
```

**Output:**
- Results: `results/phase1_validation/merged_validation.json`
- Comparison: `results/phase1_validation/comparison.json`

**Success Criteria:**
- ✅ Loss difference < 1e-4 (models identical)
- ✅ Perplexity difference < 0.01
- ✅ All examples produce same outputs

---

## Success Criteria

### Generalization (LoRA Model on Test Set)

- ✅ **Loss < 0.10** - Model generalizes to unseen examples
- ✅ **Perplexity < 2.0** - Reasonable confidence on new data
- ✅ **Easy PPL < 1.5** - Strong performance on easy questions
- ✅ **Hard PPL < 3.0** - Acceptable performance on hard questions

**If loss > 0.10:**
- Model may have overfit to training set
- Consider re-training with more regularization
- Or accept slight overfitting if hard examples are diverse

### Merge Verification

- ✅ **Loss Δ < 1e-4** - Numerical precision match
- ✅ **PPL Δ < 0.01** - Functionally identical
- ✅ **Weight comparison passes** - merge_lora.py verification

**If differences detected:**
- Check torch/PEFT versions match
- Verify precision settings (BF16)
- May need to re-merge with higher precision

---

## Troubleshooting

### Issue: "OOM during validation"

```bash
# Reduce batch size in validation script
python /workspace/scripts/phase1_validate_maml.py \
    --model_path ... \
    --test_file ... \
    --batch_size 2  # Default is 4
```

### Issue: "Test set too large"

```bash
# Use smaller test set
python /workspace/scripts/phase1_create_test_set.py \
    --test_size 100  # Instead of 200
    ...
```

### Issue: "Merge fails"

```bash
# Check available disk space
df -h /workspace

# Check model exists
ls -lh /workspace/models/phase1_maml_lora_v2/best/

# Try manual merge with more verbose output
python /workspace/scripts/phase1_merge_lora.py \
    --lora_path /workspace/models/phase1_maml_lora_v2/best \
    --output_path /workspace/models/phase1_maml_lora_v2/merged \
    --verify
```

---

## Next Steps After Validation

Once validation passes and models are downloaded:

### Option 1: Continue Training Pipeline

1. **Extract Base Model Responses** (Phase 1E prerequisite)
   ```bash
   python scripts/phase1_extract_base_responses.py \
       --model_path models/phase1_maml_lora_v2/merged \
       --data_file data/phase1/answers/training_data_clean.jsonl \
       --output_file data/phase1e/base_responses.jsonl
   ```
   - Time: 8-10 hours
   - Cost: ~$2-3 (cheaper GPU)

2. **Train Draft Model** (Phase 1E)
   - Model: Qwen2.5-0.5B (500M params)
   - Training: Multi-task (classifier + generator)
   - Time: 1-1.5 hours
   - Cost: ~$3-4

### Option 2: Proceed to Compression (Phase 2)

Skip draft model for now and compress base:

```bash
bash scripts/setup_phase2_compression.sh
```

- Compress base 14GB → 540MB (INT4)
- Test speculative decoding later
- Faster path to deployment

### Option 3: Test Inference Locally

Load merged model locally and test:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "models/phase1_maml_lora_v2/merged",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("models/phase1_maml_lora_v2/merged")

# Test on new question
question = "Explain binary search algorithm"
inputs = tokenizer(question, return_tensors="pt")
outputs = model.generate(**inputs, max_length=512)
print(tokenizer.decode(outputs[0]))
```

---

## Files Reference

**Scripts Created:**
- `scripts/phase1_create_test_set.py` - Stratified test set extraction
- `scripts/phase1_validate_maml.py` - Model validation on test set
- `scripts/phase1_merge_lora.py` - Merge LoRA weights into base
- `scripts/vastai_validate_and_merge.sh` - Automated workflow

**Outputs:**
- `data/phase1/test_set.jsonl` - Held-out test examples
- `results/phase1_validation/lora_validation.json` - LoRA metrics
- `results/phase1_validation/merged_validation.json` - Merged metrics
- `results/phase1_validation/comparison.json` - Comparison report
- `models/phase1_maml_lora_v2/merged/` - Standalone merged model

---

## Cost Summary

**Validation & Merge:**
- Runtime: 30-45 minutes
- GPU: H100 80GB at ~$1.50/hr
- Cost: **~$0.50-0.75**

**Total Phase 1 Cost:**
- Training v1 (failed): ~$7
- Training v2 (success): ~$25-30
- Validation & merge: ~$0.75
- **Grand Total: ~$33-38**

**Within budget!** (Planned: $40-45)

---

## Questions?

- Check logs in `/workspace/results/phase1_validation/`
- Review technical_specification.md Section 1.18
- See MAML_HYPERPARAMETER_FIX.md for training details
