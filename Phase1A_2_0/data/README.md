# Phase 1A 2.0 - Data Folder

## ⚠️ Dataset Not in Git Repository

**Important:** The dataset file `public_500k_filtered.jsonl` (870MB) is **NOT included in the git repository** due to GitHub's 100MB file size limit.

### How to Get the Dataset

**On Vast.ai or Training Instance:**
After cloning the repository, download the dataset:

```bash
# Navigate to Phase1A_2_0/data folder
cd /workspace/Cogumi-LLM/Phase1A_2_0/data

# Download from GitHub releases
wget https://github.com/dkeviv/Cogumi-LLM/releases/download/v1.0-dataset/public_500k_filtered.jsonl

# Verify download (should be ~870MB)
ls -lh public_500k_filtered.jsonl

# Verify line count (should be 600,000)
wc -l public_500k_filtered.jsonl
```

**Alternative - Copy from Phase0:**
If Phase0 dataset is available:
```bash
cp ../../Phase0/data/public_500k_filtered.jsonl ./
```

---

## 📁 Folder Structure

```
data/Phase1A_2_0/
├── README.md                           # This file
├── public_500k_filtered.jsonl         # Training dataset (600K examples, 870MB) - NOT IN GIT
├── tokenized/                          # Pre-tokenized dataset (for faster training)
├── checkpoints/                        # Training checkpoints (saved during training)
├── merged/                             # Merged full-precision models
└── logs/                               # Training logs and metrics
```

---

## 📊 Dataset Information

### `public_500k_filtered.jsonl`
- **Source**: Phase 0 multi-teacher distillation
- **Teachers**: Llama-405B (40%), GPT-4o (35%), Qwen3-Coder-480B (25%)
- **Size**: 600,000 examples (870MB)
- **Quality**: 8.2/10 average (GPT-4-mini scored)
- **Deduplication**: MinHash LSH @ Jaccard 0.8 (removed 150K duplicates)
- **Format**: JSONL with `instruction`, `response`, `quality_score` fields

**Usage**:
```python
import jsonlines

with jsonlines.open('data/Phase1A_2_0/public_500k_filtered.jsonl') as f:
    for example in f:
        instruction = example['instruction']
        response = example['response']
        quality = example.get('quality_score', 0)
```

---

## � Tokenized Folder

Pre-tokenized dataset for faster training (5-10% speedup):
- **Purpose**: Eliminate tokenization overhead during training
- **Size**: ~1.2GB (compressed tensor format)
- **Creation time**: 10-15 minutes (one-time preprocessing)

**Create Tokenized Dataset**:
```bash
cd ../scripts
python pretokenize_dataset.py \
    --input ../data/public_500k_filtered.jsonl \
    --output ../data/tokenized/public_500k \
    --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct"
```

**Usage in Training**:
```bash
python train_phase1a_optimized_h100.py \
    --use_pretokenized \
    --dataset_path "../data/tokenized/public_500k"
```

**Benefits**:
- ✅ 5-10% faster training (no runtime tokenization)
- ✅ Consistent tokenization across runs
- ✅ Lower CPU usage during training

---

## �💾 Checkpoints Folder

Training checkpoints saved during Phase 1A training:
- **checkpoint-5000**: After 5,000 training steps
- **checkpoint-10000**: After 10,000 training steps
- **checkpoint-15000**: After 15,000 training steps
- **checkpoint-20000**: After 20,000 training steps
- **checkpoint-final**: Final checkpoint after ~28,000 steps (3 epochs)

**Structure**:
```
checkpoints/
├── checkpoint-5000/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── training_args.bin
├── checkpoint-10000/
└── checkpoint-final/
```

---

## 🔀 Merged Folder

Full-precision merged models (adapter + base):
- **merged_phase1a_v2.0**: Final merged model after Phase 1A 2.0 training
- Expected size: ~10GB (full precision)
- Expected performance: 89-91% GPT-4 baseline

**Merge Command**:
```bash
python scripts/Phase1A_2_0/merge_adapter_fullprecision.py \
    --base_model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --adapter_path "data/Phase1A_2_0/checkpoints/checkpoint-final" \
    --output_path "data/Phase1A_2_0/merged/merged_phase1a_v2.0"
```

---

## 📝 Logs Folder

Training logs and metrics:
- **training.log**: Main training log with loss, perplexity, etc.
- **tensorboard/**: TensorBoard event files
- **wandb/**: Weights & Biases logs (if enabled)
- **validation_results.json**: Validation metrics after training

---

## 🚀 Training Usage

### Copy Dataset to Local NVMe (20-30% speedup)
```bash
# On H100 instance
cp data/Phase1A_2_0/public_500k_filtered.jsonl /tmp/dataset.jsonl
```

### Start Training
```bash
python scripts/Phase1A_2_0/train_phase1a_optimized_h100.py \
    --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --dataset_path "/tmp/dataset.jsonl" \
    --output_dir "data/Phase1A_2_0/checkpoints" \
    --logging_dir "data/Phase1A_2_0/logs" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --save_steps 2000
```

**Expected Results**:
- Training time: 8-12 hours on H100 80GB
- Cost: $20-30 @ $2.50/hr
- Final model: 89-91% GPT-4 performance

---

## ✅ Validation

After training, validate the merged model:
```bash
# Merge adapter to base
python scripts/Phase1A_2_0/merge_adapter_fullprecision.py

# Validate performance
python scripts/validate_merged_model.py \
    --model_path "data/Phase1A_2_0/merged/merged_phase1a_v2.0"
```

**Expected Validation Results**:
- MATH: 6% wins, 70% ties, 24% losses (vs GPT-4)
- CODE: 48% wins, 20% ties, 32% losses (vs GPT-4)
- Overall: 89-91% GPT-4 equivalent

---

## 🔗 References

- **Training Scripts**: `scripts/Phase1A_2_0/`
- **Documentation**: `scripts/Phase1A_2_0/README.md`
- **Requirements**: `scripts/Phase1A_2_0/requirements-stable-precompiled.txt`
- **Dataset Source**: `data/phase1/public_500k_filtered.jsonl` (original)

---

**Status**: ✅ Ready for Phase 1A 2.0 training with full precision base model
