# Pre-tokenization Instructions

## 🎯 Purpose
Pre-tokenize the dataset once to eliminate tokenization overhead during training (5-10% speedup).

## 📋 Prerequisites
- Virtual environment with dependencies installed
- See `scripts/requirements-stable-precompiled.txt`

## 🚀 Steps

### 1. Activate Environment
```bash
cd Phase1A_2_0/scripts
source venv_phase1a_2_0/bin/activate
```

### 2. Run Pre-tokenization (10-15 minutes)
```bash
python pretokenize_dataset.py \
    --input ../data/public_500k_filtered.jsonl \
    --output ../data/tokenized/public_500k \
    --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct"
```

### 3. Verify Output
```bash
ls -lh ../data/tokenized/public_500k/
# Should show: dataset_info.json, state.json, data-00000-of-00001.arrow
```

## 📊 Expected Results

- **Input**: `public_500k_filtered.jsonl` (870MB)
- **Output**: `tokenized/public_500k/` (~1.2GB arrow format)
- **Time**: 10-15 minutes
- **CPU cores**: Uses all available cores (num_proc=8)

## 🔧 Usage in Training

### Without Pre-tokenization (Default)
```bash
python train_phase1a_optimized_h100.py \
    --dataset_path ../data/public_500k_filtered.jsonl
```

### With Pre-tokenization (5-10% faster)
```bash
python train_phase1a_optimized_h100.py \
    --use_pretokenized \
    --dataset_path ../data/tokenized/public_500k
```

## ✅ Benefits

- ✅ 5-10% faster training (no runtime tokenization)
- ✅ Consistent tokenization across runs
- ✅ Lower CPU usage during training
- ✅ One-time cost (10-15 minutes)

## 📝 Notes

- Pre-tokenization is **optional** but recommended for H100 training
- The raw JSONL file is still used if pre-tokenized dataset is unavailable
- Pre-tokenized dataset uses Arrow format (efficient binary format)
- Tokenization uses the same model as training (Llama-3.1-8B-Instruct)

---

**Status**: Ready to run after environment setup
