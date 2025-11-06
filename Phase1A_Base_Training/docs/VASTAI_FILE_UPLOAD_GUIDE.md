# Vast.ai File Structure & Upload Guide

## 📁 Current File Locations

### On Local Mac
```
/Users/vivekdurairaj/Projects/Cogumi-LLM/
├── data/
│   └── phase1/
│       ├── benchmark_results_full/          # Benchmark results (local path)
│       │   ├── math_intermediate.json
│       │   ├── code_intermediate.json
│       │   └── benchmark_report_*.json
│       └── training_from_benchmark/         # Extracted training data (READY)
│           ├── math_failures_from_benchmark.jsonl  (47 examples)
│           └── code_failures_from_benchmark.jsonl  (26 examples)
├── scripts/
│   └── extract_failures_from_benchmark.py   # Extraction script (auto-detects paths)
└── checkpoints/
    └── final/                               # Phase 1A trained model
```

### On Vast.ai
```
/workspace/data/Cogumi-LLM/                       # Main project directory
├── checkpoints/
│   ├── final/                               # Phase 1A trained model (ALREADY THERE)
│   ├── benchmark_results_full/              # Benchmark results (ALREADY THERE)
│   │   ├── math_intermediate.json
│   │   ├── code_intermediate.json
│   │   └── benchmark_report_*.json
│   └── phase1b_from_benchmark/              # Will be created by training
├── scripts/                                 # ALREADY EXISTS - use this!
│   ├── extract_failures_from_benchmark.py   # Upload here
│   ├── run_phase1b_benchmark_training.sh    # Upload here
│   └── train_qlora_optimized.py            # Already there
└── data/
    └── training_from_benchmark/             # Will be created by extraction
        ├── math_failures_from_benchmark.jsonl
        └── code_failures_from_benchmark.jsonl
```

---

## 🚀 Option 1: Upload Pre-Extracted Training Data (FASTEST) ⚡

**What:** Upload the 73 training examples already extracted on local Mac  
**Time:** <1 minute  
**Advantage:** Skip extraction step, train immediately

### Files to Upload to Vast.ai:

```bash
# On local Mac, create upload package
cd /Users/vivekdurairaj/Projects/Cogumi-LLM
mkdir -p upload_to_vastai
cp data/phase1/training_from_benchmark/*.jsonl upload_to_vastai/
```

### Upload to Vast.ai:

**Option A: Using Jupyter notebook (if available):**
```python
# In Vast.ai Jupyter notebook
!mkdir -p /workspace/data/Cogumi-LLM/data/training_from_benchmark

# Then upload files through Jupyter UI to:
# /workspace/data/Cogumi-LLM/data/training_from_benchmark/
```

**Option B: Using SCP/SSH:**
```bash
# On local Mac
scp -P <vast_ssh_port> \
  data/phase1/training_from_benchmark/*.jsonl \
  root@<vast_ip>:/workspace/data/Cogumi-LLM/data/training_from_benchmark/
```

**Option C: Using rsync:**
```bash
# On local Mac
rsync -avz -e "ssh -p <vast_ssh_port>" \
  data/phase1/training_from_benchmark/ \
  root@<vast_ip>:/workspace/data/Cogumi-LLM/data/training_from_benchmark/
```

### Then Train Immediately:

```bash
# On Vast.ai
cd /workspace/data/Cogumi-LLM
python train_qlora_optimized.py \
  --model_name unsloth/meta-llama-3.1-8b-instruct-bnb-4bit \
  --dataset_path data/training_from_benchmark/*.jsonl \
  --output_dir checkpoints/phase1b_from_benchmark \
  --num_train_epochs 2 \
  --learning_rate 5e-6 \
  --per_device_train_batch_size 4
```

**Time:** 15-20 min training  
**Cost:** $0.50-1

---

## 🔧 Option 2: Extract on Vast.ai (If Upload Fails)

**What:** Run extraction script on Vast.ai to regenerate training data  
**Time:** <1 minute extraction + 15-20 min training  
**Advantage:** Self-contained, no file transfer needed

### Files to Upload:

Only need 1 file:
```bash
# On local Mac
cd /Users/vivekdurairaj/Projects/Cogumi-LLM
# Upload this single file to Vast.ai
scripts/extract_failures_from_benchmark.py
```

### Upload Script to Vast.ai:

**Via Jupyter:**
1. Open Jupyter on Vast.ai
2. Navigate to `/workspace/data/Cogumi-LLM/scripts/`
3. Upload `extract_failures_from_benchmark.py`

**Via SCP:**
```bash
# On local Mac
scp -P <vast_ssh_port> \
  scripts/extract_failures_from_benchmark.py \
  root@<vast_ip>:/workspace/data/Cogumi-LLM/scripts/
```

### Run Extraction on Vast.ai:

```bash
# On Vast.ai
cd /workspace/data/Cogumi-LLM
python scripts/extract_failures_from_benchmark.py
```

**Output:**
- Reads from: `/workspace/data/Cogumi-LLM/checkpoints/benchmark_results_full/`
- Saves to: `/workspace/data/Cogumi-LLM/data/training_from_benchmark/`
- Creates: 73 training examples (47 MATH + 26 CODE)

### Then Train:

```bash
python train_qlora_optimized.py \
  --model_name unsloth/meta-llama-3.1-8b-instruct-bnb-4bit \
  --dataset_path data/training_from_benchmark/*.jsonl \
  --output_dir checkpoints/phase1b_from_benchmark \
  --num_train_epochs 2 \
  --learning_rate 5e-6 \
  --per_device_train_batch_size 4
```

---

## ✅ Verification Checklist

### Before Training:

**On Vast.ai, verify these paths exist:**

```bash
# Check benchmark results are present
ls -lh /workspace/data/Cogumi-LLM/checkpoints/benchmark_results_full/
# Should see: math_intermediate.json, code_intermediate.json

# Check Phase 1A model exists
ls -lh /workspace/data/Cogumi-LLM/checkpoints/final/
# Should see: adapter_config.json, adapter_model.safetensors, etc.

# Check training data is ready (after Option 1 or Option 2)
ls -lh /workspace/data/Cogumi-LLM/data/training_from_benchmark/
# Should see: math_failures_from_benchmark.jsonl, code_failures_from_benchmark.jsonl

# Verify example counts
wc -l /workspace/data/Cogumi-LLM/data/training_from_benchmark/*.jsonl
# Should show: 47 math, 26 code = 73 total
```

---

## 📊 File Summary

| File | Local Path | Vast.ai Path | Status | Size |
|------|-----------|--------------|--------|------|
| **Benchmark Results** | `data/phase1/benchmark_results_full/` | `checkpoints/benchmark_results_full/` | ✅ Already on Vast.ai | ~500KB |
| **Phase 1A Model** | `checkpoints/final/` | `checkpoints/final/` | ✅ Already on Vast.ai | ~10GB |
| **Extraction Script** | `scripts/extract_failures_from_benchmark.py` | `scripts/extract_failures_from_benchmark.py` | 📤 Need to upload | 11KB |
| **Training Data** | `data/phase1/training_from_benchmark/*.jsonl` | `training_from_benchmark/*.jsonl` | 📤 Upload OR extract | 150KB |

---

## 🎯 Recommended Workflow

**FASTEST PATH (Recommended):** ⚡

1. **Upload extraction script** (Option 2)
   ```bash
   # On local Mac
   scp -P <port> scripts/extract_failures_from_benchmark.py root@<ip>:/workspace/data/Cogumi-LLM/scripts/
   ```

2. **Extract on Vast.ai** (<1 min)
   ```bash
   # On Vast.ai
   cd /workspace/data/Cogumi-LLM
   python scripts/extract_failures_from_benchmark.py
   ```

3. **Verify extraction**
   ```bash
   wc -l data/training_from_benchmark/*.jsonl
   # Should show: 47 + 26 = 73 total
   ```

4. **Train immediately** (15-20 min, $0.50-1)
   ```bash
   python train_qlora_optimized.py \
     --dataset_path training_from_benchmark/*.jsonl \
     --output_dir checkpoints/phase1b_from_benchmark \
     --num_train_epochs 2 \
     --learning_rate 5e-6
   ```

**Total time:** ~20-25 minutes  
**Total cost:** $0.60-1.20

---

## 🐛 Troubleshooting

**Problem:** Benchmark files not found
```bash
# Check actual path on Vast.ai
find /workspace -name "*intermediate.json" 2>/dev/null
# Update BENCHMARK_DIR in script if different
```

**Problem:** Permission denied
```bash
# Fix permissions
chmod +x /workspace/data/Cogumi-LLM/scripts/extract_failures_from_benchmark.py
```

**Problem:** Training data looks wrong
```bash
# Inspect first example
head -1 /workspace/data/Cogumi-LLM/data/training_from_benchmark/math_failures_from_benchmark.jsonl | python -m json.tool
# Should show: instruction, output, category, failure_type
```

---

**Created:** October 27, 2025  
**Purpose:** Align local Mac and Vast.ai file paths for Phase 1B.1  
**Status:** Script updated with auto-detection, ready to use
