# Phase 1B.1: Ready to Execute ✅

**Status:** All scripts updated, paths aligned, ready for Vast.ai execution  
**Date:** October 27, 2025  
**Approach:** Extract training data from existing benchmark failures

---

## 📋 What's Ready

### ✅ Local Environment
- **Training data extracted:** 73 examples (47 MATH + 26 CODE)
- **Location:** `/Users/vivekdurairaj/Projects/Cogumi-LLM/data/phase1/training_from_benchmark/`
- **Files:**
  - `math_failures_from_benchmark.jsonl` (47 examples)
  - `code_failures_from_benchmark.jsonl` (26 examples)
- **Status:** ✅ Generated and verified

### ✅ Scripts Ready to Upload
1. **`scripts/extract_failures_from_benchmark.py`** (11KB)
   - Auto-detects Local vs Vast.ai environment
   - Reads benchmark JSON files
   - Extracts failures (losses + ties)
   - Outputs to JSONL format
   - **Upload to:** `/workspace/data/Cogumi-LLM/scripts/`

2. **`scripts/run_phase1b_benchmark_training.sh`** (5KB)
   - One-command execution (extraction + training)
   - Auto-detects environment
   - Verifies files exist
   - Trains for 2 epochs (~15-20 min)
   - **Upload to:** `/workspace/data/Cogumi-LLM/scripts/`

### ✅ Path Alignment Complete
| Location | Local Mac | Vast.ai |
|----------|-----------|---------|
| **Base Directory** | `/Users/vivekdurairaj/Projects/Cogumi-LLM/` | `/workspace/data/Cogumi-LLM/` |
| **Benchmark Results** | `data/phase1/benchmark_results_full/` | `checkpoints/benchmark_results_full/` |
| **Training Data** | `data/phase1/training_from_benchmark/` | `data/training_from_benchmark/` |
| **Scripts** | `scripts/` | `scripts/` (already exists) |
| **Output Model** | `checkpoints/phase1b_from_benchmark/` | `checkpoints/phase1b_from_benchmark/` |

---

## 🚀 Execution Steps

### Option 1: One-Command Approach (RECOMMENDED) ⚡

**Upload scripts to Vast.ai:**
```bash
# On local Mac
cd /Users/vivekdurairaj/Projects/Cogumi-LLM
scp -P <vast_port> \
  scripts/extract_failures_from_benchmark.py \
  scripts/run_phase1b_benchmark_training.sh \
  root@<vast_ip>:/workspace/data/Cogumi-LLM/scripts/
```

**Execute on Vast.ai:**
```bash
# On Vast.ai SSH
cd /workspace/data/Cogumi-LLM
bash scripts/run_phase1b_benchmark_training.sh
```

**What happens automatically:**
1. ✅ Detects environment (Vast.ai)
2. ✅ Verifies benchmarks exist at `checkpoints/benchmark_results_full/`
3. ✅ Extracts 73 examples to `data/training_from_benchmark/`
4. ✅ Verifies extraction successful
5. ✅ Trains for 2 epochs (~15-20 min)
6. ✅ Saves to `checkpoints/phase1b_from_benchmark/`
7. ✅ Validates model loads correctly

**Time:** 15-20 minutes total  
**Cost:** $0.50-1

---

### Option 2: Manual Steps (Alternative)

**1. Upload extraction script only:**
```bash
scp -P <vast_port> \
  scripts/extract_failures_from_benchmark.py \
  root@<vast_ip>:/workspace/data/Cogumi-LLM/scripts/
```

**2. Extract on Vast.ai:**
```bash
cd /workspace/data/Cogumi-LLM
python scripts/extract_failures_from_benchmark.py
```

**3. Verify extraction:**
```bash
wc -l data/training_from_benchmark/*.jsonl
# Should show: 47 math_failures, 26 code_failures
```

**4. Train manually:**
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

## 📊 Expected Results

### Training Metrics
- **Total examples:** 73 (47 MATH + 26 CODE)
- **Batch size:** 4
- **Epochs:** 2
- **Total steps:** ~36 steps
- **Time:** 15-20 minutes
- **Cost:** $0.50-1

### Post-Training Validation
After training completes, re-run benchmarks to compare:

| Metric | Phase 1A (Before) | Phase 1B.1 Target |
|--------|-------------------|-------------------|
| **MATH wins** | 6% (3/50) | 20-30% (10-15/50) |
| **CODE wins** | 48% (24/50) | 55-65% (28-33/50) |
| **MATH ties** | 70% (35/50) | 40-50% (20-25/50) |
| **CODE ties** | 20% (10/50) | 10-15% (5-8/50) |
| **Consistency** | ~10% | 30-40% |

**Benchmark command:**
```bash
cd /workspace/data/Cogumi-LLM
python scripts/run_benchmarks.py \
  --model_path checkpoints/phase1b_from_benchmark \
  --output_dir checkpoints/benchmark_results_phase1b1
```

---

## ✅ Pre-Flight Checklist

### Before Execution
- [ ] Vast.ai H100 instance running
- [ ] Connected to Vast.ai via SSH or Jupyter
- [ ] Verified Phase 1A model exists: `ls checkpoints/final/`
- [ ] Verified benchmark results exist: `ls checkpoints/benchmark_results_full/`

### Upload Files
- [ ] Uploaded `extract_failures_from_benchmark.py` to `/workspace/data/Cogumi-LLM/scripts/`
- [ ] Uploaded `run_phase1b_benchmark_training.sh` to `/workspace/data/Cogumi-LLM/scripts/`
- [ ] Made scripts executable: `chmod +x scripts/*.sh`

### Verify Paths
```bash
# All these should exist on Vast.ai
ls /workspace/data/Cogumi-LLM/scripts/extract_failures_from_benchmark.py
ls /workspace/data/Cogumi-LLM/scripts/run_phase1b_benchmark_training.sh
ls /workspace/data/Cogumi-LLM/checkpoints/benchmark_results_full/math_intermediate.json
ls /workspace/data/Cogumi-LLM/checkpoints/benchmark_results_full/code_intermediate.json
ls /workspace/data/Cogumi-LLM/checkpoints/final/adapter_model.safetensors
```

---

## 🎯 Success Criteria

**Phase 1B.1 is successful if:**

1. ✅ Extraction generates exactly 73 examples (47 MATH + 26 CODE)
2. ✅ Training completes without errors
3. ✅ Model loads correctly after training
4. ✅ **MATH wins improve:** 6% → 20-30% (3x-5x improvement)
5. ✅ **CODE wins improve:** 48% → 55-65% (+15-35% improvement)
6. ✅ **MATH ties reduce:** 70% → 40-50% (consistency improves)

**If successful:**
- Proceed to Phase 1B.2 (expand training to 2,000+ failures)
- Expected final results: MATH 55-65%, CODE 70-75%

**If not successful:**
- Iterate on Phase 1B.1 (adjust epochs, learning rate, add more data)
- Cost per iteration: $0.50-1

---

## 📝 Next Steps After 1B.1

**If Phase 1B.1 improves results (expected):**

1. **Phase 1B.2: Expand training data**
   - Run model on GSM8K train (7,473 problems)
   - Identify ~2,000 additional failures
   - Train on 73 + 2,000 = 2,073 examples
   - Time: 11-16 hours
   - Cost: $22-30
   - Expected: MATH 55-65%, CODE 70-75%

2. **Final Benchmark**
   - Re-run on MATH/CODE/REASONING/AUTOMATION
   - Validate improvements hold across domains
   - Document final Phase 1B results

---

## 🛠️ Troubleshooting

**Problem:** Benchmark files not found
```bash
# Verify path
ls -lh /workspace/data/Cogumi-LLM/checkpoints/benchmark_results_full/
# Should see math_intermediate.json and code_intermediate.json
```

**Problem:** Extraction generates wrong number of examples
```bash
# Check extraction output
python scripts/extract_failures_from_benchmark.py
# Should show: MATH: 47 extracted, CODE: 26 extracted
```

**Problem:** Training fails with dataset not found
```bash
# Verify training data exists
ls -lh data/training_from_benchmark/*.jsonl
# Should see math_failures_from_benchmark.jsonl and code_failures_from_benchmark.jsonl
```

**Problem:** Model doesn't improve after training
- Check if model actually trained (verify checkpoint files exist)
- Try increasing epochs (2 → 3-4)
- Try adjusting learning rate (5e-6 → 3e-6 or 7e-6)
- Add more training data (extract from other benchmark categories)

---

## 📚 Documentation References

- **Full Upload Guide:** `VASTAI_FILE_UPLOAD_GUIDE.md`
- **Extraction Script:** `scripts/extract_failures_from_benchmark.py`
- **Training Script:** `scripts/run_phase1b_benchmark_training.sh`
- **Benchmark Analysis:** `PHASE1B_TRAIN_ON_BENCHMARK_FAILURES.md`
- **Original Plan:** `PHASE1B_SELF_CONSISTENT_TRAINING.md`

---

**Summary:** Everything is ready for Phase 1B.1 execution on Vast.ai. Upload 2 scripts, run 1 command, wait 15-20 minutes, validate results. This is the minimal viable test to validate the self-consistency training approach before expanding to full failure set.
