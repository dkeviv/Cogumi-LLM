# Phase 1B: Failure Analysis Pipeline

3-step modular pipeline for identifying model weaknesses and preparing targeted training data.

## 📋 Overview

**Goal:** Identify where Phase 1A model fails → Generate GPT-5 targeted data → Phase 1C training

**Approach:** Separate generation from judging for efficiency and flexibility

## 📁 Current Scripts

- ✅ **step1_create_test_dataset.py** - Create curated test dataset
- ✅ **step2_generate_outputs.py** - Generate model responses  
- ✅ **step3_llm_batch_compare.py** - Smart LLM batch comparison (50x fewer API calls!)
- ✅ **step3_compare_outputs_fast.py** - Similarity metrics fallback (no API calls)
- ✅ **step3_judge_outputs.py** - Original approach (20K calls, very slow)
- ✅ **phase1b_cluster_failures.py** - Cluster similar failures
- ✅ **phase1b_label_patterns.py** - Auto-label failure patterns
- ✅ **run_complete_pipeline.sh** - Orchestrates Steps 1-3

**Note:** Old monolithic `phase1b_test_model.py` has been removed and replaced by the 3-step pipeline above.

## 🚀 Quick Start

### Run Complete Pipeline (All 3 Steps)

```bash
# Smart LLM batch comparison (1-2 hours with 405B)
./Phase1B_2_0/run_complete_pipeline.sh

# Or use 70B for faster processing (20-30 minutes)
export JUDGE_MODEL="meta-llama/Llama-3.3-70B-Instruct"
./Phase1B_2_0/run_complete_pipeline.sh
```

**Smart Batching:** 50 examples per API call = 50x fewer calls than naive approach!

### Run Individual Steps

```bash
# Step 1: Create test dataset (5-10 mins)
python "Phase1B_2_0/step1_create_test_dataset.py" \
    --dataset_path ./Phase1A_2_0/data/public_500k_filtered.jsonl \
    --output_path ./data/phase1b/test_dataset_20k.jsonl \
    --num_samples 20000

# Step 2: Generate model outputs (1-2 hours)
python "Phase1B_2_0/step2_generate_outputs.py" \
    --model_path ./Phase1A_2_0/models/phase1a_merged_10gb \
    --test_dataset ./data/phase1b/test_dataset_20k.jsonl \
    --output_path ./data/phase1b/model_outputs_20k.jsonl

# Step 3: Smart LLM batch comparison (~1-2 hours with 405B)
python "Phase1B_2_0/step3_llm_batch_compare.py" \
    --model_outputs ./data/phase1b/model_outputs_20k.jsonl \
    --output_path ./data/phase1b/comparison_results.jsonl \
    --llm_model meta-llama/Llama-3.1-405B-Instruct \
    --batch_size 50
```

## 📊 Pipeline Steps

### Step 1: Create Curated Test Dataset
- **Script:** `step1_create_test_dataset.py`
- **Time:** 5-10 minutes
- **Purpose:** Sample 20K examples with proper category representation
- **Categories:** math, code, reasoning, creative, qa, other
- **Method:** Stratified sampling ensures diverse coverage
- **Output:** 
  - `test_dataset_20k.jsonl` - Curated test examples
  - `test_dataset_stats.json` - Category distribution

### Step 2: Generate Model Outputs
- **Script:** `step2_generate_outputs.py`
- **Time:** 1-2 hours
- **Purpose:** Run merged model on test dataset once
- **Key Benefit:** Outputs reusable for multiple judging runs
- **Output:**
  - `model_outputs_20k.jsonl` - Model responses
  - `generation_stats.json` - Performance metrics

### Step 3: Smart LLM Batch Comparison
- **Script:** `step3_llm_batch_compare.py`
- **Time:** 1-2 hours (405B) or 20-30 minutes (70B)
- **Purpose:** LLM compares entire files in batches
- **Method:** Brilliant insight - give LLM BOTH files at once!
  - **Batch 50 examples per API call**
  - LLM returns JSON array of pass/fail decisions
  - **20,000 samples → ~400 API calls** (vs 20K naive approach)
- **Key Benefits:**
  - 🧠 **LLM accuracy** for nuanced comparisons (paraphrasing, math, code)
  - ⚡ **50x fewer API calls** than naive approach (400 vs 20K)
  - 💰 **50x cheaper** than individual judging
  - 🎯 **Better accuracy** than pure similarity metrics
- **Alternative:** Use `step3_compare_outputs_fast.py` for zero-cost similarity metrics
- **Output:**
  - `judged_results.jsonl` - All results with scores
  - `failures.jsonl` - Only failures (score <7)
  - `summary.json` - Statistics and category breakdown

## 🎯 After Phase 1B

```bash
# Step 4: Cluster failures
python "Phase1B_2_0/phase1b_cluster_failures.py" \
    --failures ./data/phase1b/failures.jsonl \
    --output ./data/phase1b/clusters.json

# Step 5: Label patterns
python "Phase1B_2_0/phase1b_label_patterns.py" \
    --clusters ./data/phase1b/clusters.json \
    --output ./data/phase1b/patterns.json

# Step 6: Generate targeted GPT-5 data (Phase 1C)
# Use patterns.json to guide data generation
```

## 💡 Tips

### Reusing Outputs
Once Step 2 completes, you can re-run Step 3 with different parameters:

```bash
# Try different judge model
python "Phase1B_2_0/step3_judge_outputs.py" \
    --model_outputs ./data/phase1b/model_outputs_20k.jsonl \
    --output_path ./data/phase1b/judged_results_405b.jsonl \
    --judge_model meta-llama/Llama-3.1-405B-Instruct

# Try different failure threshold
python "Phase1B_2_0/step3_judge_outputs.py" \
    --model_outputs ./data/phase1b/model_outputs_20k.jsonl \
    --output_path ./data/phase1b/judged_results_threshold6.jsonl \
    --failure_threshold 6
```

### Debugging
Inspect intermediate outputs:
- Check test dataset categories: `cat data/phase1b/test_dataset_stats.json`
- Check generation performance: `cat data/phase1b/generation_stats.json`
- Check judging summary: `cat data/phase1b/summary.json`

### Speed vs Quality
- **70B:** Recommended for initial analysis, fast iteration
- **405B:** Use for final analysis or critical domains

## 📁 Output Structure

```
data/phase1b/
├── test_dataset_20k.jsonl      # Step 1: Curated test set
├── test_dataset_stats.json     # Step 1: Category distribution
├── model_outputs_20k.jsonl     # Step 2: Model responses (REUSABLE)
├── generation_stats.json       # Step 2: Performance metrics
├── judged_results.jsonl        # Step 3: All results with scores
├── failures.jsonl              # Step 3: Only failures
├── summary.json                # Step 3: Summary statistics
├── clusters.json               # Step 4: Clustered failures
└── patterns.json               # Step 5: Labeled patterns
```

## 🔄 Advantages of 3-Step Approach

1. **Efficiency:** Generate outputs once, judge multiple times
2. **Flexibility:** Change judge model/threshold without regeneration
3. **Debugging:** Inspect outputs before comparison
4. **Cost:** ~$0 with FREE HuggingFace Inference API (405B/70B)
5. **Speed:** Smart batching = 50x faster than naive approach
6. **Tracking:** Category-level performance analysis

## ⏱️ Time Estimates

- **Step 1:** 5-10 minutes (dataset creation)
- **Step 2:** 1-2 hours (model inference on 20K samples)
- **Step 3 (405B):** 1-2 hours (smart LLM batch comparison)
- **Step 3 (70B):** 20-30 minutes (faster LLM)
- **Total (405B):** ~2.5-3.5 hours ⚡
- **Total (70B):** ~1.5-2.5 hours ⚡⚡

**Comparison:**
- Naive approach (20K calls): 68+ hours
- Smart batching (400 calls): 1-2 hours
- **Speedup: 50x faster!** 🎉

## 📊 Expected Results

- **Test samples:** 20,000 (3.1% of 640K dataset)
- **Expected failures:** 2-4K (10-20% failure rate)
- **Category coverage:** Proportional to dataset distribution
- **API calls:** ~400 (batch of 50 examples each)
- **Accuracy:** LLM-quality judgments (handles paraphrasing, math, code)
- **Cost:** $0 (FREE HuggingFace Inference API)
