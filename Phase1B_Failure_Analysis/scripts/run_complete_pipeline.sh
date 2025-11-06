#!/bin/bash
# Phase 1B: Complete 3-Step Pipeline
# Creates test dataset, generates outputs, and judges with Llama

set -e  # Exit on error

echo "================================================================================"
echo "🚀 PHASE 1B: COMPLETE 3-STEP FAILURE ANALYSIS PIPELINE"
echo "================================================================================"
echo ""

# Configuration
MODEL_PATH="./Phase1A_2_0/models/phase1a_merged_10gb"
DATASET_PATH="./Phase1A_2_0/data/public_500k_filtered.jsonl"
OUTPUT_DIR="./data/phase1b"
NUM_SAMPLES=20000

# Judge model: Use 405B for highest quality (default) or 70B for speed
JUDGE_MODEL="${JUDGE_MODEL:-meta-llama/Llama-3.1-405B-Instruct}"
# To use 70B for faster judging: export JUDGE_MODEL="meta-llama/Llama-3.3-70B-Instruct"

echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Dataset: $DATASET_PATH"
echo "  Output directory: $OUTPUT_DIR"
echo "  Test samples: $NUM_SAMPLES"
echo "  Judge model: $JUDGE_MODEL"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Create test dataset
echo "================================================================================"
echo "📊 STEP 1/3: Creating curated test dataset..."
echo "================================================================================"
echo ""

python "Phase1B_2_0/step1_create_test_dataset.py" \
    --dataset_path "$DATASET_PATH" \
    --output_path "$OUTPUT_DIR/test_dataset_20k.jsonl" \
    --num_samples $NUM_SAMPLES

if [ $? -ne 0 ]; then
    echo "❌ Step 1 failed"
    exit 1
fi

echo ""
echo "✅ Step 1 complete!"
echo ""

# Step 2: Generate model outputs
echo "================================================================================"
echo "🤖 STEP 2/3: Generating model outputs..."
echo "================================================================================"
echo ""

python "Phase1B_2_0/step2_generate_outputs.py" \
    --model_path "$MODEL_PATH" \
    --test_dataset "$OUTPUT_DIR/test_dataset_20k.jsonl" \
    --output_path "$OUTPUT_DIR/model_outputs_20k.jsonl"

if [ $? -ne 0 ]; then
    echo "❌ Step 2 failed"
    exit 1
fi

echo ""
echo "✅ Step 2 complete!"
echo ""

# Step 3: LLM batch comparison (smart!)
echo "================================================================================"
echo "🧠 STEP 3/3: LLM Batch File Comparison (Smart Approach)..."
echo "================================================================================"
echo ""
echo "Batching 50 examples per LLM call:"
echo "  - 20K samples → ~400 API calls (50x reduction!)"
echo "  - Using ${JUDGE_MODEL:-meta-llama/Llama-3.1-405B-Instruct}"
echo "  - Estimated time: ~1-2 hours (vs 68 hours for individual calls)"
echo ""

python "Phase1B_2_0/step3_llm_batch_compare.py" \
    --model_outputs "$OUTPUT_DIR/model_outputs_20k.jsonl" \
    --output_path "$OUTPUT_DIR/comparison_results.jsonl" \
    --llm_model "${JUDGE_MODEL:-meta-llama/Llama-3.1-405B-Instruct}" \
    --batch_size 50

if [ $? -ne 0 ]; then
    echo "❌ Step 3 failed"
    exit 1
fi

echo ""
echo "✅ Step 3 complete!"
echo ""

# Summary
echo "================================================================================"
echo "🎉 PHASE 1B COMPLETE!"
echo "================================================================================"
echo ""
echo "Output files:"
echo "  📊 Test dataset: $OUTPUT_DIR/test_dataset_20k.jsonl"
echo "  🤖 Model outputs: $OUTPUT_DIR/model_outputs_20k.jsonl"
echo "  ⚡ Comparison results: $OUTPUT_DIR/comparison_results.jsonl"
echo "  ❌ Failures: $OUTPUT_DIR/failures.jsonl"
echo "  📈 Summary: $OUTPUT_DIR/summary.json"
echo ""
echo "Next steps:"
echo "  1. Review summary.json for overall performance"
echo "  2. Analyze failures.jsonl to identify patterns"
echo "  3. Run clustering: python Phase1B_2_0/phase1b_cluster_failures.py"
echo "  4. Run labeling: python Phase1B_2_0/phase1b_label_patterns.py"
echo ""
echo "Optional: Validate a sample with LLM judge"
echo "  python Phase1B_2_0/step3_judge_outputs.py \\"
echo "      --model_outputs $OUTPUT_DIR/model_outputs_20k.jsonl \\"
echo "      --output_path $OUTPUT_DIR/llm_validation.jsonl \\"
echo "      --sample_size 100"
echo ""
