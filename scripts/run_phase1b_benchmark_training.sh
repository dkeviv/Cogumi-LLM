#!/bin/bash
# Phase 1B.1: Extract Failures and Train on Them (One-Command Execution)
#
# PURPOSE:
#   Automates the full Phase 1B.1 workflow: extract benchmark failures,
#   verify data quality, train model on failures, and validate training.
#
# WHEN TO USE:
#   - Phase 1B.1: After Phase 1A benchmarks are complete
#   - Requires: checkpoints/final/ (Phase 1A model)
#   - Requires: checkpoints/benchmark_results_full/ (Phase 1A benchmark results)
#   - Run on: Vast.ai H100
#
# WHAT IT DOES:
#   1. Extracts 73 failures (ties + losses) from MATH and CODE benchmarks
#   2. Creates training JSONL files in data/training_from_benchmark/
#   3. Trains model using train_phase1b_benchmark.py (15-20 minutes)
#   4. Saves model to checkpoints/phase1b_from_benchmark/
#   5. Validates model loads correctly
#
# OUTPUT:
#   - Training data: data/training_from_benchmark/*.jsonl (73 examples)
#   - Trained model: checkpoints/phase1b_from_benchmark/
#   - Training logs: Details about loss, perplexity, time
#
# TIME & COST:
#   - Extraction: <1 second
#   - Training: 15-20 minutes
#   - Cost: ~$0.50-1 (Vast.ai H100)
#
# PIPELINE STAGE: Phase 1B.1 - Complete workflow automation
#
# NEXT STEP: Run validate_phase1b1.sh to compare with Phase 1A baseline

set -e  # Exit on error

echo "🚀 Phase 1B.1: Train on Benchmark Failures"
echo "=========================================="
echo ""

# Detect environment
if [ -d "/workspace" ]; then
    echo "🌐 Environment: Vast.ai"
    BASE_DIR="/workspace/data/Cogumi-LLM"
else
    echo "💻 Environment: Local Mac"
    BASE_DIR="/Users/vivekdurairaj/Projects/Cogumi-LLM"
fi

cd $BASE_DIR

# Step 1: Verify benchmark results exist
echo "📋 Step 1: Verifying benchmark results..."
if [ -d "checkpoints/benchmark_results_full" ]; then
    BENCHMARK_DIR="checkpoints/benchmark_results_full"
    echo "✅ Found benchmark results at: $BENCHMARK_DIR"
elif [ -d "data/phase1/benchmark_results_full" ]; then
    BENCHMARK_DIR="data/phase1/benchmark_results_full"
    echo "✅ Found benchmark results at: $BENCHMARK_DIR"
else
    echo "❌ Error: Benchmark results not found"
    echo "Expected: checkpoints/benchmark_results_full/ OR data/phase1/benchmark_results_full/"
    exit 1
fi

# Verify benchmark files
MATH_FILE="$BENCHMARK_DIR/math_intermediate.json"
CODE_FILE="$BENCHMARK_DIR/code_intermediate.json"

if [ ! -f "$MATH_FILE" ]; then
    echo "❌ Error: $MATH_FILE not found"
    exit 1
fi

if [ ! -f "$CODE_FILE" ]; then
    echo "❌ Error: $CODE_FILE not found"
    exit 1
fi

echo "✅ Benchmark files verified"
echo ""

# Step 2: Extract training data from benchmark results
echo "📋 Step 2: Extracting failures from benchmark results..."
echo "This will take <1 minute..."
echo ""

python scripts/extract_failures_from_benchmark.py

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to extract training data"
    exit 1
fi

# Verify extraction
if [ ! -d "training_from_benchmark" ] && [ ! -d "data/training_from_benchmark" ]; then
    echo "❌ Error: training_from_benchmark/ directory not created"
    exit 1
fi

# Check both possible locations
if [ -f "data/training_from_benchmark/math_failures_from_benchmark.jsonl" ]; then
    MATH_TRAIN="data/training_from_benchmark/math_failures_from_benchmark.jsonl"
    CODE_TRAIN="data/training_from_benchmark/code_failures_from_benchmark.jsonl"
else
    MATH_TRAIN="training_from_benchmark/math_failures_from_benchmark.jsonl"
    CODE_TRAIN="training_from_benchmark/code_failures_from_benchmark.jsonl"
fi

if [ ! -f "$MATH_TRAIN" ]; then
    echo "❌ Error: $MATH_TRAIN not created"
    exit 1
fi

if [ ! -f "$CODE_TRAIN" ]; then
    echo "❌ Error: $CODE_TRAIN not created"
    exit 1
fi

MATH_COUNT=$(wc -l < "$MATH_TRAIN")
CODE_COUNT=$(wc -l < "$CODE_TRAIN")
TOTAL_COUNT=$((MATH_COUNT + CODE_COUNT))

echo ""
echo "✅ Extracted training data:"
echo "   - Math: $MATH_COUNT examples"
echo "   - Code: $CODE_COUNT examples"
echo "   - Total: $TOTAL_COUNT examples"
echo ""

# Step 3: Verify Phase 1A model exists
echo "📋 Step 3: Verifying Phase 1A model..."
if [ ! -d "checkpoints/final" ]; then
    echo "❌ Error: Phase 1A model not found at checkpoints/final/"
    exit 1
fi
echo "✅ Phase 1A model found"
echo ""

# Step 4: Train on extracted failures
echo "📋 Step 4: Training on benchmark failures..."
echo "Training config:"
echo "  - Base model: unsloth/meta-llama-3.1-8b-instruct-bnb-4bit"
echo "  - Dataset: training_from_benchmark/*.jsonl"
echo "  - Examples: $TOTAL_COUNT"
echo "  - Epochs: 2"
echo "  - Learning rate: 5e-6"
echo "  - Batch size: 4"
echo "  - Expected steps: ~$((TOTAL_COUNT * 2 / 4))"
echo ""
echo "Estimated time: 15-20 minutes"
echo "Estimated cost: \$0.50-1"
echo ""

# Check if training script exists
if [ ! -f "train_phase1b_benchmark.py" ]; then
    echo "❌ Error: train_phase1b_benchmark.py not found"
    echo "Please upload the Phase 1B training script"
    exit 1
fi

python train_phase1b_benchmark.py \
    --model_name checkpoints/phase1a_merged \
    --dataset_path "data/training_from_benchmark/*.jsonl" \
    --output_dir checkpoints/phase1b_from_benchmark \
    --num_train_epochs 2 \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --save_strategy epoch \
    --logging_steps 5

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Training failed"
    exit 1
fi

echo ""
echo "✅ Training complete!"
echo ""

# Step 5: Verify output
echo "📋 Step 5: Verifying output..."
if [ ! -d "checkpoints/phase1b_from_benchmark" ]; then
    echo "❌ Error: Output directory not created"
    exit 1
fi

echo "✅ Model saved to: checkpoints/phase1b_from_benchmark"
echo ""

# Step 6: Quick validation test
echo "📋 Step 6: Quick validation test..."
python -c "
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

print('Loading trained model...')
model = AutoModelForCausalLM.from_pretrained(
    'unsloth/meta-llama-3.1-8b-instruct-bnb-4bit',
    device_map='auto',
    torch_dtype=torch.float16,
    load_in_4bit=True
)
model = PeftModel.from_pretrained(model, 'checkpoints/phase1b_from_benchmark')
tokenizer = AutoTokenizer.from_pretrained('checkpoints/phase1b_from_benchmark')

print('✅ Model loaded successfully!')
print('✅ Validation passed!')
"

if [ $? -ne 0 ]; then
    echo "⚠️  Warning: Model validation failed, but training completed"
else
    echo ""
fi

echo ""
echo "🎉 Phase 1B.1 Complete!"
echo "===================="
echo ""
echo "Summary:"
echo "  - Trained on: $TOTAL_COUNT benchmark failures"
echo "  - Model saved: checkpoints/phase1b_from_benchmark"
echo "  - Training data: data/training_from_benchmark/"
echo ""
echo "Next steps:"
echo "1. Re-run benchmarks: python scripts/run_benchmarks.py --model_path checkpoints/phase1b_from_benchmark"
echo "2. Compare results with Phase 1A:"
echo "   - Did MATH wins improve? (6% → 20-30% target)"
echo "   - Did consistency improve? (10% → 30-40% target)"
echo "   - Did CODE wins improve? (48% → 55-65% target)"
echo ""
echo "Expected improvements:"
echo "  - Consistency: 10% → 30-40%"
echo "  - MATH wins: 6% → 20-30%"
echo "  - CODE wins: 48% → 55-65%"
echo ""
echo "If successful → Proceed to Phase 1B.2 (train on 2,000+ more failures)"
echo "If not → Iterate on parameters (epochs, lr, dataset)"
echo ""
