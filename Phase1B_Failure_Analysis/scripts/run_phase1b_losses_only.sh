#!/bin/bash
# Phase 1B.1 Training - TRUE LOSSES ONLY (Fix for Catastrophic Forgetting)
#
# CRITICAL FIX:
#   Original training used 73 examples (45 ties + 28 losses)
#   Training on "ties" caused catastrophic forgetting (model unlearned correct answers)
#   
#   Results from bad training:
#   - MATH ties: 70% → 18% (model changed correct answers!)
#   - MATH losses: 24% → 78% (catastrophic regression)
#   
#   This script trains on ONLY 28 true losses (no ties)
#
# EXPECTED IMPROVEMENTS:
#   - MATH: Should improve without forgetting (ties should stay ~70%)
#   - CODE: Should improve on the 16 genuine failures
#   - No catastrophic forgetting (base knowledge preserved)

set -e  # Exit on error

echo "================================================================================"
echo "🔧 Phase 1B.1 Training - TRUE LOSSES ONLY"
echo "================================================================================"
echo ""
echo "Dataset: 28 examples (12 MATH + 16 CODE losses only)"
echo "Base model: checkpoints/phase1a_merged"
echo "Output: checkpoints/phase1b_losses_only"
echo ""
echo "FIX: Removed 45 'tie' examples that caused catastrophic forgetting"
echo ""
echo "================================================================================"
echo ""

# Activate virtual environment
if [ -d "/workspace/data/Cogumi-LLM/venv" ]; then
    source /workspace/data/Cogumi-LLM/venv/bin/activate
    echo "✅ Virtual environment activated"
elif [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️  No venv found, using system Python"
fi

# Navigate to project root
if [ -d "/workspace/data/Cogumi-LLM" ]; then
    cd /workspace/data/Cogumi-LLM
    echo "✅ Changed to: /workspace/data/Cogumi-LLM"
elif [ -d "$HOME/Cogumi-LLM" ]; then
    cd "$HOME/Cogumi-LLM"
    echo "✅ Changed to: $HOME/Cogumi-LLM"
else
    echo "✅ Already in project directory: $(pwd)"
fi

# Verify filtered data exists
echo "🔍 Checking for filtered data..."
echo "Current directory: $(pwd)"
echo "Files in training_from_benchmark:"
ls -lh data/phase1/training_from_benchmark/ 2>/dev/null || echo "Directory not found!"
echo ""

if [ ! -f "data/phase1/training_from_benchmark/math_losses_only.jsonl" ]; then
    echo "❌ Filtered data not found!"
    echo "Looking for: data/phase1/training_from_benchmark/math_losses_only.jsonl"
    echo ""
    echo "Please run: python scripts/filter_true_losses.py"
    exit 1
fi

echo "✅ Filtered data found!"
echo ""
echo "📊 Dataset Statistics:"
echo "-------------------------------------------"
wc -l data/phase1/training_from_benchmark/*_losses_only.jsonl
echo ""

# Training configuration
MODEL_NAME="checkpoints/phase1a_merged"
DATASET_PATH="data/phase1/training_from_benchmark/*_losses_only.jsonl"
OUTPUT_DIR="checkpoints/phase1b_losses_only"
EPOCHS=2
LEARNING_RATE=5e-6

echo "🚀 Starting training..."
echo "-------------------------------------------"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
echo ""

# Run training
python train_phase1b_benchmark.py \
    --model_name "$MODEL_NAME" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $EPOCHS \
    --learning_rate $LEARNING_RATE

echo ""
echo "================================================================================"
echo "✅ Training Complete!"
echo "================================================================================"
echo ""
echo "NEXT STEPS:"
echo "-------------------------------------------"
echo "1. Validate with optimized script (15-20 min, \$0.75):"
echo "   bash scripts/validate_phase1b1.sh"
echo ""
echo "2. Expected improvements (without forgetting):"
echo "   - MATH wins: Should improve on the 12 genuine failures"
echo "   - MATH ties: Should STAY ~70% (no catastrophic forgetting)"
echo "   - CODE wins: Should improve on the 16 genuine failures"
echo ""
echo "3. Success criteria:"
echo "   - MATH losses should DECREASE (not increase to 78%!)"
echo "   - MATH ties should STAY HIGH (model keeps correct answers)"
echo "   - Overall improvement without knowledge loss"
echo ""
echo "================================================================================"
