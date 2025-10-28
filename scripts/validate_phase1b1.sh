#!/bin/bash
# Phase 1B.1 Optimized Validation - Reuses Phase 1A GPT-4 Responses
#
# PURPOSE:
#   Validates Phase 1B.1 by comparing with Phase 1A baseline.
#   OPTIMIZATION: Reuses GPT-4 responses from Phase 1A benchmarks (50% cost savings!)
#
# KEY INSIGHT:
#   Phase 1A benchmarks already have GPT-4 responses for the same prompts.
#   We only need to:
#   1. Load Phase 1A results (prompts + GPT-4 responses already saved)
#   2. Generate Phase 1B.1 responses for SAME prompts
#   3. Judge: Phase 1B.1 vs Phase 1A's GPT-4 (no GPT-4 regeneration!)
#
# COST SAVINGS:
#   Original: 100 prompts × (1 GPT-4 gen + 1 judge) = 200 API calls (~$1.50)
#   Optimized: 100 prompts × (1 judge only) = 100 API calls (~$0.75)
#   Savings: 50% ($0.75 saved per validation)
#
# TIME SAVINGS:
#   Original: 30-40 minutes
#   Optimized: 15-20 minutes (50% faster!)
#
# WHEN TO USE:
#   - After Phase 1B.1 training completes
#   - Requires: checkpoints/benchmark_results_full/ (Phase 1A WITH GPT-4 responses)
#   - Requires: checkpoints/phase1b_from_benchmark/ (Phase 1B.1 model)
#
# PIPELINE STAGE: Phase 1B.4 - Cost-optimized validation

set -e  # Exit on error

echo "🔍 Phase 1B.1 Validation - OPTIMIZED (Reuses GPT-4 Responses)"
echo "=============================================================="
echo ""
echo "✨ OPTIMIZATION: Reusing Phase 1A GPT-4 responses"
echo "💰 Cost savings: ~50% (~\$0.75 instead of \$1.50)"
echo "⏱️  Time savings: ~50% (15-20 min instead of 30-40 min)"
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

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY environment variable not set"
    echo ""
    echo "Set it with:"
    echo "  export OPENAI_API_KEY='your-key-here'"
    echo ""
    echo "Then run:"
    echo "  bash scripts/validate_phase1b1.sh"
    exit 1
fi

# Verify Phase 1B.1 model exists
echo "📋 Step 1: Verifying Phase 1B.1 model..."

# Check for losses_only version first (corrected training), fall back to original
if [ -d "checkpoints/phase1b_losses_only" ]; then
    PHASE1B1_MODEL="checkpoints/phase1b_losses_only"
    echo "✅ Using Phase 1B.1 (losses only - corrected): $PHASE1B1_MODEL"
elif [ -d "checkpoints/phase1b_from_benchmark" ]; then
    PHASE1B1_MODEL="checkpoints/phase1b_from_benchmark"
    echo "⚠️  Using Phase 1B.1 (original - may have catastrophic forgetting): $PHASE1B1_MODEL"
else
    echo "❌ Error: Phase 1B.1 model not found"
    echo "Looking for:"
    echo "  - checkpoints/phase1b_losses_only (preferred)"
    echo "  - checkpoints/phase1b_from_benchmark (fallback)"
    echo ""
    echo "Run Phase 1B.1 training first:"
    echo "  bash scripts/run_phase1b_losses_only.sh"
    exit 1
fi

# Verify Phase 1A results exist (CRITICAL - need GPT-4 responses!)
echo ""
echo "📋 Step 2: Verifying Phase 1A benchmark results..."
PHASE1A_RESULTS="checkpoints/benchmark_results_full"
if [ ! -d "$PHASE1A_RESULTS" ]; then
    echo "❌ Error: Phase 1A benchmark results not found at $PHASE1A_RESULTS"
    echo ""
    echo "Phase 1A benchmarks are REQUIRED (they contain GPT-4 responses we'll reuse)"
    echo "Run Phase 1A benchmarks first:"
    echo "  python scripts/automated_gpt4_benchmark.py \\"
    echo "      --model_path checkpoints/final \\"
    echo "      --openai_key \$OPENAI_API_KEY \\"
    echo "      --output_dir checkpoints/benchmark_results_full \\"
    echo "      --categories math code \\"
    echo "      --num_samples 50"
    exit 1
fi

# Verify Phase 1A has GPT-4 responses
MATH_FILE="$PHASE1A_RESULTS/math_intermediate.json"
if [ ! -f "$MATH_FILE" ]; then
    echo "❌ Error: Phase 1A math results not found: $MATH_FILE"
    exit 1
fi

# Check if Phase 1A results have gpt4_response field
HAS_GPT4=$(python3 -c "
import json
with open('$MATH_FILE') as f:
    data = json.load(f)
    has_gpt4 = 'gpt4_response' in data['results'][0] if data['results'] else False
    print('yes' if has_gpt4 else 'no')
" 2>/dev/null || echo "no")

if [ "$HAS_GPT4" = "no" ]; then
    echo "❌ Error: Phase 1A results don't have GPT-4 responses"
    echo "Re-run Phase 1A benchmarks with automated_gpt4_benchmark.py"
    exit 1
fi

echo "✅ Phase 1A results found with GPT-4 responses"
echo "✅ Ready to reuse GPT-4 responses (saving ~50% cost!)"

# Run optimized validation
echo ""
echo "📋 Step 3: Running optimized validation..."
echo ""
echo "What's happening:"
echo "  1. Loading Phase 1A prompts and GPT-4 responses (no cost)"
echo "  2. Generating Phase 1B.1 responses (no cost - local inference)"
echo "  3. Judging Phase 1B.1 vs Phase 1A's GPT-4 (100 API calls, ~\$0.75)"
echo ""
echo "Original approach would:"
echo "  1. Generate new GPT-4 responses (100 calls, ~\$0.75) ❌ SKIPPED"
echo "  2. Judge Phase 1B.1 vs new GPT-4 (100 calls, ~\$0.75)"
echo ""
echo "Starting in 5 seconds... (Ctrl+C to cancel)"
sleep 5

OUTPUT_DIR="checkpoints/benchmark_results_phase1b1"

python scripts/validate_phase1b1_optimized.py \
    --phase1a_results $PHASE1A_RESULTS \
    --phase1b_model $PHASE1B1_MODEL \
    --openai_key $OPENAI_API_KEY \
    --output_dir $OUTPUT_DIR \
    --categories math code

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Validation failed"
    exit 1
fi

echo ""
echo "✅ Validation complete!"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "Summary: $OUTPUT_DIR/validation_summary.txt"
echo ""
echo "================================================================================"
echo "DECISION CRITERIA"
echo "================================================================================"
echo ""
echo "Review the results above to decide:"
echo ""
echo "✅ SUCCESS if:"
echo "   • MATH wins improved by 3x-5x (target: 6% → 20-30%)"
echo "   • CODE wins improved by +15-35% (target: 48% → 55-65%)"
echo "   • MATH ties reduced (target: 70% → <50%)"
echo "   • Overall consistency improved"
echo ""
echo "➡️  Next steps if successful:"
echo "   1. Proceed to Phase 1B.2 (scale up to 2,000+ failures)"
echo "   2. Expected improvements: MATH 55-65%, CODE 70-75%"
echo "   3. Training time: 11-16 hours, ~\$22-30"
echo ""
echo "🔄 Next steps if not successful:"
echo "   1. Iterate Phase 1B.1 with adjusted parameters"
echo "   2. Try: increase epochs (2→3-4), adjust learning rate"
echo "   3. Extract more diverse failure examples"
echo ""
echo "================================================================================"
echo ""
echo "💡 COST NOTE: This validation saved ~\$0.75 by reusing Phase 1A GPT-4 responses!"
echo ""
