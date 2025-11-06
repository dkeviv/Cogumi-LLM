#!/bin/bash
# Test Phase 1A Baseline Performance
#
# PURPOSE:
#   Re-validates Phase 1A merged model to establish solid baseline.
#   Uses same validation approach as Phase 1B validation.
#
# WHY:
#   - Verify Phase 1A baseline hasn't changed
#   - Ensure validation script works correctly
#   - Get fresh comparison data
#
# OUTPUT:
#   Results showing Phase 1A performance on 100 examples (50 MATH + 50 CODE)

set -e

echo "================================================================================"
echo "🧪 Testing Phase 1A Baseline Performance"
echo "================================================================================"
echo ""
echo "Model: checkpoints/phase1a_merged"
echo "Test Set: 100 examples (50 MATH + 50 CODE)"
echo "Purpose: Establish fresh baseline for comparison"
echo ""
echo "================================================================================"
echo ""

# Detect environment
if [ -d "/workspace" ]; then
    echo "🌐 Environment: Vast.ai"
    BASE_DIR="/workspace/data/Cogumi-LLM"
else
    echo "💻 Environment: Local"
    BASE_DIR="$(pwd)"
fi

cd "$BASE_DIR"

# Verify Phase 1A model exists
if [ ! -d "checkpoints/phase1a_merged" ]; then
    echo "❌ Error: Phase 1A merged model not found"
    echo "Expected: checkpoints/phase1a_merged"
    exit 1
fi

echo "✅ Phase 1A model found"
echo ""

# Verify benchmark results exist
if [ ! -d "checkpoints/benchmark_results_full" ]; then
    echo "❌ Error: Benchmark results not found"
    echo "Expected: checkpoints/benchmark_results_full"
    exit 1
fi

echo "✅ Benchmark results found"
echo ""

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY not set"
    echo "Setting from environment or prompting..."
    
    # Try to read from config file if exists
    if [ -f ".env" ]; then
        source .env
    fi
    
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "❌ Error: OPENAI_API_KEY required for validation"
        echo "Please set: export OPENAI_API_KEY=your_key"
        exit 1
    fi
fi

echo "✅ OpenAI API key found"
echo ""

# Debug: Show that key is set (first 10 chars only)
echo "🔑 API Key (first 10 chars): ${OPENAI_API_KEY:0:10}..."
echo ""

# Run validation
echo "🚀 Starting Phase 1A baseline validation..."
echo "-------------------------------------------"
echo "Expected results (from previous run):"
echo "  MATH: ~6% wins, ~70% ties, ~24% losses"
echo "  CODE: ~48% wins, ~20% ties, ~32% losses"
echo ""
echo "If results differ significantly, something may be wrong with:"
echo "  - The model checkpoint"
echo "  - The validation script"
echo "  - The test data"
echo ""
echo "================================================================================"
echo ""

python scripts/validate_phase1b1_optimized.py \
    --phase1a_results checkpoints/benchmark_results_full \
    --phase1b_model checkpoints/phase1a_merged \
    --openai_key "$OPENAI_API_KEY" \
    --output_dir results/phase1a_baseline_retest

echo ""
echo "================================================================================"
echo "✅ Phase 1A Baseline Validation Complete"
echo "================================================================================"
echo ""
echo "Results saved to: results/phase1a_baseline_retest/"
echo ""
echo "Compare with Phase 1B results to see if training helped or hurt performance."
echo ""
echo "================================================================================"
