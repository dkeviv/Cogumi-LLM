#!/bin/bash
# Validate Phase 1B.1 Results - Compare with Phase 1A Baseline
# Runs GPT-4 comparison benchmarks on Phase 1B.1 model

set -e  # Exit on error

echo "🔍 Phase 1B.1 Validation - GPT-4 Comparison Benchmarks"
echo "======================================================="
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
    echo "Please set it with:"
    echo "  export OPENAI_API_KEY='your-key-here'"
    echo ""
    echo "Or for Vast.ai SSH:"
    echo "  ssh -p <port> root@<ip> 'cd /workspace/data/Cogumi-LLM && export OPENAI_API_KEY=your-key && bash scripts/validate_phase1b1.sh'"
    exit 1
fi

# Verify Phase 1B.1 model exists
echo "📋 Step 1: Verifying Phase 1B.1 model..."
PHASE1B1_MODEL="checkpoints/phase1b_from_benchmark"
if [ ! -d "$PHASE1B1_MODEL" ]; then
    echo "❌ Error: Phase 1B.1 model not found"
    echo "Expected: $PHASE1B1_MODEL"
    echo "Did training complete successfully?"
    exit 1
fi
echo "✅ Phase 1B.1 model found: $PHASE1B1_MODEL"
echo ""

# Verify Phase 1A results exist for comparison
echo "📋 Step 2: Checking Phase 1A baseline results..."
PHASE1A_RESULTS="checkpoints/benchmark_results_full"
if [ ! -d "$PHASE1A_RESULTS" ]; then
    echo "⚠️  Warning: Phase 1A benchmark results not found at $PHASE1A_RESULTS"
    echo "Will run benchmarks but cannot compare with baseline"
    HAS_BASELINE=false
else
    echo "✅ Phase 1A baseline found: $PHASE1A_RESULTS"
    HAS_BASELINE=true
fi
echo ""

# Run benchmarks on Phase 1B.1 model
echo "📋 Step 3: Running GPT-4 comparison benchmarks..."
echo ""
echo "⏱️  Estimated time: 30-40 minutes"
echo "💰 Estimated cost: ~\$1-1.50 (GPT-4 API)"
echo ""
echo "Testing 50 samples each:"
echo "  • MATH: GSM8K math problems"
echo "  • CODE: HumanEval coding tasks"
echo ""
echo "For each problem:"
echo "  1. Generate Phase 1B.1 model response"
echo "  2. Generate GPT-4 response"
echo "  3. Use GPT-4 to judge which is better"
echo ""
echo "Starting in 5 seconds... (Ctrl+C to cancel)"
sleep 5

OUTPUT_DIR="checkpoints/benchmark_results_phase1b1"
mkdir -p $OUTPUT_DIR

# Run benchmark with GPT-4 comparison
python scripts/automated_gpt4_benchmark.py \
    --model_path $PHASE1B1_MODEL \
    --openai_key $OPENAI_API_KEY \
    --output_dir $OUTPUT_DIR \
    --categories math code \
    --num_samples 50

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Benchmark failed"
    exit 1
fi

echo ""
echo "✅ Benchmarks complete!"
echo ""

# Extract and display results
echo "================================================================================"
echo "📊 PHASE 1B.1 VALIDATION RESULTS"
echo "================================================================================"
echo ""

# Function to extract stats from intermediate JSON
extract_stats() {
    local file=$1
    local category=$2
    
    if [ ! -f "$file" ]; then
        echo "⚠️  Warning: $category results file not found: $file"
        return 1
    fi
    
    # Extract wins, losses, ties
    local wins=$(python3 -c "import json; d=json.load(open('$file')); print(d.get('wins', 0))")
    local losses=$(python3 -c "import json; d=json.load(open('$file')); print(d.get('losses', 0))")
    local ties=$(python3 -c "import json; d=json.load(open('$file')); print(d.get('ties', 0))")
    local total=$((wins + losses + ties))
    
    # Calculate percentages
    local win_pct=0
    local tie_pct=0
    if [ $total -gt 0 ]; then
        win_pct=$(python3 -c "print(f'{$wins/$total*100:.1f}')")
        tie_pct=$(python3 -c "print(f'{$ties/$total*100:.1f}')")
    fi
    
    # Return as comma-separated string
    echo "$wins,$losses,$ties,$total,$win_pct,$tie_pct"
}

# Extract Phase 1B.1 results
echo "Phase 1B.1 Model Results:"
echo "------------------------"

MATH_1B_FILE="$OUTPUT_DIR/math_intermediate.json"
CODE_1B_FILE="$OUTPUT_DIR/code_intermediate.json"

if [ -f "$MATH_1B_FILE" ]; then
    IFS=',' read -r MATH_1B_WINS MATH_1B_LOSSES MATH_1B_TIES MATH_1B_TOTAL MATH_1B_WIN_PCT MATH_1B_TIE_PCT <<< $(extract_stats "$MATH_1B_FILE" "MATH")
    echo "  MATH: $MATH_1B_WINS wins, $MATH_1B_LOSSES losses, $MATH_1B_TIES ties (out of $MATH_1B_TOTAL)"
    echo "        Win rate: $MATH_1B_WIN_PCT%, Tie rate: $MATH_1B_TIE_PCT%"
else
    echo "  ⚠️  MATH results not found"
fi

if [ -f "$CODE_1B_FILE" ]; then
    IFS=',' read -r CODE_1B_WINS CODE_1B_LOSSES CODE_1B_TIES CODE_1B_TOTAL CODE_1B_WIN_PCT CODE_1B_TIE_PCT <<< $(extract_stats "$CODE_1B_FILE" "CODE")
    echo "  CODE: $CODE_1B_WINS wins, $CODE_1B_LOSSES losses, $CODE_1B_TIES ties (out of $CODE_1B_TOTAL)"
    echo "        Win rate: $CODE_1B_WIN_PCT%, Tie rate: $CODE_1B_TIE_PCT%"
else
    echo "  ⚠️  CODE results not found"
fi

echo ""

# Compare with Phase 1A baseline if available
if [ "$HAS_BASELINE" = true ]; then
    echo "Phase 1A Baseline (for comparison):"
    echo "----------------------------------"
    
    MATH_1A_FILE="$PHASE1A_RESULTS/math_intermediate.json"
    CODE_1A_FILE="$PHASE1A_RESULTS/code_intermediate.json"
    
    if [ -f "$MATH_1A_FILE" ]; then
        IFS=',' read -r MATH_1A_WINS MATH_1A_LOSSES MATH_1A_TIES MATH_1A_TOTAL MATH_1A_WIN_PCT MATH_1A_TIE_PCT <<< $(extract_stats "$MATH_1A_FILE" "MATH")
        echo "  MATH: $MATH_1A_WINS wins, $MATH_1A_LOSSES losses, $MATH_1A_TIES ties (out of $MATH_1A_TOTAL)"
        echo "        Win rate: $MATH_1A_WIN_PCT%, Tie rate: $MATH_1A_TIE_PCT%"
    fi
    
    if [ -f "$CODE_1A_FILE" ]; then
        IFS=',' read -r CODE_1A_WINS CODE_1A_LOSSES CODE_1A_TIES CODE_1A_TOTAL CODE_1A_WIN_PCT CODE_1A_TIE_PCT <<< $(extract_stats "$CODE_1A_FILE" "CODE")
        echo "  CODE: $CODE_1A_WINS wins, $CODE_1A_LOSSES losses, $CODE_1A_TIES ties (out of $CODE_1A_TOTAL)"
        echo "        Win rate: $CODE_1A_WIN_PCT%, Tie rate: $CODE_1A_TIE_PCT%"
    fi
    
    echo ""
    echo "Comparison Summary:"
    echo "------------------"
    
    if [ -f "$MATH_1A_FILE" ] && [ -f "$MATH_1B_FILE" ]; then
        MATH_WIN_IMPROVEMENT=$(python3 -c "print(f'{$MATH_1B_WIN_PCT - $MATH_1A_WIN_PCT:+.1f}')")
        MATH_TIE_CHANGE=$(python3 -c "print(f'{$MATH_1B_TIE_PCT - $MATH_1A_TIE_PCT:+.1f}')")
        
        echo "  MATH:"
        echo "    Win rate:  $MATH_1A_WIN_PCT% → $MATH_1B_WIN_PCT% ($MATH_WIN_IMPROVEMENT percentage points)"
        echo "    Tie rate:  $MATH_1A_TIE_PCT% → $MATH_1B_TIE_PCT% ($MATH_TIE_CHANGE percentage points)"
        echo "    Target:    20-30% wins, <50% ties"
    fi
    
    if [ -f "$CODE_1A_FILE" ] && [ -f "$CODE_1B_FILE" ]; then
        CODE_WIN_IMPROVEMENT=$(python3 -c "print(f'{$CODE_1B_WIN_PCT - $CODE_1A_WIN_PCT:+.1f}')")
        CODE_TIE_CHANGE=$(python3 -c "print(f'{$CODE_1B_TIE_PCT - $CODE_1A_TIE_PCT:+.1f}')")
        
        echo "  CODE:"
        echo "    Win rate:  $CODE_1A_WIN_PCT% → $CODE_1B_WIN_PCT% ($CODE_WIN_IMPROVEMENT percentage points)"
        echo "    Tie rate:  $CODE_1A_TIE_PCT% → $CODE_1B_TIE_PCT% ($CODE_TIE_CHANGE percentage points)"
        echo "    Target:    55-65% wins, <15% ties"
    fi
fi

echo ""
echo "================================================================================"
echo "DECISION CRITERIA"
echo "================================================================================"
echo ""
echo "✅ SUCCESS if:"
echo "   • MATH wins improved by 3x-5x (target: 6% → 20-30%)"
echo "   • CODE wins improved by +15-35% (target: 48% → 55-65%)"
echo "   • MATH ties reduced significantly (target: 70% → 40-50%)"
echo "   • Overall consistency improved"
echo ""
echo "➡️  If successful:"
echo "   → Proceed to Phase 1B.2 (expand training to 2,000+ failures)"
echo "   → Extract failures from GSM8K train set (7,473 problems)"
echo "   → Expected training: 11-16 hours, ~\$22-30"
echo ""
echo "🔄 If not successful:"
echo "   → Iterate Phase 1B.1 with adjusted parameters"
echo "   → Try: increase epochs (2→3-4), adjust learning rate"
echo "   → Add more diverse failure examples"
echo ""
echo "================================================================================"
echo ""

# Save detailed summary
SUMMARY_FILE="$OUTPUT_DIR/validation_summary.txt"
{
    echo "Phase 1B.1 Validation Summary"
    echo "============================="
    echo ""
    echo "Date: $(date)"
    echo ""
    echo "Training Configuration:"
    echo "  - Dataset: 73 benchmark failures (47 MATH + 26 CODE)"
    echo "  - Source: Phase 1A benchmark ties + losses"
    echo "  - Epochs: 2"
    echo "  - Learning rate: 5e-6"
    echo "  - Batch size: 4, Accumulation: 4 (Effective: 16)"
    echo "  - Model saved: $PHASE1B1_MODEL"
    echo ""
    echo "Benchmark Results (50 samples each):"
    if [ -f "$MATH_1B_FILE" ]; then
        echo "  MATH: $MATH_1B_WINS/$MATH_1B_TOTAL wins ($MATH_1B_WIN_PCT%), $MATH_1B_TIES ties ($MATH_1B_TIE_PCT%)"
    fi
    if [ -f "$CODE_1B_FILE" ]; then
        echo "  CODE: $CODE_1B_WINS/$CODE_1B_TOTAL wins ($CODE_1B_WIN_PCT%), $CODE_1B_TIES ties ($CODE_1B_TIE_PCT%)"
    fi
    echo ""
    
    if [ "$HAS_BASELINE" = true ]; then
        echo "Comparison with Phase 1A Baseline:"
        if [ -f "$MATH_1A_FILE" ] && [ -f "$MATH_1B_FILE" ]; then
            echo "  MATH: $MATH_1A_WIN_PCT% → $MATH_1B_WIN_PCT% ($MATH_WIN_IMPROVEMENT pp)"
        fi
        if [ -f "$CODE_1A_FILE" ] && [ -f "$CODE_1B_FILE" ]; then
            echo "  CODE: $CODE_1A_WIN_PCT% → $CODE_1B_WIN_PCT% ($CODE_WIN_IMPROVEMENT pp)"
        fi
        echo ""
    fi
    
    echo "Detailed results saved in: $OUTPUT_DIR"
    echo "  - math_intermediate.json: MATH benchmark details"
    echo "  - code_intermediate.json: CODE benchmark details"
    echo "  - benchmark_report_*.json: Full GPT-4 comparison report"
} > "$SUMMARY_FILE"

echo ""
echo "📄 Summary saved to: $SUMMARY_FILE"
echo ""
echo "✅ Validation complete!"
echo ""
