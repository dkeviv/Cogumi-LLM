#!/bin/bash
# Quick benchmark runner after Phase 1A training completes
# Usage: bash run_phase1b_benchmark.sh YOUR_OPENAI_KEY

set -e

MODEL_PATH="/data/Cogumi-LLM/checkpoints/final"
OPENAI_KEY="${1:-}"
OUTPUT_DIR="./benchmark_results_phase1a"

if [ -z "$OPENAI_KEY" ]; then
    echo "❌ Error: OpenAI API key required"
    echo "Usage: bash run_phase1b_benchmark.sh YOUR_OPENAI_KEY"
    exit 1
fi

echo "🚀 Starting Phase 1B: Failure Analysis Benchmark"
echo "================================================="
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo ""

# Run quick benchmark (50 samples per category for speed)
python scripts/automated_gpt4_benchmark.py \
    --model_path "$MODEL_PATH" \
    --openai_key "$OPENAI_KEY" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples 50 \
    --categories math code reasoning knowledge instruction

echo ""
echo "✅ Benchmark complete!"
echo "📊 Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. Review results in $OUTPUT_DIR/benchmark_report_*.json"
echo "2. Identify weak categories (score < 85%)"
echo "3. Run Phase 1C: Generate targeted GPT-5 examples for weak areas"
