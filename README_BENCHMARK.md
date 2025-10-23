# Automated GPT-4 Benchmarking System

Comprehensive evaluation pipeline comparing your trained model against GPT-4 baseline.

## Quick Start

### After Phase 1A Training Completes:

1. **On Vast.ai** (recommended):
   ```bash
   # Open Phase1B_Benchmark.ipynb in JupyterLab
   # Run all cells with your OpenAI API key
   ```

2. **Via Command Line**:
   ```bash
   bash scripts/run_phase1b_benchmark.sh YOUR_OPENAI_KEY
   ```

3. **Manual Python**:
   ```bash
   python scripts/automated_gpt4_benchmark.py \
       --model_path /data/Cogumi-LLM/checkpoints/final \
       --openai_key YOUR_OPENAI_KEY \
       --output_dir ./benchmark_results \
       --num_samples 50
   ```

## What It Does

### 1. Tests 6 Categories

- **Math**: GSM8K mathematical reasoning
- **Code**: HumanEval coding tasks
- **Reasoning**: ARC-Challenge logical problems
- **Knowledge**: MMLU general knowledge
- **Instruction**: Alpaca instruction following
- **Creativity**: Open-ended creative tasks

### 2. Comparison Method

For each test:
1. Your model generates a response
2. GPT-4 generates a response
3. GPT-4 judges which is better (blind evaluation)
4. Tracks wins/losses/ties

### 3. Outputs

**Benchmark Report** (`benchmark_report_TIMESTAMP.json`):
```json
{
  "overall": {
    "score_vs_gpt4": 89.5,
    "performance_rating": "✅ EXCELLENT (90-95% of GPT-4)",
    "wins": 179,
    "losses": 21,
    "ties": 50
  },
  "by_category": {
    "code": {"score": 95.2, "wins": 48, ...},
    "math": {"score": 87.3, "wins": 35, ...},
    ...
  }
}
```

**Failure Examples** (`failure_examples.json`):
- All examples where GPT-4 won
- Used for Phase 1C targeted distillation

## Cost Estimates

### Quick Benchmark (50 samples/category)
- **Time**: ~30-60 minutes
- **Cost**: ~$5-10
- **Use**: Initial evaluation after Phase 1A

### Full Benchmark (100 samples/category)
- **Time**: ~1-2 hours
- **Cost**: ~$15-25
- **Use**: Final validation before Phase 2

### Deep Benchmark (500 samples/category)
- **Time**: ~4-6 hours
- **Cost**: ~$75-125
- **Use**: Production quality gate

## Interpreting Results

### Score Ratings

| Score | Rating | Next Step |
|-------|--------|-----------|
| ≥100% | 🏆 EXCEEDS GPT-4 | Proceed to Phase 2 |
| 95-100% | ⭐ MATCHES GPT-4 | Proceed to Phase 2 |
| 90-95% | ✅ EXCELLENT | Proceed to Phase 2 |
| 85-90% | 👍 GOOD | Optional Phase 1C |
| 80-85% | 📊 ACCEPTABLE | Recommended Phase 1C |
| <80% | ⚠️ NEEDS WORK | Required Phase 1C |

### Decision Tree

```
Phase 1A Complete
       ↓
Run Phase 1B Benchmark
       ↓
   Score ≥ 90%? ──YES──→ Phase 2: Compression
       ↓ NO
   Score ≥ 85%? ──YES──→ Optional: Quick Phase 1C (10K examples)
       ↓ NO
   Score < 85% ────────→ Required: Full Phase 1C (40K examples)
```

## Phase 1C: Targeted Distillation

If benchmark shows weak areas (< 85%), run targeted GPT-5 distillation:

```python
# Identify weak categories from benchmark
weak_categories = ["math", "reasoning"]  # Example

# Generate targeted examples
python scripts/generate_gpt5_distillation.py \
    --failure_examples ./benchmark_results/failure_examples.json \
    --categories math reasoning \
    --num_examples 20000 \
    --output_dir ./data/phase1c
```

## Example Output

```
🎯 BENCHMARK SUMMARY
============================================================

📊 Overall Performance vs GPT-4:
   Score: 89.5%
   Rating: ✅ EXCELLENT (90-95% of GPT-4)
   Win Rate: 71.6%

📈 Breakdown by Category:

   CODE:
      Score: 95.2%
      W/L/T: 48/2/0

   INSTRUCTION:
      Score: 92.1%
      W/L/T: 43/5/2

   KNOWLEDGE:
      Score: 88.7%
      W/L/T: 39/8/3

   REASONING:
      Score: 87.3%
      W/L/T: 35/10/5

   MATH:
      Score: 82.1%
      W/L/T: 28/16/6

   CREATIVITY:
      Score: 79.5%
      W/L/T: 25/20/5

⏱️ Total Time: 45.3 minutes
============================================================

⚠️ Weak areas (need GPT-5 distillation):
   - math: 82.1%
   - creativity: 79.5%
```

## Files

- `automated_gpt4_benchmark.py` - Main benchmark script
- `run_phase1b_benchmark.sh` - Quick runner script
- `Phase1B_Benchmark.ipynb` - Interactive notebook
- `README_BENCHMARK.md` - This file

## Next Steps

After benchmarking:

1. **Review results** in `benchmark_results/`
2. **Identify weak areas** (< 85% score)
3. **Decide**: Proceed to Phase 2 or run Phase 1C distillation
4. **If Phase 1C needed**: Generate targeted GPT-5 examples for weak areas

---

**Last Updated**: October 2025
**Status**: Ready for Phase 1B 🚀
