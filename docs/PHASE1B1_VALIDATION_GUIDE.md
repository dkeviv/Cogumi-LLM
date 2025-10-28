# Phase 1B.1 Validation Guide

## Quick Start

### On Vast.ai (Recommended)

```bash
# Set your OpenAI API key
export OPENAI_API_KEY='your-key-here'

# Run validation
cd /workspace/data/Cogumi-LLM
bash scripts/validate_phase1b1.sh
```

### What It Does

1. **Verifies model exists**: Checks `checkpoints/phase1b_from_benchmark/`
2. **Loads Phase 1A baseline**: From `checkpoints/benchmark_results_full/`
3. **Runs GPT-4 comparison benchmarks**:
   - 50 MATH problems (GSM8K)
   - 50 CODE problems (HumanEval)
4. **For each problem**:
   - Generates Phase 1B.1 model response
   - Generates GPT-4 response
   - Uses GPT-4 as judge to pick winner
5. **Compares with Phase 1A**: Shows improvements
6. **Provides decision criteria**: Success/iterate guidance

## Expected Results

### Phase 1A Baseline (from previous run)
- **MATH**: ~6% wins, ~70% ties, ~24% losses
- **CODE**: ~48% wins, ~20% ties, ~32% losses
- **Issue**: High tie rate = inconsistent responses

### Phase 1B.1 Targets (after training on 73 failures)
- **MATH**: 20-30% wins (3x-5x improvement), <50% ties
- **CODE**: 55-65% wins (+15-35% improvement), <15% ties
- **Goal**: Improved consistency, fewer ties

## Success Criteria

✅ **PASS** if:
- MATH wins: 6% → 20-30% (3x-5x improvement)
- CODE wins: 48% → 55-65% (+15-35% improvement)
- MATH ties: 70% → <50% (better consistency)
- CODE ties: 20% → <15% (better consistency)

➡️ **Next Step**: Proceed to Phase 1B.2
- Extract ~2,000 failures from GSM8K train set
- Train on 73 + 2,000 = 2,073 examples
- Expected: 11-16 hours, $22-30

🔄 **ITERATE** if:
- Improvements below 3x for MATH
- CODE wins <55%
- Ties still >50% for MATH

**Iteration options:**
- Increase epochs: 2 → 3-4
- Adjust learning rate: 5e-6 → 3e-6 or 7e-6
- Add more diverse failure examples
- Extract failures from other benchmarks

## Time & Cost

- **Duration**: 30-40 minutes
- **Cost**: ~$1-1.50 (GPT-4 API calls)
- **GPU**: H100 idle during benchmarking (only inference, no training)

## Output Files

All saved to `checkpoints/benchmark_results_phase1b1/`:

1. **math_intermediate.json**: Detailed MATH results
   - Each problem: prompt, model response, GPT-4 response, judgment
   - Summary: wins, losses, ties

2. **code_intermediate.json**: Detailed CODE results
   - Same structure as MATH

3. **benchmark_report_*.json**: Overall summary
   - Score vs GPT-4 for each category
   - Performance rating
   - Timestamp

4. **validation_summary.txt**: Human-readable summary
   - Training config
   - Results comparison
   - Decision guidance

## Interpreting Results

### Example Good Result
```
MATH:
  Phase 1A: 3/50 wins (6%), 35 ties (70%)
  Phase 1B.1: 12/50 wins (24%), 22 ties (44%)
  → 4x improvement ✅, ties reduced ✅

CODE:
  Phase 1A: 24/50 wins (48%), 10 ties (20%)
  Phase 1B.1: 30/50 wins (60%), 6 ties (12%)
  → +25% improvement ✅, ties reduced ✅

Decision: PROCEED to Phase 1B.2 ✅
```

### Example Needs Iteration
```
MATH:
  Phase 1A: 3/50 wins (6%), 35 ties (70%)
  Phase 1B.1: 6/50 wins (12%), 30 ties (60%)
  → 2x improvement ⚠️ (target: 3x-5x)

CODE:
  Phase 1A: 24/50 wins (48%), 10 ties (20%)
  Phase 1B.1: 26/50 wins (52%), 8 ties (16%)
  → +8% improvement ⚠️ (target: +15-35%)

Decision: ITERATE Phase 1B.1 with adjusted params 🔄
```

## Troubleshooting

### "OPENAI_API_KEY not set"
```bash
export OPENAI_API_KEY='sk-...'
# Or pass inline:
OPENAI_API_KEY='sk-...' bash scripts/validate_phase1b1.sh
```

### "Phase 1B.1 model not found"
- Check training completed: `ls -lh checkpoints/phase1b_from_benchmark/`
- Should have: adapter_config.json, adapter_model.safetensors, etc.

### "Phase 1A baseline not found"
- Warning only - validation will still run
- Just won't show comparison
- To get baseline: Run Phase 1A benchmarks first

### Script fails during benchmarking
- Check GPU memory: `nvidia-smi`
- Check OpenAI API key valid
- Check internet connection
- Re-run - intermediate results saved every 10 examples

## After Validation

### If Successful
1. Review results: `cat checkpoints/benchmark_results_phase1b1/validation_summary.txt`
2. Proceed to Phase 1B.2:
   ```bash
   # Extract failures from GSM8K train
   python scripts/extract_gsm8k_failures.py \
       --model_path checkpoints/final \
       --output_dir data/phase1b2 \
       --num_samples 7473
   
   # Train on expanded dataset
   bash scripts/run_phase1b2_training.sh
   ```

### If Needs Iteration
1. Review failure patterns in intermediate JSONs
2. Adjust training parameters in `train_phase1b_benchmark.py`:
   ```python
   num_train_epochs=3  # Increase from 2
   learning_rate=3e-6  # Or try 7e-6
   ```
3. Re-run training: `bash scripts/run_phase1b_benchmark_training.sh`
4. Re-validate

## Notes

- **Intermediate saves**: Results saved every 10 examples, can resume if interrupted
- **Deterministic**: Same problems tested as Phase 1A for fair comparison
- **Judge model**: GPT-4 as judge (not GPT-4-turbo), consistent with Phase 1A
- **Temperature**: Math/code at temp=0.0 for deterministic reasoning
- **Categories**: Only "math" and "code" tested (Phase 1B focus areas)

## Integration with Phase 1A

Phase 1B.1 is a **targeted improvement** on Phase 1A:
- Phase 1A: Trained on 600K general examples
- Phase 1B.1: Trained on 73 specific failures
- Goal: Improve consistency on hard problems without forgetting general knowledge
- Validation: Compare same 50+50 problems to measure improvement

Think of it as:
- Phase 1A = Foundation (broad knowledge)
- Phase 1B.1 = Reinforcement (targeted fixes)
- Phase 1B.2 = Expansion (scale up fixes)
