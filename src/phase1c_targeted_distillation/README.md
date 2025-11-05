# Phase 1C: Targeted Distillation (Combined 1C/1D)

**Purpose:** Generate improved examples with Claude/GPT-4o-mini, create bidirectional pairs, and train with smart early stopping.

## Scripts

- **generate_claude_examples.py** - Generate improved examples using Claude Sonnet 4.5 or GPT-4o-mini
- **create_bidirectional_pairs.py** - Create forward + reverse training pairs
- **train_phase1c_combined_smart.py** - Smart training with convergence-based early stopping
- **run_phase1c_combined_workflow.sh** - Automated end-to-end workflow

## Workflow

1. Generate 4,942 improved examples (GPT-4o-mini $25 or Claude $150)
2. Create bidirectional pairs: 2,389 self-critique → 4,778 pairs
3. Create bidirectional pairs: 4,942 Claude → 9,884 pairs
4. Combine: 14,662 total training examples
5. Smart train: Early stopping with convergence detection (5-7h)
6. Merge LoRA and validate

## Outputs

- **Location:** `data/phase1c/`
- **Files:**
  - `improved_examples.jsonl` - 4,942 Claude/GPT examples
  - `self_critique_bidirectional.jsonl` - 4,778 pairs
  - `claude_bidirectional.jsonl` - 9,884 pairs
  - `combined_training_bidirectional.jsonl` - 14,662 total
- **Model:** `data/checkpoints/phase1c_combined/`

## Performance

- **Expected:** 63.34% → 88-92% pass rate
- **Training:** 5-7h with smart early stopping (saves 4-6h vs fixed epochs)
- **Cost:** $40-45 (GPT-4o-mini) or $165-170 (Claude)

## Documentation

See `docs/PHASE1CD_QUICKSTART.md` for complete execution guide.

## Status

⏳ **READY TO EXECUTE** - All scripts created, tested, and documented
