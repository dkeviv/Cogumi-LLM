# Archive: Phase 1 Intermediate Data

**Date Archived:** November 14, 2025

## Files in This Archive

### Checkpoint Files (Cumulative Bug)
- `checkpoint_coding.jsonl` - 10K lines (Coding only)
- `checkpoint_math.jsonl` - 20K lines (Coding + Math)
- `checkpoint_tool_use.jsonl` - 40K lines (Cumulative)
- `checkpoint_reasoning.jsonl` - 50K lines (Cumulative)
- `checkpoint_reading.jsonl` - 55K lines (Cumulative)
- `checkpoint_summarization.jsonl` - 60K lines (Cumulative)
- `checkpoint_common_sense.jsonl` - 65K lines (Cumulative)
- `checkpoint_instruction.jsonl` - 304K lines (Full accumulation)

**Issue:** Each checkpoint contained ALL previous domains, causing massive duplication when loaded together.

### `questions_60k.jsonl` (304,993 lines)
**Stats:**
- Total: 304,993 questions
- Unique: 44,700 (only 14.7% unique!)
- Duplicates: 260,293 (85.3% duplication rate)

**Domain Distribution (vs Target):**
- Coding: 120,000 (target: 10,000) - 12x over
- Math: 60,000 (target: 10,000) - 6x over
- Tool Use: 50,000 (target: 10,000) - 5x over
- Reasoning: 40,000 (target: 10,000) - 4x over
- Reading: 15,000 (target: 5,000) - 3x over
- Summarization: 10,000 (target: 5,000) - 2x over
- Common Sense: 5,000 (target: 5,000) - ✓ correct
- Instruction: 4,993 (target: 5,000) - ✓ correct

**Root Cause:** Checkpoint loading logic accumulated all questions from all cumulative checkpoints.

## Current Active Data

**`questions_60k_clean.jsonl`** (44,700 unique questions)
- Deduplicated from questions_60k.jsonl
- Used as baseline for token-balanced generation

## Next Generation

**`questions_60k_token_balanced.jsonl`** (60,000 questions - to be generated)
- 59,210 easy questions (98.7%)
- 789 hard questions (1.3%)
- Achieves 60% easy tokens / 40% hard tokens
- Prevents reasoning bleed and verbosity on easy tasks
