# Phase 1: Current Clean State

**Last Updated:** November 14, 2025

## 🎯 Goal
Generate 60K token-balanced questions (60% easy tokens / 40% hard tokens) for MAML training.

## ✅ Active Files

### Scripts (2 files)
1. **`phase1_calculate_token_balance.py`**
   - Shows token distribution analysis
   - Calculates required easy/hard split
   - Verifies: Need 98.7% easy / 1.3% hard by count to achieve 60/40 token split

2. **`phase1_generate_token_balanced.py`** ⭐ PRIMARY SCRIPT
   - Generates token-balanced 60K dataset
   - Parallel async (20 concurrent requests)
   - Incremental checkpoint saving per domain+difficulty
   - Tag normalization (force lowercase)
   - Random shuffle to prevent domain clustering

3. **`test_token_balance_logic.py`**
   - Pre-flight check before generation
   - Validates logic and calculates generation plan

### Data (1 file)
- **`questions_60k_clean.jsonl`** (12 MB, 44,700 questions)
  - Deduplicated baseline
  - Used as starting point for token-balanced generation

## 📦 Archived Files

### Scripts (6 archived)
- `phase1_generate_questions.py` (v1 - slow, LaTeX errors)
- `phase1_generate_questions_v2.py` (v2 - still slow, improved parsing)
- `phase1_generate_questions_v3_parallel.py` (v3 - fast but wrong distribution)
- `phase1_generate_missing_questions.py` (interim helper)
- `phase1_deduplicate_questions.py` (completed, integrated into main)
- `phase1_generate_questions_parallel.py` (final v3, superseded)

### Data (9 archived)
- 8 cumulative checkpoint files (caused duplication bug)
- 1 duplicated output file (304K with 85% duplicates)

See `scripts/archive/phase1_deprecated/README.md` for details.

## 🚀 Next Steps

1. **Run pre-flight check:**
   ```bash
   python scripts/test_token_balance_logic.py
   ```

2. **Generate token-balanced 60K:**
   ```bash
   python scripts/phase1_generate_token_balanced.py
   ```
   - Expected: ~44K new easy questions
   - Duration: ~110 minutes (44K / 400 questions per minute)
   - Output: `data/phase1/questions_60k_token_balanced.jsonl`

3. **Validate with GPT-4-mini** ($9)
4. **Generate EASY answers with GPT-4o-mini** ($2.52)
5. **Generate HARD answers with Claude Sonnet 4** ($414)
6. **Train with proper loss masking**

## 📊 Target Distribution

| Domain | Total | Easy (98.7%) | Hard (1.3%) |
|--------|-------|--------------|-------------|
| Coding | 10,000 | 9,870 | 130 |
| Math | 10,000 | 9,870 | 130 |
| Tool Use | 10,000 | 9,870 | 130 |
| Reasoning | 10,000 | 9,870 | 130 |
| Reading | 5,000 | 4,935 | 65 |
| Summarization | 5,000 | 4,935 | 65 |
| Common Sense | 5,000 | 4,935 | 65 |
| Instruction | 5,000 | 4,935 | 65 |
| **TOTAL** | **60,000** | **59,210** | **789** |

**Token Distribution:**
- Easy: 59,210 × 15 tokens = 888,150 tokens (60.0%)
- Hard: 789 × 750 tokens = 591,750 tokens (40.0%)
- **Total: 1,479,900 tokens** ✅

## 🎓 Key Learnings (Applied)

1. ✅ Token-weighted sampling (not count-based)
2. ✅ Parallel async for speed (400x faster)
3. ✅ Independent checkpoints per domain+difficulty
4. ✅ Tag normalization (force lowercase consistency)
5. ✅ Random shuffle to prevent domain clustering
6. ✅ Clean workspace (archive old iterations)
