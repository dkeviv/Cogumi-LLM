# Archive: Deprecated Phase 1 Scripts

**Date Archived:** November 14, 2025

## Why Archived

These scripts were superseded by `phase1_generate_token_balanced.py` which implements proper token-weighted sampling (60% easy tokens / 40% hard tokens) as per best practices.

## Archived Scripts

### `phase1_generate_questions.py` (v1)
- **Issue:** Sequential generation, extremely slow (12+ hours)
- **Issue:** LaTeX parsing errors in 50% of responses
- **Issue:** Batch size too large (100 questions/call)
- **Replaced by:** v2 with better parsing

### `phase1_generate_questions_v2.py` (v2)
- **Issue:** Still sequential, slow (~10-12 hours)
- **Improvement:** Better JSON parsing with regex cleanup
- **Improvement:** Resume capability from checkpoints
- **Issue:** Checkpoint system caused massive duplication
- **Replaced by:** v3 parallel version

### `phase1_generate_questions_v3_parallel.py` (v3)
- **Achievement:** Async/parallel with 20 concurrent requests
- **Achievement:** Fast execution (3 minutes vs 12 hours!)
- **Issue:** Checkpoint accumulation bug (304K questions, 85% duplicates)
- **Issue:** Wrong distribution (40/60 easy/hard by COUNT instead of 60/40 by TOKENS)
- **Replaced by:** Token-balanced version

### `phase1_generate_missing_questions.py`
- **Purpose:** Generate missing 15,300 questions from v3 output
- **Issue:** Based on wrong distribution (didn't account for token weighting)
- **Replaced by:** Token-balanced generation with correct 98.7/1.3 split

### `phase1_deduplicate_questions.py`
- **Purpose:** Deduplicate 304K → 44.7K unique questions
- **Achievement:** Removed 260K duplicates (85% deduplication)
- **Status:** Completed successfully, functionality integrated into token-balanced script
- **Output:** `questions_60k_clean.jsonl` (used as baseline)

### `phase1_generate_questions_parallel.py` (Final working v3)
- **Achievement:** Completed generation in 3.1 minutes (vs 12+ hours)
- **Achievement:** 20 concurrent async API calls
- **Issue:** Generated with wrong 40/60 count split (should be 98.7/1.3 for token balance)
- **Status:** Superseded by token-balanced version with correct distribution

## Key Learnings

1. **Token-weighted sampling is critical:** Easy examples (15 tokens) vs Hard examples (750 tokens) require 98.7% easy / 1.3% hard by COUNT to achieve 60% easy / 40% hard by TOKENS.

2. **Parallel async is essential:** 400x speedup (3 minutes vs 12 hours) makes iteration feasible.

3. **Checkpoint design matters:** Cumulative checkpoints caused massive duplication. New version uses independent checkpoints per domain+difficulty.

4. **Tag normalization is critical:** API returns "EASY"/"HARD" but we need "easy"/"hard" - must force normalization.

## Current Script (Active)

**`phase1_generate_token_balanced.py`** - Token-weighted generation with:
- ✅ Proper 98.7% easy / 1.3% hard split for 60/40 token balance
- ✅ Parallel async generation (20 concurrent)
- ✅ Incremental checkpoint saving per domain+difficulty
- ✅ Tag normalization (force lowercase "easy"/"hard")
- ✅ Random shuffle to prevent domain clustering
- ✅ Per-domain progress tracking
