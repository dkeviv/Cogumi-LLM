# Phase 1 Outdated Scripts Archive

**Archive Date:** November 14, 2025  
**Reason:** Scripts superseded by better implementations or no longer needed

---

## Archived Scripts

### 1. `download_benchmarks.py` ❌ DEPRECATED
- **Replaced by:** `scripts/download_all_benchmarks.py`
- **Reason:** Incomplete benchmark coverage, missing fallback strategies
- **Issues:**
  - Failed to download several benchmarks
  - No error handling for network issues
  - No progress indicators
  - Unreliable

### 2. `download_benchmarks_v2.py` ❌ DEPRECATED
- **Replaced by:** `scripts/download_all_benchmarks.py`
- **Reason:** Intermediate version, superseded by final comprehensive solution
- **Issues:**
  - Alternative approach but still incomplete
  - Less robust than final version
  - Missing some fallback strategies

### 3. `phase1_final_consolidation.py` ❌ DEPRECATED
- **Replaced by:** `scripts/phase1_balance_final_60k.py`
- **Reason:** Superseded by better balancing implementation
- **Issues:**
  - Manual consolidation approach
  - Less sophisticated balancing logic
  - Doesn't handle public dataset augmentation
  - Token balance not verified

**New Script Benefits:**
- Automatic balancing to 7,500 per domain
- Public dataset integration
- Token balance verification (60/40)
- Batch mixing validation

### 4. `test_diverse_prompts.py` ⚠️ ARCHIVED (Test Script)
- **Replaced by:** N/A (no replacement needed)
- **Reason:** Development test script, purpose fulfilled
- **Status:** Was used during development to test prompt diversity
- **Note:** Can be deleted or kept for reference

### 5. `check_question_quality.py` 📦 ARCHIVED (Utility)
- **Status:** Utility script for sampling questions during generation
- **Reason for Archive:** Generation phase complete
- **Future Use:** May be useful for future question generation quality checks
- **Note:** Can be reactivated if needed

---

## Current Active Scripts (In `scripts/`)

✅ **Active Phase 1 Scripts:**
1. `phase1_augment_with_public_datasets.py` - Augments with public datasets
2. `phase1_balance_final_60k.py` - Balances dataset to final 60K
3. `phase1_test_answer_generation.py` - Generates answers with teacher models
4. `download_all_benchmarks.py` - Downloads benchmarks with fallbacks

---

## Migration Path

If you accidentally try to use an archived script:

1. **For benchmarks:** Use `download_all_benchmarks.py`
   ```bash
   python scripts/download_all_benchmarks.py
   ```

2. **For dataset balancing:** Use `phase1_balance_final_60k.py`
   ```bash
   python scripts/phase1_balance_final_60k.py
   ```

3. **For answer generation:** Use `phase1_test_answer_generation.py`
   ```bash
   python scripts/phase1_test_answer_generation.py
   ```

---

## Archive Structure

```
scripts/
├── archive/
│   ├── phase1_deprecated/     # Very old deprecated scripts
│   ├── phase1_generation/     # Old generation scripts
│   └── phase1_outdated/       # Recently archived (this directory)
│       ├── README.md
│       ├── download_benchmarks.py
│       ├── download_benchmarks_v2.py
│       ├── phase1_final_consolidation.py
│       ├── test_diverse_prompts.py
│       └── check_question_quality.py
└── [active scripts]
```

---

## Notes

- All archived scripts have deprecation headers added
- Scripts preserved for historical reference
- Can be safely deleted if disk space needed
- Refer to technical_specification.md for current implementation details
