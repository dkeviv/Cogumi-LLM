# ⚠️ ARCHIVED - DO NOT USE

**Status**: DEPRECATED  
**Archived Date**: October 2025  
**Reason**: Old deduplication implementations superseded by parallel version

This folder contains deprecated and superseded utility scripts kept for historical reference.

## Archived Scripts

### `deduplication.py` - DEPRECATED
- **Status:** ❌ DEPRECATED (Do NOT use)
- **Archived Date:** October 17, 2025
- **Replaced By:** `../deduplication_parallel.py`
- **Reason:** MD5 hashing too slow (6-8 hours for 674K samples)
- **Performance:** 
  - This (MD5): 6-8 hours for 674K samples
  - Replacement: 9 minutes for 674K samples
- **Migration:** Use `ParallelDataDeduplicator` instead of `DataDeduplicator`

### `deduplication_optimized.py` - SUPERSEDED
- **Status:** ⚠️ SUPERSEDED (Still functional but not recommended)
- **Archived Date:** October 17, 2025
- **Replaced By:** `../deduplication_parallel.py`
- **Reason:** Parallel version 6x faster with multiprocessing
- **Performance:**
  - This (sequential xxhash): 56 minutes for 674K samples
  - Replacement (parallel xxhash): 9 minutes for 674K samples
- **When to Use:** Single-core environments or small datasets (<10K samples)
- **When to Use Replacement:** Production runs, large datasets (>10K samples), multi-core systems

## Current Active Scripts

See parent directory (`../`) for current, maintained scripts:
- **`deduplication_parallel.py`** - CURRENT: Parallel MinHash LSH deduplication (RECOMMENDED)
  - Performance: 9 minutes for 674K samples
  - Features: xxhash, multiprocessing, progress bars
  - Use for: All production deduplication tasks

## Archive Policy

Scripts are archived when:
1. **Deprecated:** Fundamentally flawed or obsolete (marked ❌)
2. **Superseded:** Replaced by significantly better implementation (marked ⚠️)

Archived scripts are kept for:
- Historical reference
- Understanding evolution of implementation
- Comparison benchmarking
- Edge cases where old version might still be needed

**Last Updated:** October 17, 2025
