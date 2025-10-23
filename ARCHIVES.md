# Archive Folders - Quick Reference

This document lists all archived/deprecated folders in the Cogumi-LLM repository.

## ⚠️ DO NOT USE THESE FOLDERS

All folders listed below contain **outdated, deprecated, or superseded** code/documentation.

## Archive Locations

### 1. `archive_old_src/` - OLD SOURCE CODE
**Path**: `/archive_old_src/`  
**Status**: ❌ DEPRECATED  
**Reason**: Old Axolotl-based implementation  
**Use Instead**: `src/` directory  
**Details**: See `archive_old_src/README_ARCHIVE.md`

### 2. `configs/archive/` - OLD CONFIGURATIONS
**Path**: `/configs/archive/`  
**Status**: ❌ DEPRECATED  
**Reason**: Axolotl YAML configs (framework no longer used)  
**Use Instead**: `notebooks/H100_Training_Clean.ipynb` or `docs/technical_specification.md`  
**Details**: See `configs/archive/README_ARCHIVE.md`

### 3. `docs/archive/` - OLD DOCUMENTATION
**Path**: `/docs/archive/`  
**Status**: ❌ DEPRECATED  
**Reason**: Documentation for deprecated architecture  
**Use Instead**: Current docs in `docs/` root (especially `technical_specification.md`)  
**Details**: See `docs/archive/README_ARCHIVE.md`

### 4. `docs/archive2/` - OLD DOCUMENTATION (Second Archive)
**Path**: `/docs/archive2/`  
**Status**: ❌ DEPRECATED  
**Reason**: Additional outdated documentation  
**Use Instead**: Current docs in `docs/` root  
**Details**: See `docs/archive2/README_ARCHIVE.md`

### 5. `scripts/archive/` - OLD SCRIPTS
**Path**: `/scripts/archive/`  
**Status**: ❌ DEPRECATED  
**Reason**: Old scripts for deprecated framework  
**Use Instead**: Scripts in `scripts/` root directory  
**Details**: See `scripts/archive/README_ARCHIVE.md`

### 6. `src/utils/archive/` - OLD DEDUPLICATION SCRIPTS
**Path**: `/src/utils/archive/`  
**Status**: ❌ DEPRECATED  
**Reason**: Slow deduplication implementations (MD5, sequential)  
**Use Instead**: `src/utils/deduplication_parallel.py`  
**Details**: See `src/utils/archive/README.md`

## Why Keep Archives?

Archives are kept for:
- Historical reference
- Understanding project evolution
- Learning from past decisions
- Comparing old vs new approaches
- Academic/research purposes

## How to Identify Archives

Look for:
1. Folder names containing `archive`
2. `README_ARCHIVE.md` files (warning signs)
3. References to "Axolotl" (old framework)
4. Old dependency versions (PyTorch <2.8, transformers <4.57)

## What to Use Instead

### For Training
- **Notebook**: `notebooks/H100_Training_Clean.ipynb` (production-ready)
- **Script**: Generated `train.py` from notebook Cell 14
- **Dependencies**: See golden set in notebook Cell 1

### For Documentation
- **Technical Spec**: `docs/technical_specification.md` (updated Oct 2025)
- **Quick Start**: `docs/PHASE1A_QUICKSTART.md`
- **Status**: `docs/CURRENT_STATUS.md`
- **Checklist**: `docs/IMPLEMENTATION_CHECKLIST.md`

### For Scripts
- **Benchmarking**: `scripts/automated_gpt4_benchmark.py`
- **Downloads**: `scripts/download_*.py`
- **Quick Benchmark**: `scripts/run_phase1b_benchmark.sh`

### For Source Code
- **Phase 0**: `src/phase0_dataset/` (complete)
- **Utilities**: `src/utils/` (current implementations)
- **Future phases**: `src/phase1_base/`, `src/phase2_compression/`, etc.

## Migration Guide

If you're using archived code:

1. **Stop using it immediately**
2. **Read the archive's README_ARCHIVE.md** for specific replacement
3. **Update to current implementation** (see "What to Use Instead" above)
4. **Test thoroughly** (implementations differ significantly)
5. **Update dependencies** (see `notebooks/H100_Training_Clean.ipynb` Cell 1)

## Questions?

If you're unsure whether to use a file:
1. Check if it's in an `archive` folder → Don't use it
2. Check `git log` for last update → If >3 months old, verify with current docs
3. Check documentation date → Anything before Oct 2025 may be outdated
4. Ask in GitHub Discussions → Link to specific file

---

**Last Updated**: October 22, 2025  
**Maintenance**: Review quarterly, add new archives as needed
