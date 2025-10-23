# ⚠️ ARCHIVED - DO NOT USE

**Status**: DEPRECATED  
**Archived Date**: October 2025  
**Reason**: Old implementation superseded by current `src/` directory

## Contents

This folder contains the original implementation from early development phases:

- `data_collection/`: Original dataset downloaders and curators
- `evaluation/`: Basic benchmark scripts (replaced by `automated_gpt4_benchmark.py`)
- `phase0_chat/`: Early chat interface experiments
- `phase1_distillation/`: Initial distillation approach (replaced by current pipeline)
- `phase2_compression/`: Prototype compression scripts
- `phase3a_general_modifiers/`: Early modifier experiments
- `phase3b_coding_modifiers/`: Early coding modifier experiments
- `phase4_runtime/`: Initial runtime experiments

## Why Archived?

1. **Architecture Changed**: Moved from Axolotl to HuggingFace + Unsloth
2. **Better Implementation**: Current `src/` has optimized, production-ready code
3. **Dependencies Outdated**: Uses old library versions incompatible with H100
4. **Performance Issues**: Not optimized for modern GPUs

## Current Implementation

Use these instead:
- Dataset: `src/phase0_dataset/` (complete and tested)
- Training: `notebooks/H100_Training_Clean.ipynb` (production-ready)
- Benchmarking: `scripts/automated_gpt4_benchmark.py` (comprehensive)
- Compression: `src/phase2_compression/` (when implemented)

## Kept For

- Historical reference
- Learning from early design decisions
- Comparing old vs new approaches

**DO NOT import or use code from this directory in production.**
