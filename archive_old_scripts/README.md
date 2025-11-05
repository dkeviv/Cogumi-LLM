# Archived Scripts from Phase 1A 1.0 (QLoRA Era)

## ⚠️ DEPRECATED - DO NOT USE

These scripts are from the **Phase 1A 1.0 (QLoRA)** era which experienced catastrophic merge corruption. They have been superseded by Phase 1A 2.0 (full precision training).

## Scripts

### train_phase1a_fullprecision.py
- **Purpose**: Attempted fix for QLoRA merge issue by training on full precision base
- **Issue**: Still part of Phase 1A 1.0 era, addressed 4-bit quantization problems
- **Status**: DEPRECATED - Use Phase1A_2_0 training pipeline instead
- **Date**: October 28, 2025

### train_phase1b_benchmark.py  
- **Purpose**: Targeted training on benchmark failures to fix consistency issues
- **Issue**: Designed for Phase 1A 1.0's merge corruption (70% ties in MATH, 28% in CODE)
- **Status**: DEPRECATED - Phase 1B 2.0 uses Haiku judging + self-critique approach
- **Date**: October 27, 2025

### convert_docs.py
- **Purpose**: Utility to convert Word documents to Markdown
- **Status**: Utility script, kept for reference
- **Date**: October 19, 2025

## Why Deprecated?

**Phase 1A 1.0 Problems:**
- Used 4-bit quantized base during training (Unsloth optimization)
- Merge corruption caused catastrophic forgetting
- Math accuracy dropped from 70% ties to 28% ties after merge
- Required complete restart with full precision approach

**Phase 1A 2.0 Solution:**
- Full precision bfloat16 training (meta-llama/Meta-Llama-3.1-8B-Instruct)
- Clean merge with no corruption
- 15GB model output
- Located at: `Phase1A_2_0/`

**Phase 1B 2.0 New Approach:**
- Haiku judging for authoritative failure identification (7,331 failures)
- Self-critique pipeline for local model improvement
- No reliance on benchmark-based training
- Located at: `Phase 1B_2_0/`

## Replacement

Use current pipeline:
- **Training**: `Phase1A_2_0/` (full precision approach)
- **Failure Analysis**: `Phase 1B_2_0/` (Haiku judging + self-critique)
- **Scripts**: `Phase 1B_2_0/step*.py` (self-critique pipeline)

---

**Archived**: November 3, 2025
**Reason**: Phase 1A 1.0 complete failure, Phase 1A 2.0 successful completion
