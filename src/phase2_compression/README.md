# Phase 2: Extreme Compression

**Duration:** 6 weeks  
**Cost:** $402  
**Status:** ⏳ Pending Phase 1 completion

## Overview
Aggressive compression pipeline to reduce 10GB Phase 1 model → 520MB while maintaining 89-91% GPT-4 performance.

## Compression Pipeline

### Phase 2A: Neural Magic Pruning (2 weeks, $180)
- **Goal:** 10GB → 3.5GB (65% sparsity)
- Grid search across sparsity levels (60%, 65%, 70%)
- Gradual pruning over 2K steps
- Post-pruning recovery fine-tuning

### Phase 2B: AWQ Quantization (1 week, $90)
- **Goal:** 3.5GB → 900MB (4-bit)
- Mixed-precision quantization with AWQ
- Group size: 128
- Calibration on 2K samples

### Phase 2C: GGUF Export (3 days, $0)
- **Goal:** 900MB → 600MB (Q5_K_M)
- Convert PyTorch → GGUF format
- Validate token-by-token agreement (>95%)

### Phase 2D: Zstd Compression (2 days, $0)
- **Goal:** 600MB → 500MB
- Dictionary training on 100MB sample
- Lossless compression (level 10)
- Validate SHA-256 checksums

### Phase 2E: Recovery Fine-Tuning (1 week, $70)
- **Goal:** 500MB → 520MB, recover +1-2% quality
- Identify hardest 12K examples (top 2% perplexity)
- Enhance with GPT-5
- Conservative LoRA fine-tuning (rank 64, lr 8e-7)

### Phase 2F: Confidence Calibration (3 days, $35)
- **Goal:** Enable accurate routing
- Generate 30K calibration queries
- Train temperature + Platt scaling
- Target: ECE <0.05, 97% routing accuracy

## Scripts

### Phase 2A: Pruning
- `prepare_calibration.py` - Sample 10K diverse examples
- `pruning_grid_search.py` - Test 60%, 65%, 70% sparsity
- `select_best_sparsity.py` - Choose optimal level
- `gradual_pruning.py` - Prune over 2K steps
- `recovery_finetune.py` - Post-pruning recovery

### Phase 2B: Quantization
- `prepare_awq_calibration.py` - 2K samples for AWQ
- `awq_quantize.py` - 4-bit mixed-precision quantization
- `validate_quantization.py` - Compare with original

### Phase 2C: GGUF Export
- `convert_to_gguf.py` - PyTorch → GGUF Q5_K_M
- `validate_gguf.py` - Token agreement >95%

### Phase 2D: Zstd Compression
- `train_zstd_dict.py` - Dictionary training (128KB)
- `compress_with_zstd.py` - Level 10 compression
- `validate_compression.py` - Lossless verification

### Phase 2E: Recovery
- `measure_perplexity.py` - Test on full 600K dataset
- `select_hardest.py` - Top 2% hardest examples
- `enhance_with_gpt5.py` - GPT-5 enhancement
- `recovery_lora.py` - Conservative LoRA training

### Phase 2F: Calibration
- `generate_calibration_data.py` - 30K queries with logits
- `label_quality.py` - GPT-4-mini quality scoring
- `train_calibration.py` - Temperature + Platt scaling
- `validate_calibration.py` - ECE and routing accuracy

## Expected Outcomes
- **520MB compressed base model**
- **89-91% GPT-4 performance** (vs 88-100% pre-compression)
- **Calibrated confidence scores** for routing
- **Quality loss:** ~7-9% from original 10GB model

## Compression Statistics
| Stage | Size | Format | Quality | Notes |
|-------|------|--------|---------|-------|
| Original | 10GB | FP16 | 100% | Phase 1C output |
| Pruned | 3.5GB | Sparse FP16 | 96-98% | 65% sparsity |
| Quantized | 900MB | 4-bit AWQ | 94-96% | Group size 128 |
| GGUF | 600MB | Q5_K_M | 93-95% | Optimized format |
| Compressed | 500MB | Zstd | 93-95% | Lossless |
| Recovered | 520MB | + LoRA | 91-93% | +1-2% recovery |

## Next Steps
After Phase 2, proceed to **Phase 3: Modifiers** to create specialized domain adapters.

See `docs/EXECUTION_PLAN.md` for detailed timeline.
