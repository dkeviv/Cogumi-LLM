# Phase 1A: Base Model Training

**Purpose:** Train Llama-3.1-8B-Instruct base model with QLoRA/full precision on 600K curated examples.

## Scripts

- **train_phase1a_optimized_h100.py** - Main training script with H100 optimizations
- **merge_lora_adapter.py** - Merge LoRA weights back to base model
- **validate_merged_model.py** - Validate merged model quality
- **pretokenize_dataset.py** - Pre-tokenize dataset for faster training

## Outputs

- **Location:** `Phase1A_2_0/models/phase1a_merged_10gb/`
- **Size:** 15GB (10GB base + 5GB LoRA)
- **Performance:** 75-82% GPT-4 baseline

## Documentation

See `docs/phase1a_2_0_full_precision/` for detailed guides.

## Status

✅ **COMPLETE** - Base model trained and merged (15GB)
