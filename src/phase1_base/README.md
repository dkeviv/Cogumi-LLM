# Phase 1: Base Model Training

**Duration:** 4 weeks  
**Cost:** $505  
**Status:** ⏳ Starting next

## Overview
Train LLAMA-3.2-8B base model from scratch using the 600K curated dataset, with vocabulary optimization and GPT-5 targeted enhancement.

## Sub-Phases

### Phase 1A: Vocabulary & Base Training (2.5 weeks, $220)
1. Vocabulary analysis and trimming (128K → 25K tokens)
2. QLoRA training on 600K examples
3. Validation on MMLU, HumanEval, GSM8K

### Phase 1B: Failure Analysis (1 week, $5)
1. Comprehensive testing on 50K diverse examples
2. Identify 12-14K failures
3. Cluster failures using Sentence-BERT + KMeans
4. Auto-label clusters with GPT-4-mini

### Phase 1C: GPT-5 Enhancement (1 week, $280)
1. Generate 40K targeted examples with GPT-5
2. Quality filter with GPT-4-mini (>8/10)
3. Distillation training (90% GPT-5 + 10% original)
4. Final validation → 88-100% GPT-4 performance

## Scripts

### Vocabulary Optimization
- `vocab_analysis.py` - Analyze token frequency in 10K samples
- `vocab_trimming.py` - Trim tokenizer 128K → 25K (saves 3.4GB)
- `validate_vocab.py` - Ensure <3% perplexity increase

### Base Training
- `merge_lora.py` - Merge LoRA weights into base model
- `validate_base.py` - Run MMLU, HumanEval, GSM8K benchmarks

### Failure Analysis
- `prepare_test_sets.py` - Create 50K diverse test set
- `test_base.py` - Test model and identify failures
- `embed_failures.py` - Generate Sentence-BERT embeddings
- `cluster_failures.py` - KMeans clustering (k=10)
- `label_clusters.py` - Auto-label with GPT-4-mini

### GPT-5 Data Generation
- `generate_prompts.py` - Create prompts for each failure pattern
- `generate_gpt5_data.py` - Call GPT-5 API (40K examples)
- `score_examples.py` - Quality scoring with GPT-4-mini
- `recovery_finetune.py` - Train on GPT-5 data

## Axolotl Configuration
Training managed by Axolotl framework:
- Config: `configs/base_training.yaml`
- Distillation config: `configs/distillation_training.yaml`

## Expected Outcomes
- **10GB merged base model** (75-82% GPT-4 after Phase 1A)
- **8-12 labeled failure patterns** (Phase 1B)
- **10GB enhanced model** (88-100% GPT-4 after Phase 1C)

## Next Steps
After Phase 1 completion, proceed to **Phase 2: Compression** to reduce 10GB → 520MB.

See `docs/EXECUTION_PLAN.md` for detailed week-by-week plan.
