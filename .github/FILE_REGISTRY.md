# Cogumi-LLM File Registry

**Purpose:** Comprehensive inventory of all code files and documentation organized by project phase.  
**Maintained:** Updated after every major implementation or file change.  
**Last Updated:** November 9, 2025

---

## 📋 Table of Contents

- [Phase 0: Dataset Creation](#phase-0-dataset-creation)
- [Phase 1A: Base Training](#phase-1a-base-training)
- [Phase 1B: Failure Analysis](#phase-1b-failure-analysis)
- [Phase 1C: Targeted Distillation](#phase-1c-targeted-distillation)
- [Phase 2: Compression](#phase-2-compression)
- [Phase 3: Domain Modifiers](#phase-3-domain-modifiers)
- [Phase 4: Router System](#phase-4-router-system)
- [Phase 5: Deployment](#phase-5-deployment)
- [Shared Utilities](#shared-utilities)
- [Documentation](#documentation)
- [Configuration](#configuration)

---

**Location:** `Phase0_Dataset/`  
**Output:** 600K curated examples at `data/phase1/public_500k_filtered.jsonl`

| File | Purpose | Methods/Algorithms | Achieves |
|------|---------|-------------------|----------|
| `scripts/dataset_downloader.py` | Download public datasets | HTTP requests, streaming download | Downloads Magpie, SlimOrca, WizardLM datasets |
| `scripts/dataset_curator.py` | Multi-teacher distillation | Llama-405B (40%), GPT-4o (35%), Qwen3-Coder (25%) mixing | Creates diverse 750K raw dataset |
| `scripts/quality_scorer.py` | Quality filtering | GPT-4-mini scoring (>7/10 threshold), batch API | Filters to 600K high-quality examples |
| `scripts/deduplication_parallel.py` | Deduplication | MinHash LSH (Jaccard 0.8), xxhash, multiprocessing | Removes 150K duplicates, 3000+ examples/sec |
| `scripts/verify_dataset.py` | Dataset validation | Statistical analysis, format checking | Verifies 600K final dataset quality |

**Documentation:**
- `README.md` - Phase 0 overview, pipeline, results
- `docs/PHASE0_COMPLETE_SUMMARY.md` - Completion summary with metrics

---

## Phase 1A: Base Training


|------|---------|-------------------|----------|
| `Phase1A_2_0/scripts/train_phase1a_optimized_h100.py` | Base QLoRA training | QLoRA (rank 64, 4-bit on FP16 base), bfloat16, gradient checkpointing | Trains on 600K examples, 89-91% GPT-4 performance |
| `Phase1A_2_0/scripts/merge_lora_adapter.py` | Merge LoRA to base | PEFT merge, model consolidation | Creates 15GB merged model |
| `Phase1A_2_0/scripts/pretokenize_dataset.py` | Dataset preprocessing | Tokenization, padding, chunking | Speeds up training data loading |
| `Phase1A_Archived_Quantized/README.md` | Archive documentation | N/A | Documents failed quantized approach for reference |
| `shared/utils/diagnose_gpu.py` | GPU diagnostics | CUDA checks, memory profiling | Validates H100 environment |

**Notebooks:**
- `notebooks/H100_Training_Clean.ipynb` - H100-optimized training (Unsloth + Flash Attention 2)
- `notebooks/Phase1A_Training_Colab.ipynb` - Google Colab version (A100 compatible)

**Documentation:**
- `docs/PHASE1A_QUICKSTART.md` - Quick start guide for base training
- `docs/H100_UNSLOTH_MIGRATION.md` - H100 optimization migration guide
- `docs/H100_QUICK_REFERENCE.md` - H100 setup and commands reference

**Configuration:**
- `configs/base_training.yaml` - Training hyperparameters (lr 2e-4, batch 8, epochs 3)
## Phase 1B: Failure Analysis

**Location:** `Phase1B_Failure_Analysis/`  
**Output:** 

|------|---------|-------------------|----------|
| **Test Dataset Creation** |||
| `scripts/step1_create_test_dataset.py` | Create 20K test set | Stratified sampling, domain distribution | Balanced test dataset from Phase 0 |
| `scripts/recover_failed_rewrites.py` | Recover failures | Retry logic, error handling | Recovers incomplete rewrites |
| `scripts/step3_prepare_copilot_batches.py` | Prepare critique batches | Batch splitting (34 examples each), format conversion | 73 batches for critique generation |
| `scripts/step3_merge_copilot_results.py` | Merge critiques | Batch consolidation, deduplication | Combines 73 batches → 2,484 critiques |
| `data/consolidate_batches_for_phase1c.py` | Consolidate for training | Final merge, validation | Creates phase1c_self_critique_train.jsonl |
| **Failure Export** |||
| `scripts/extract_all_failures_prepare_training.py` | Export hard failures | Filter, format conversion | Creates phase1c_hard_failures.jsonl (4,942) |

**Documentation:**
- `README.md` - Phase 1B overview, 12-step pipeline, results
- `docs/PHASE1B_QUICKSTART.md` - Quick start for benchmarking
- `GENERATION_GUIDE.py` - Guide for critique batch generation

**Data:**
- `data/test_dataset_20k.jsonl` - 20K test cases
- `data/phase1c_hard_failures.jsonl` - 4,942 hard failure cases
- `data/phase1c_self_critique_train.jsonl` - 2,484 self-critique examples with authentic critiques
- `data/copilot_batches/` - 73 critique batch results (Copilot + Claude Sonnet 4.5)

---

## Phase 1C: Targeted Distillation

**Status:** ⏳ READY TO TRAIN  
**Location:** `Phase1C_Targeted_Distillation/`  
**Input:** 7,426 training examples (2,484 self-critique + 4,942 hard failures)  
**Expected Output:** 15GB enhanced model, 88-92% GPT-4 performance

| File | Purpose | Methods/Algorithms | Achieves |
|------|---------|-------------------|----------|
| **Training** |||
| `scripts/train_phase1c_combined_smart.py` | Direct training with early stopping | LoRA (rank 64), early stopping (patience=3), convergence detection | Trains on 7,426 examples, 5-7h, $15-20 cost |
| **Legacy Scripts (Deprecated)** |||
| `scripts/generate_claude_examples.py` | ❌ DEPRECATED - Claude generation | API generation | Replaced by direct training |
| `scripts/create_bidirectional_pairs.py` | ❌ DEPRECATED - Bidirectional pairs | Forward-backward pairing | Removed in strategic pivot |
| `scripts/run_phase1c_combined_workflow.sh` | ❌ DEPRECATED - Full workflow | Multi-step pipeline | Replaced by direct training |
| **Training Methods** |||
| `scripts/self_consistency_distillation.py` | Self-consistency training | Multi-sample agreement, voting | Improves consistency |
| `scripts/deterministic_distillation.py` | Deterministic training | Fixed random seed, reproducible | Ensures reproducibility |
| `scripts/self_reinforcement_training.py` | Self-reinforcement | Reward-based learning | Improves on own outputs |

**Documentation:**
- `README.md` - Phase 1C overview, pivot documentation, critique generation context
- `docs/PHASE1CD_QUICKSTART.md` - ✅ UPDATED - Direct training guide with correct paths
- `scripts/README.md` - Scripts overview, workflow

**Data:**
- `data/phase1c_combined_direct.jsonl` - ✅ CREATED - 7,426 combined examples ready for training

**Key Context:**
- **Critique Generation:** GitHub Copilot powered by Claude Sonnet 4.5
- **Root Cause Finding:** Hard failures mainly from JSON parsing errors, truncation (NOT task difficulty)
- **Strategic Pivot (Nov 8, 2025):** Removed bidirectional pairs approach, saved $150-165

---

## Phase 2: Compression

**Status:** ❌ PENDING (blocked on Phase 1C)  
**Location:** `Phase2_Compression/`, `src/phase2_compression/`  
**Target:** 15GB → 520MB (19.2x compression)

| File | Purpose | Methods/Algorithms | Achieves |
|------|---------|-------------------|----------|
| **Planned Scripts** |||
| `scripts/neural_magic_prune.py` | Structured pruning | Neural Magic llm-compressor, 65% sparsity | 15GB → 3.5GB |
| `scripts/awq_quantize.py` | Mixed-precision quantization | AutoAWQ 4-bit, group size 128 | 3.5GB → 900MB |
| `scripts/gguf_export.py` | GGUF conversion | llama.cpp, Q5_K_M format | 900MB → 600MB |
| `scripts/zstd_compress.py` | Lossless compression | Zstandard level 10, dictionary training | 600MB → 500MB |
| `scripts/recovery_finetune.py` | Quality recovery | LoRA fine-tuning on hardest 12K | 500MB → 520MB with +1-2% quality |

**Notebooks:**
- `notebooks/Phase2_Compression_Colab.ipynb` - Compression pipeline in Colab

**Documentation:**
- `docs/PHASE2_QUICK_START.md` - Compression quick start guide

**Configuration:**
- `configs/compression.yaml` - Compression parameters and thresholds

---

## Phase 3: Domain Modifiers

**Status:** ❌ PENDING (blocked on Phase 2)  
**Location:** `Phase3_Modifiers/`, `src/phase3_modifiers/`  
**Target:** 3 specialized modifiers (47MB code + 48MB reasoning + 40MB automation)

| File | Purpose | Methods/Algorithms | Achieves |
|------|---------|-------------------|----------|
| **Planned Scripts** |||
| `scripts/train_code_modifier.py` | Code specialization | 3-tier teaching, LoRA rank 128, pruning | 47MB, 115-130% GPT-4 on HumanEval |
| `scripts/train_reasoning_modifier.py` | Reasoning specialization | 3-tier teaching, LoRA rank 112, CoT | 48MB, 100-108% GPT-4 on MMLU |
| `scripts/train_automation_modifier.py` | Automation specialization | 3-tier teaching, LoRA rank 96, tool-use | 40MB, 105-118% GPT-4 on ToolBench |
| `scripts/compress_modifier.py` | Modifier compression | Pruning 78-85% sparsity | 260MB → 40-48MB per modifier |

**3-Tier Teaching Strategy:**
- Tier 1 (60-70%): Free/cheap models (Llama-405B, Qwen-Coder-480B, Claude-3.5)
- Tier 2 (20-25%): Mid-tier models (GPT-4o, DeepSeek-Coder)
- Tier 3 (10-15%): GPT-5 for hardest cases only

**Configuration:**
- `configs/modifiers.yaml` - Modifier training parameters

---

## Phase 4: Router System

**Status:** ❌ PENDING (blocked on Phase 3)  
**Location:** `Phase4_Router/`, `src/phase4_router/`  
**Target:** 13MB router + 3MB escalation detector

| File | Purpose | Methods/Algorithms | Achieves |
|------|---------|-------------------|----------|
| **Planned Scripts** |||
| `scripts/train_router.py` | Route training | 3-layer feedforward, confidence scores | 13MB, 97% routing accuracy |
| `scripts/train_escalation_detector.py` | Escalation detection | BERT distillation → LSTM | 3MB, 94% detection accuracy |
| `scripts/optimize_thresholds.py` | Threshold tuning | A/B testing on 5K queries | Optimal 80% confidence threshold |

**Configuration:**
- `configs/router.yaml` - Router architecture and thresholds

---

## Phase 5: Deployment

**Status:** ❌ PENDING (blocked on Phase 4)  
**Location:** `Phase5_Deployment/`, `src/phase5_deployment/`  
**Target:** HuggingFace deployment, Gradio interface, monitoring

| File | Purpose | Methods/Algorithms | Achieves |
|------|---------|-------------------|----------|
| **Planned Scripts** |||
| `scripts/upload_huggingface.py` | HF upload | Model card generation, repo creation | Uploads 668MB system |
| `scripts/setup_inference_api.py` | Inference API | T4 GPU serverless setup | REST API with streaming |
| `scripts/create_gradio_interface.py` | UI creation | Gradio chat interface | User-facing chat demo |
| `scripts/setup_monitoring.py` | Monitoring | Grafana dashboard setup | Real-time metrics tracking |

---

## Shared Utilities

**Location:** `shared/utils/`  
**Purpose:** Reusable components across all phases

| File | Purpose | Methods/Algorithms | Achieves |
|------|---------|-------------------|----------|
| `deduplication_parallel.py` | ✅ ACTIVE - Parallel deduplication | MinHash LSH, xxhash, multiprocessing | 3000+ examples/sec, Jaccard 0.8 |
| `batch_api.py` | Batch API wrapper | OpenAI/Anthropic batch API, async | 50% cost reduction, rate limiting |
| `cost_tracker.py` | Cost tracking | Token counting, API cost calculation | Real-time cost monitoring |
| `logging.py` | Logging utilities | Rich formatting, structured logs | Consistent logging across phases |
| `validation.py` | Data validation | Schema checks, format validation | Ensures data quality |
| `verify_h100_environment.py` | H100 validation | CUDA checks, driver verification | Validates H100 GPU setup |
| **Archive (Deprecated)** |||
| `archive/deduplication.py` | ❌ DEPRECATED - MD5 version | MD5 hashing (slower) | Replaced by xxhash version |
| `archive/deduplication_optimized.py` | ❌ SUPERSEDED - Sequential version | xxhash (sequential) | Replaced by parallel version |

---

## Documentation

**Location:** `docs/`, `.github/`

### Core Documentation

| File | Purpose | Content Type |
|------|---------|--------------|
| `docs/CURRENT_STATUS.md` | **PRIMARY** - Project status and changelog | High-level audit trail, searchable decisions |
| `docs/IMPLEMENTATION_CHECKLIST.md` | **MASTER PROJECT PLAN** - Detailed task tracking | Granular checklist with status markers (✅/⏳/❌) |
| `docs/EXECUTION_PLAN.md` | Project roadmap | Phase overview, high-level milestones |
| `docs/technical_specification.md` | Technical details | Algorithms, parameters, implementation specifics |
| `README.md` | Project overview | Quick start, benchmarks, system specs |

### Phase-Specific Documentation

| File | Purpose | Phase |
|------|---------|-------|
| `docs/PHASE0_COMPLETE_SUMMARY.md` | Phase 0 completion summary | Phase 0 |
| `docs/PHASE1A_QUICKSTART.md` | Phase 1A quick start | Phase 1A |
| `docs/PHASE1_QUALITY_VALIDATION.md` | Phase 1 quality criteria | Phase 1 |
| `docs/PHASE2_QUICK_START.md` | Phase 2 compression guide | Phase 2 |
| `Phase1B_Failure_Analysis/README.md` | Phase 1B 12-step pipeline | Phase 1B |
| `Phase1C_Targeted_Distillation/README.md` | Phase 1C overview with pivot context | Phase 1C |
| `Phase1C_Targeted_Distillation/docs/PHASE1CD_QUICKSTART.md` | ✅ Phase 1C training guide (UPDATED) | Phase 1C |

### Development Documentation

| File | Purpose |
|------|---------|
| `docs/H100_UNSLOTH_MIGRATION.md` | H100 optimization migration |
| `docs/H100_QUICK_REFERENCE.md` | H100 setup commands |
| `docs/H100_NOTEBOOK_CHANGES.md` | H100 notebook modifications |
| `docs/PARALLEL_TASKS_DURING_TRAINING.md` | Tasks to do during training |
| `docs/QUICK_PARALLEL_TASKS.md` | Quick parallel task list |
| `docs/ISSUES_LOG.md` | Bug tracking and solutions |

### GitHub Configuration

| File | Purpose |
|------|---------|
| `.github/copilot-instructions.md` | AI assistant guidelines and context |
| `.github/instructions/copilot-instructions.md` | Duplicate of above (VS Code compatibility) |
| `.github/IMPLEMENTATION_CHECKLIST.md` | Master project plan (authoritative) |
| `.github/**PROJECT_STRUCTURE.md` | Project structure navigation |
| `.github/FILE_REGISTRY.md` | **THIS FILE** - Complete file inventory |

---

## Configuration

**Location:** `configs/`

| File | Purpose | Phase |
|------|---------|-------|
| `base_training.yaml` | Base training hyperparameters | Phase 1A |
| `distillation_training.yaml` | Distillation parameters | Phase 1C |
| `compression.yaml` | Compression settings | Phase 2 |
| `modifiers.yaml` | Modifier training config | Phase 3 |
| `router.yaml` | Router architecture | Phase 4 |

---

## Maintenance Guidelines

### When to Update This File

1. **After creating new scripts** - Add to appropriate phase section
2. **After deprecating files** - Move to archive section, add deprecation note
3. **After major refactoring** - Update file locations and purposes
4. **After completing phases** - Update status markers (❌ → ⏳ → ✅)
5. **After pivots/changes** - Update documentation references

### Update Protocol

```bash
# 1. Update FILE_REGISTRY.md with new entries
# 2. Commit with descriptive message
git add .github/FILE_REGISTRY.md
git commit -m "docs: update FILE_REGISTRY.md - [describe changes]"
git push
```

### Status Markers

- ✅ **COMPLETE** - Phase/file fully implemented and tested
- ⏳ **IN PROGRESS** - Currently being worked on
- ❌ **PENDING** - Not started, blocked, or planned
- 🔧 **DEPRECATED** - Old version, do not use
- 🆕 **NEW** - Recently added (within last week)

---

## Quick Navigation

### Find Files by Purpose

**Data Processing:** Phase0_Dataset/scripts/, shared/utils/deduplication_parallel.py  
**Model Training:** Phase1A_2_0/scripts/, Phase1C_Targeted_Distillation/scripts/train_phase1c_combined_smart.py  
**Evaluation:** Phase1B_Failure_Analysis/scripts/step3_*.py  
**Compression:** Phase2_Compression/scripts/ (planned)  
**Deployment:** Phase5_Deployment/scripts/ (planned)  
**Utilities:** shared/utils/  
**Documentation:** docs/, .github/

### Find Files by Status

**Active Use:** deduplication_parallel.py, train_phase1c_combined_smart.py, step3_llm_evaluation.py  
**Recently Updated:** PHASE1CD_QUICKSTART.md, FILE_REGISTRY.md, IMPLEMENTATION_CHECKLIST.md  
**Deprecated:** create_bidirectional_pairs.py, generate_claude_examples.py, deduplication.py (MD5)

---

**Last Updated:** November 9, 2025  
**Next Review:** After Phase 1C training completion  
**Maintained By:** Project lead + GitHub Copilot
