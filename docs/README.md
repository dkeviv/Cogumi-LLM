# Documentation Organization

## 📋 Main Tracking Documents (Root Level)

These files track overall project status and should remain at the root:

- **CURRENT_STATUS.md** - Single source of truth for project status and changelog
- **IMPLEMENTATION_CHECKLIST.md** - Detailed task tracking across all phases (0-15)
- **EXECUTION_PLAN.md** - Week-by-week execution guide and high-level roadmap
- **technical_specification.md** - Complete technical methodology and architecture
- **ISSUES_LOG.md** - Bug tracking and debugging knowledge base

## 📁 Phase-Specific Documentation

### MVP Phases (0-9)

- **phase0_dataset/** - Dataset curation, deduplication, validation
- **phase1a_1_0_qlora_failed/** - DEPRECATED: First QLoRA attempt (merge corruption issue)
- **phase1a_2_0_full_precision/** - Current: Full precision bfloat16 training (15GB model)
- **phase1b_2_0_failure_analysis/** - Haiku judging, failure analysis (7,331 failures identified)
- **phase1c_targeted_distillation/** - Self-critique + GPT-5 targeted distillation
- **phase2_compression/** - 5-stage compression pipeline
  - pruning/ - Neural Magic structured pruning (65% sparsity)
  - quantization/ - AWQ 4-bit quantization
  - gguf/ - GGUF export (Q5_K_M)
  - zstd/ - Zstandard lossless compression
  - recovery/ - Recovery fine-tuning with hard examples
  - calibration/ - Confidence calibration for routing
- **phase3_code_modifier/** - Code modifier (47MB, 115-130% GPT-4)
- **phase4_reasoning_modifier/** - Reasoning modifier (48MB, 100-108% GPT-4)
- **phase5_automation_modifier/** - Automation modifier (40MB, 105-118% GPT-4)
- **phase6_router_system/** - Router (13MB) + Escalation detector (3MB)
- **phase7_meta_learning/** - MAML training (12MB, +10-15% few-shot)
- **phase8_deployment/** - HuggingFace upload, Inference API, Gradio interface
- **phase9_validation/** - Automated quality gates, human evaluation, benchmarks

### Post-MVP Enhancements (10-15)

- **post_mvp/**
  - phase10_multi_mode/ - Fast vs Accurate mode architecture
  - phase11_self_consistency/ - Multi-path voting (N=5) for hard problems
  - phase12_self_critique/ - BERT-based critique classifier (10MB)
  - phase13_adaptive_learning/ - Adaptive threshold learning from user feedback
  - phase14_more_modifiers/ - 5 additional domain modifiers (196MB total)
  - phase15_shared_backbone/ - Optional shared backbone refactoring (if >15 domains)

### Supporting Documentation

- **h100_training/** - H100-specific guides, Unsloth migration, precompiled binaries
- **general_guides/** - Parallel tasks, benchmark quick starts, general workflows
- **dev/** - Detailed technical methodologies and enhancement tables
- **archive/** - Deprecated documentation from Phase 1A 1.0 and old approaches
- **archive2/** - (If exists) Earlier archived materials

## 🔍 Finding Documentation

### By Phase

1. **Dataset Ready?** → Check `phase0_dataset/PHASE0_COMPLETE_SUMMARY.md`
2. **Training Phase 1A?** → Check `phase1a_2_0_full_precision/PHASE1A_QUICKSTART.md`
3. **Failure Analysis?** → Check `phase1b_2_0_failure_analysis/` (Haiku judging approach)
4. **Self-Critique Pipeline?** → Check `phase1c_targeted_distillation/` (step9-12 scripts)
5. **Compression?** → Check `phase2_compression/PHASE2_QUICK_START.md`
6. **H100 Training?** → Check `h100_training/H100_QUICK_REFERENCE.md`

### By Topic

- **Overall Progress** → `IMPLEMENTATION_CHECKLIST.md` or `CURRENT_STATUS.md`
- **Technical Details** → `technical_specification.md`
- **Week-by-Week Plan** → `EXECUTION_PLAN.md`
- **Bugs/Issues** → `ISSUES_LOG.md`
- **H100 Setup** → `h100_training/H100_UNSLOTH_MIGRATION.md`
- **Benchmarking** → `general_guides/QUICK_START_BENCHMARK.md`

## 🗂️ Archival Policy

Files are archived when:

1. **Phase iteration superseded**: Phase 1A 1.0 (QLoRA) → archived after Phase 1A 2.0 (full precision)
2. **Approach pivoted**: Old Phase 1B files → archived after Phase 1B 2.0 (Haiku judging)
3. **One-time summaries**: Dependency updates, documentation updates, etc.
4. **Merge corruption related**: Catastrophic forgetting, benchmark analysis from failed QLoRA

## 📊 Current Status

- **Phase 0**: ✅ COMPLETE (640K dataset)
- **Phase 1A 1.0**: ❌ FAILED (QLoRA merge corruption)
- **Phase 1A 2.0**: ✅ COMPLETE (15GB full precision model)
- **Phase 1B 2.0**: ✅ COMPLETE (7,331 failures via Haiku judging)
- **Phase 1C**: ⏳ IN PROGRESS (self-critique pipeline ready, awaiting Vast.ai execution)
- **Overall Progress**: ~8% (Phase 0-1B complete, Phase 1C pending)

---

**Last Updated**: November 3, 2025
