# Project-Wide Documentation

## 📋 What's Here

**This folder contains ONLY project-wide documentation.** Phase-specific docs are now in `Phase*/docs/`.

### Project Management (Keep Here)
- **EXECUTION_PLAN.md** - 9-phase system overview, timeline, costs
- **IMPLEMENTATION_CHECKLIST.md** - Granular task tracking across all phases
- **CURRENT_STATUS.md** - Progress tracker, major decisions, changelog

### Technical Documentation (Keep Here)
- **technical_specification.md** - Complete algorithms, methods, parameters

### Cross-Phase Guides (Keep Here)
- **ISSUES_LOG.md** - Bug tracking and debugging knowledge base
- **general_guides/** - Guides that apply to multiple phases
- **dev/** - Development notes, pipeline evolution

---

**All phase-specific documentation has been moved to the respective phase folders:**

| Phase | Location | Documentation |
|-------|----------|---------------|
| **Phase 0** | `Phase0_Dataset/docs/` | Dataset curation, deduplication, MinHash LSH, validation |
| **Phase 1A** | `Phase1A_Base_Training/docs/` | Full precision bfloat16 training, H100 guides, QLoRA history |
| **Phase 1B** | `Phase1B_Failure_Analysis/docs/` | Haiku judging, failure clustering, 7,331 failures identified |
| **Phase 1C** | `Phase1C_Targeted_Distillation/docs/` | Self-critique pipeline, GPT-5 distillation, AWS setup |
| **Phase 1E** | `Phase1E_Speed_Infrastructure/docs/` | vLLM, TensorRT-LLM, speed optimization |
| **Phase 2** | `Phase2_Compression/docs/` | 5-stage compression (pruning, quantization, GGUF, zstd, recovery) |
| **Phase 3** | `Phase3_Code_Modifier/docs/` | Code modifier (47MB, 115-130% GPT-4) |
| **Phase 4** | `Phase4_Reasoning_Modifier/docs/` | Reasoning modifier (48MB, 100-108% GPT-4) |
| **Phase 5** | `Phase5_Automation_Modifier/docs/` | Automation modifier (40MB, 105-118% GPT-4) |
| **Phase 6** | `Phase6_Router_System/docs/` | Router (13MB) + Escalation detector (3MB) |
| **Phase 7** | `Phase7_Meta_Learning/docs/` | MAML training (12MB, +10-15% few-shot) |
| **Phase 8** | `Phase8_Deployment/docs/` | HuggingFace upload, Inference API, Gradio |
| **Phase 9** | `Phase9_Validation/docs/` | Automated quality gates, benchmarks |

### Quick Access to Phase Documentation

```bash
# Phase 0: Dataset Creation
cat Phase0_Dataset/docs/PHASE0_COMPLETE_SUMMARY.md

# Phase 1A: Base Training
cat Phase1A_Base_Training/docs/PHASE1A_QUICKSTART.md
cat Phase1A_Base_Training/docs/H100_QUICK_REFERENCE.md

# Phase 1C: Targeted Distillation
cat Phase1C_Targeted_Distillation/docs/README.md
cat Phase1C_Targeted_Distillation/docs/AWS_SETUP_PHASE1CD.md

# Phase 2: Compression
cat Phase2_Compression/docs/PHASE2_QUICK_START.md
```

### Supporting Documentation (Still in docs/)

- **general_guides/** - Cross-phase workflows, parallel tasks, benchmarking
- **dev/** - Technical methodologies, enhancement tables, pipeline evolution
- **archive/** - Deprecated Phase 1A 1.0 (QLoRA) and old approaches
- **archive2/** - Earlier archived materials

## 🔍 Finding Documentation

### By Phase (NEW Structure)

1. **Dataset Ready?** → `Phase0_Dataset/docs/PHASE0_COMPLETE_SUMMARY.md`
2. **Training Phase 1A?** → `Phase1A_Base_Training/docs/PHASE1A_QUICKSTART.md`
3. **Failure Analysis?** → `Phase1B_Failure_Analysis/docs/` (Haiku judging approach)
4. **Self-Critique Pipeline?** → `Phase1C_Targeted_Distillation/docs/README.md`
5. **Compression?** → `Phase2_Compression/docs/PHASE2_QUICK_START.md`
6. **H100 Training?** → `Phase1A_Base_Training/docs/H100_QUICK_REFERENCE.md`

### By Topic (Project-Wide)

- **Overall Progress** → `docs/IMPLEMENTATION_CHECKLIST.md` or `docs/CURRENT_STATUS.md`
- **Technical Details** → `docs/technical_specification.md`
- **Week-by-Week Plan** → `docs/EXECUTION_PLAN.md`
- **Bugs/Issues** → `docs/ISSUES_LOG.md`
- **Benchmarking** → `docs/general_guides/QUICK_START_BENCHMARK.md`

## � Current Status

- **Phase 0**: ✅ COMPLETE (600K curated dataset)
- **Phase 1A**: ✅ COMPLETE (15GB full precision base model)
- **Phase 1B**: ✅ COMPLETE (7,331 failures via Haiku judging)
- **Phase 1C**: ⏳ IN PROGRESS (self-critique pipeline ready, awaiting execution)
- **Overall Progress**: ~8% (Phase 0-1B complete, Phase 1C pending)

## 📁 Repository Structure

```
cogumi-llm/
├── docs/                              # PROJECT-WIDE documentation (this folder)
│   ├── EXECUTION_PLAN.md
│   ├── IMPLEMENTATION_CHECKLIST.md
│   ├── CURRENT_STATUS.md
│   ├── technical_specification.md
│   ├── ISSUES_LOG.md
│   ├── general_guides/
│   ├── dev/
│   └── archive/
│
├── Phase0_Dataset/                    # Phase 0: Dataset Creation
│   ├── scripts/                       # Deduplication, curation
│   ├── data/                          # 600K curated examples
│   └── docs/                          # Phase 0 documentation
│
├── Phase1A_Base_Training/             # Phase 1A: Base Model Training
│   ├── scripts/                       # Training scripts
│   ├── data/                          # Training data
│   ├── models/                        # 15GB base model
│   └── docs/                          # Phase 1A docs + H100 guides
│
├── Phase1B_Failure_Analysis/          # Phase 1B: Failure Analysis
│   ├── scripts/                       # Haiku judging, clustering
│   ├── data/                          # 7,331 failures
│   └── docs/                          # Phase 1B documentation
│
├── Phase1C_Targeted_Distillation/     # Phase 1C: Self-Critique + GPT-5
│   ├── scripts/                       # Self-critique pipeline
│   ├── data/                          # Distillation data
│   └── docs/                          # Phase 1C docs + AWS setup
│
├── Phase2_Compression/                # Phase 2: 5-Stage Compression
│   ├── scripts/                       # Pruning, quantization, GGUF
│   ├── models/                        # Compressed models
│   └── docs/                          # Phase 2 documentation
│
├── Phase3-9_[...]/                    # Future phases
│   ├── scripts/
│   ├── data/
│   ├── models/
│   └── docs/
│
└── shared/                            # Shared utilities
    ├── scripts/
    ├── utils/
    └── configs/
```

See `PROJECT_STRUCTURE.md` at root for complete structure documentation.

---

**Last Updated**: January 2025 (Structure reorganization complete)
