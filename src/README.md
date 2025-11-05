# Source Code Organization

This directory contains all source code organized by pipeline phase, aligned with `docs/` structure.

## Directory Structure

```
src/
├── phase0_dataset/              # Phase 0: Dataset Creation
├── phase1a_base_training/       # Phase 1A: Base Model Training
├── phase1b_failure_analysis/    # Phase 1B: Failure Analysis & Clustering
├── phase1c_targeted_distillation/  # Phase 1C/1D: Targeted Distillation (Combined)
├── phase1e_speed_infrastructure/   # Phase 1E-1H: Speed Optimizations
├── phase2_extreme_compression/     # Phase 2: Extreme Compression
├── phase3_code_modifier/           # Phase 3: Code Domain Modifier
├── phase4_reasoning_modifier/      # Phase 4: Reasoning Domain Modifier
├── phase5_automation_modifier/     # Phase 5: Automation Domain Modifier
├── phase6_adaptive_router/         # Phase 6: Adaptive Routing System
├── phase7_meta_learning/           # Phase 7: Meta-Learning
├── phase8_deployment/              # Phase 8: Deployment & APIs
├── phase9_validation/              # Phase 9: Final Validation
└── utils/                          # Shared utilities
```

## Phase Status

| Phase | Directory | Status | Scripts |
|-------|-----------|--------|---------|
| **Phase 0** | `phase0_dataset/` | ✅ Complete | Dataset creation |
| **Phase 1A** | `phase1a_base_training/` | ✅ Complete | Training, merging, validation |
| **Phase 1B** | `phase1b_failure_analysis/` | ✅ Complete | Failure identification |
| **Phase 1C/1D** | `phase1c_targeted_distillation/` | ⏳ Ready | Generation, training, workflow |
| **Phase 1E-1H** | `phase1e_speed_infrastructure/` | ⏳ Pending | Draft model, speculative decoding |
| **Phase 2** | `phase2_extreme_compression/` | ⏳ Pending | Pruning, quantization, GGUF |
| **Phase 3** | `phase3_code_modifier/` | ⏳ Pending | Code domain LoRA |
| **Phase 4** | `phase4_reasoning_modifier/` | ⏳ Pending | Reasoning domain LoRA |
| **Phase 5** | `phase5_automation_modifier/` | ⏳ Pending | Automation domain LoRA |
| **Phase 6** | `phase6_adaptive_router/` | ⏳ Pending | Adaptive routing |
| **Phase 7** | `phase7_meta_learning/` | ⏳ Pending | Meta-learning system |
| **Phase 8** | `phase8_deployment/` | ⏳ Pending | Deployment scripts |
| **Phase 9** | `phase9_validation/` | ⏳ Pending | Final validation |

## Quick Navigation

### Current Phase: Phase 1C/1D

**Location:** `src/phase1c_targeted_distillation/`

**Scripts:**
- `generate_claude_examples.py` - Generate improved examples
- `create_bidirectional_pairs.py` - Create forward + reverse pairs
- `train_phase1c_combined_smart.py` - Smart training with early stopping
- `run_phase1c_combined_workflow.sh` - Complete automated workflow

**Documentation:** See `docs/PHASE1CD_QUICKSTART.md`

**Execute:**
```bash
./src/phase1c_targeted_distillation/run_phase1c_combined_workflow.sh
```

### Utilities

**Location:** `src/utils/`

**Contains:**
- GPU diagnostics
- Environment verification
- Setup scripts
- Shared helper functions

## Naming Conventions

- **Phase folders:** `phaseN_descriptive_name/`
- **Script files:** `action_phase_variant.py`
- **README:** Each phase has a README.md explaining its purpose

## Documentation Alignment

Each phase folder aligns with:
- `docs/phaseN_*/` - Detailed documentation
- `docs/EXECUTION_PLAN.md` - High-level roadmap
- `docs/IMPLEMENTATION_CHECKLIST.md` - Task tracking

## Legacy Folders (To Be Archived)

The following top-level folders contain older phase implementations:
- `Phase0/` - Original Phase 0 implementation
- `Phase1A_2_0/` - Phase 1A results (15GB model)
- `Phase 1B_2_0/` - Phase 1B results (failure data)
- `archive_old_src/` - Previous src structure

**Note:** These are kept for data/model outputs. Scripts have been moved to `src/`.
