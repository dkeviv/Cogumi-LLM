# Cogumi-LLM Project Structure

**Last Updated:** November 5, 2025  
**Reorganization:** Simplified phase-based structure

## 🎯 Simplified Structure Philosophy

**One phase, one folder.** Each phase contains everything it needs:
- **scripts/** - All executable code (Python, shell scripts)
- **data/** - All data files specific to this phase
- **models/** - Trained models (for phases that produce models)
- **docs/** - Phase-specific documentation

**No more confusion between `src/` and `scripts/`** - they're the same thing!

---

## 📁 Directory Structure

```
Cogumi-LLM/
│
├── Phase0_Dataset/                    # Phase 0: Dataset Creation ✅
│   ├── scripts/                       # Dataset generation, filtering, deduplication
│   ├── data/                          # 600K curated examples
│   └── docs/                          # Phase 0 documentation
│
├── Phase1A_Base_Training/             # Phase 1A: Base Model Training ✅
│   ├── scripts/                       # Training, merging, validation scripts
│   ├── data/                          # Training data
│   ├── models/                        # 15GB trained base model
│   └── docs/                          # Training guides, H100 setup
│
├── Phase1B_Failure_Analysis/          # Phase 1B: Failure Analysis ✅
│   ├── scripts/                       # Failure identification, clustering
│   ├── data/                          # 4,942 hard failures, clusters
│   └── docs/                          # Analysis methodology
│
├── Phase1C_Targeted_Distillation/     # Phase 1C/1D: Combined Training ⏳
│   ├── scripts/                       # Claude generation, bidirectional pairs, training
│   ├── data/                          # Improved examples, training pairs
│   └── docs/                          # Quick start guide
│
├── Phase1E_Speed_Infrastructure/      # Phase 1E-1H: Speed Optimizations ⏳
│   ├── scripts/                       # Draft model, speculative decoding
│   ├── data/                          # Speed benchmark data
│   └── docs/                          # Optimization guides
│
├── Phase2_Compression/                # Phase 2: Extreme Compression ⏳
│   ├── scripts/                       # Pruning, quantization, GGUF export
│   ├── data/                          # Calibration data
│   ├── models/                        # Compressed models
│   └── docs/                          # Compression methodology
│
├── Phase3_Code_Modifier/              # Phase 3: Code Domain Modifier ⏳
│   ├── scripts/                       # Code domain training
│   ├── data/                          # Code examples
│   ├── models/                        # Code modifier (47MB)
│   └── docs/                          # Code domain docs
│
├── Phase4_Reasoning_Modifier/         # Phase 4: Reasoning Domain Modifier ⏳
│   ├── scripts/                       # Reasoning domain training
│   ├── data/                          # Reasoning examples
│   ├── models/                        # Reasoning modifier (48MB)
│   └── docs/                          # Reasoning domain docs
│
├── Phase5_Automation_Modifier/        # Phase 5: Automation Domain Modifier ⏳
│   ├── scripts/                       # Automation domain training
│   ├── data/                          # Automation examples
│   ├── models/                        # Automation modifier (40MB)
│   └── docs/                          # Automation domain docs
│
├── Phase6_Router/                     # Phase 6: Adaptive Router ⏳
│   ├── scripts/                       # Router training, escalation detection
│   ├── data/                          # Routing examples
│   ├── models/                        # Router (13MB) + escalation (3MB)
│   └── docs/                          # Router architecture
│
├── Phase7_Meta_Learning/              # Phase 7: Meta-Learning ⏳
│   ├── scripts/                       # Meta-learning training
│   ├── data/                          # Meta-learning data
│   ├── models/                        # Meta-learner (12MB)
│   └── docs/                          # Meta-learning docs
│
├── Phase8_Deployment/                 # Phase 8: Deployment ⏳
│   ├── scripts/                       # HuggingFace upload, API setup, Gradio
│   ├── data/                          # Deployment configs
│   └── docs/                          # Deployment guides
│
├── Phase9_Validation/                 # Phase 9: Final Validation ⏳
│   ├── scripts/                       # Benchmarking, human eval
│   ├── data/                          # Validation results
│   └── docs/                          # Validation reports
│
├── shared/                            # Shared across all phases
│   ├── scripts/                       # Common utilities
│   ├── utils/                         # Helper functions
│   └── configs/                       # Shared configurations
│
├── docs/                              # Project-wide documentation
│   ├── EXECUTION_PLAN.md              # High-level roadmap
│   ├── IMPLEMENTATION_CHECKLIST.md    # Task tracking
│   ├── CURRENT_STATUS.md              # Progress and decisions
│   ├── technical_specification.md     # Complete technical details
│   └── phase*_*/                      # Phase-specific doc folders
│
├── notebooks/                         # Jupyter notebooks
│   ├── H100_Training_Clean.ipynb
│   └── Phase*_*.ipynb
│
├── tests/                             # Test files
│
└── archive_old_*/                     # Legacy folders (deprecated)
```

---

## 🚀 Quick Navigation

### Current Phase: Phase 1C

**Location:** `Phase1C_Targeted_Distillation/`

**Execute Workflow:**
```bash
cd Phase1C_Targeted_Distillation/scripts
./run_phase1c_combined_workflow.sh
```

**Manual Steps:**
```bash
cd Phase1C_Targeted_Distillation/scripts

# Generate examples
python generate_claude_examples.py [args]

# Create pairs
python create_bidirectional_pairs.py [args]

# Train
python train_phase1c_combined_smart.py [args]
```

**Data Location:** `Phase1C_Targeted_Distillation/data/`

**Documentation:** `Phase1C_Targeted_Distillation/docs/` or `docs/PHASE1CD_QUICKSTART.md`

---

## 📊 Phase Status

| Phase | Folder | Status | Output |
|-------|--------|--------|--------|
| **0** | Phase0_Dataset | ✅ Complete | 600K examples |
| **1A** | Phase1A_Base_Training | ✅ Complete | 15GB base model |
| **1B** | Phase1B_Failure_Analysis | ✅ Complete | 4,942 failures |
| **1C/1D** | Phase1C_Targeted_Distillation | ⏳ Ready | Scripts ready |
| **1E-1H** | Phase1E_Speed_Infrastructure | ⏳ Pending | Not started |
| **2** | Phase2_Compression | ⏳ Pending | Not started |
| **3** | Phase3_Code_Modifier | ⏳ Pending | Not started |
| **4** | Phase4_Reasoning_Modifier | ⏳ Pending | Not started |
| **5** | Phase5_Automation_Modifier | ⏳ Pending | Not started |
| **6** | Phase6_Router | ⏳ Pending | Not started |
| **7** | Phase7_Meta_Learning | ⏳ Pending | Not started |
| **8** | Phase8_Deployment | ⏳ Pending | Not started |
| **9** | Phase9_Validation | ⏳ Pending | Not started |

---

## 🔧 Benefits of New Structure

### 1. **Simplicity**
- One phase = one folder
- No confusion between `src/` and `scripts/`
- Clear ownership of files

### 2. **Self-Contained**
- Each phase has everything it needs
- Easy to zip/share a single phase
- No hunting across multiple folders

### 3. **Consistency**
- Same structure for all phases
- Predictable locations
- Easy navigation

### 4. **Scalability**
- Easy to add new phases
- Template structure for future work
- Clean separation of concerns

### 5. **Discovery**
- Want Phase 1C scripts? → `Phase1C_Targeted_Distillation/scripts/`
- Want Phase 1C data? → `Phase1C_Targeted_Distillation/data/`
- Want Phase 1C docs? → `Phase1C_Targeted_Distillation/docs/`

---

## 📝 File Naming Convention

### Scripts
- Descriptive names: `generate_claude_examples.py`, `train_phase1c_combined_smart.py`
- Action-oriented: verb + noun pattern
- Phase-specific prefixes when needed: `phase1a_`, `phase1b_`

### Data Files
- Phase prefix required: `Phase1A_`, `Phase1B_`, `Phase1C_`
- Descriptive suffix: `_hard_failures.jsonl`, `_training_data.jsonl`
- Examples:
  - `Phase1B_hard_failures.jsonl`
  - `Phase1C_improved_examples.jsonl`
  - `Phase1C_combined_training_bidirectional.jsonl`

### Models
- Phase and purpose: `Phase1A_base_model/`, `Phase3_code_modifier/`
- Size indicator helpful: `Phase1A_base_15gb/`

---

## 🗑️ Legacy Folders (To Be Archived)

The following folders are deprecated and can be archived after verification:

- `Phase0/` → migrated to `Phase0_Dataset/`
- `Phase1A_2_0/` → migrated to `Phase1A_Base_Training/`
- `Phase 1B_2_0/` → migrated to `Phase1B_Failure_Analysis/`
- `src/` → scripts moved to phase folders
- `scripts/` → scripts moved to phase folders
- `data/phase*/` → moved to respective phase folders
- `archive_old_src/` → already archived
- `archive_old_scripts/` → already archived

**Note:** Keep these folders until we verify all migrations are complete and working.

---

## 🚦 Migration Status

- ✅ Phase structure created
- ✅ Phase 0 content migrated
- ✅ Phase 1A content migrated
- ✅ Phase 1B content migrated
- ✅ Phase 1C content migrated
- ✅ Shared utilities migrated
- ⏳ Update all references in docs
- ⏳ Update notebooks
- ⏳ Test all scripts in new locations
- ⏳ Archive old folders

---

## 📖 Related Documentation

- `docs/EXECUTION_PLAN.md` - Overall project roadmap
- `docs/IMPLEMENTATION_CHECKLIST.md` - Detailed task tracking
- `docs/CURRENT_STATUS.md` - Current progress and decisions
- `docs/PHASE1CD_QUICKSTART.md` - Phase 1C/1D execution guide
- `README.md` - Project overview

---

**Status:** Structure created, content migration in progress
