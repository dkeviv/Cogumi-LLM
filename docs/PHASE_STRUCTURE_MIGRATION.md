# Phase-Based Structure Reorganization

**Date:** November 5, 2025  
**Major Change:** Simplified to phase-centric structure

---

## 🎯 Problem Solved

**Before:**
- Scripts split across `src/`, `scripts/`, and `PhaseX/` folders
- Confusion between "src" and "scripts" (they're the same thing!)
- Data scattered across multiple locations
- Hard to find phase-specific files

**After:**
- **One phase = one folder**
- Each phase contains: `scripts/`, `data/`, `models/`, `docs/`
- Clear ownership and navigation
- Self-contained phases

---

## 📁 New Structure

```
Phase0_Dataset/
├── scripts/          # All executable code
├── data/             # Phase 0 datasets
└── docs/             # Phase 0 documentation

Phase1A_Base_Training/
├── scripts/          # Training scripts
├── data/             # Training data
├── models/           # 15GB base model
└── docs/             # Training guides

Phase1B_Failure_Analysis/
├── scripts/          # Analysis scripts
├── data/             # Failure data
└── docs/             # Analysis reports

Phase1C_Targeted_Distillation/
├── scripts/          # Generation, training scripts
├── data/             # Training pairs
└── docs/             # Guides

... (Phase 2-9 with same structure)

shared/
├── scripts/          # Cross-phase utilities
├── utils/            # Helper functions
└── configs/          # Shared configs
```

---

## 🔄 Migration Map

### Phase 0
```
Phase0/scripts/       → Phase0_Dataset/scripts/
Phase0/data/          → Phase0_Dataset/data/
```

### Phase 1A
```
src/phase1a_base_training/        → Phase1A_Base_Training/scripts/
scripts/phase1a_base_training/    → Phase1A_Base_Training/scripts/
Phase1A_2_0/scripts/              → Phase1A_Base_Training/scripts/
Phase1A_2_0/models/               → Phase1A_Base_Training/models/
Phase1A_2_0/docs/                 → Phase1A_Base_Training/docs/
```

### Phase 1B
```
Phase 1B_2_0/*.py                 → Phase1B_Failure_Analysis/scripts/
Phase 1B_2_0/data/                → Phase1B_Failure_Analysis/data/
Phase 1B_2_0/docs/                → Phase1B_Failure_Analysis/docs/
scripts/phase1b_failure_analysis/ → Phase1B_Failure_Analysis/scripts/
```

### Phase 1C
```
src/phase1c_targeted_distillation/     → Phase1C_Targeted_Distillation/scripts/
scripts/phase1c_targeted_distillation/ → Phase1C_Targeted_Distillation/scripts/
data/phase1c/                          → Phase1C_Targeted_Distillation/data/
```

### Shared
```
src/utils/     → shared/utils/
configs/       → shared/configs/
```

---

## 📝 File Naming Convention

### Data Files Must Have Phase Prefix
- ✅ `Phase1A_training_data.jsonl`
- ✅ `Phase1B_hard_failures.jsonl`
- ✅ `Phase1C_improved_examples.jsonl`
- ❌ `training_data.jsonl` (no phase prefix)
- ❌ `failures.jsonl` (no phase prefix)

### Scripts - Descriptive Names
- ✅ `generate_claude_examples.py`
- ✅ `train_phase1c_combined_smart.py`
- ✅ `run_phase1c_combined_workflow.sh`

---

## ✅ Benefits

1. **Simplicity:** No more src vs scripts confusion
2. **Self-Contained:** Each phase has everything it needs
3. **Discoverability:** Want Phase 1C? → `Phase1C_Targeted_Distillation/`
4. **Scalability:** Easy template for new phases
5. **Portability:** Can zip/share individual phases
6. **Consistency:** Same structure for all phases

---

## 🚀 Quick Reference

### Current Phase: Phase 1C

**Navigate:**
```bash
cd Phase1C_Targeted_Distillation
```

**Scripts:**
```bash
cd Phase1C_Targeted_Distillation/scripts
ls -la
```

**Data:**
```bash
cd Phase1C_Targeted_Distillation/data
ls -la
```

**Run Workflow:**
```bash
cd Phase1C_Targeted_Distillation/scripts
./run_phase1c_combined_workflow.sh
```

---

## 📖 Updated Documentation

- `PROJECT_STRUCTURE.md` - Complete new structure overview
- `Phase1C_Targeted_Distillation/README.md` - Phase 1C guide
- `docs/PHASE1CD_QUICKSTART.md` - Updated paths
- `docs/AWS_SETUP_PHASE1CD.md` - Updated paths

---

## 🗑️ Legacy Folders (To Archive Later)

These folders are deprecated but kept temporarily for verification:
- `Phase0/` - migrated
- `Phase1A_2_0/` - migrated
- `Phase 1B_2_0/` - migrated
- `src/phase1*/` - migrated
- `scripts/phase1*/` - migrated
- `data/phase*/` - migrated

**Action Required:** Archive after verification complete.

---

## ✨ Status

- ✅ New structure created
- ✅ Phase 0 migrated
- ✅ Phase 1A migrated
- ✅ Phase 1B migrated
- ✅ Phase 1C migrated
- ✅ Shared utilities migrated
- ✅ Documentation updated
- ⏳ Testing in new structure
- ⏳ Archive old folders

---

**Next Step:** Test Phase 1C workflow in new structure
