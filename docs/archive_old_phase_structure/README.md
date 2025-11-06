# Archived Phase Documentation Structure

**Archive Date**: January 2025  
**Reason**: Repository reorganization to phase-centric structure

## What's Here

This folder contains the **old phase documentation structure** from when all phase-specific docs were organized under `docs/`.

### Old Structure (DEPRECATED)
```
docs/
├── phase0_dataset/
├── phase1a_2_0_full_precision/
├── phase1b_2_0_failure_analysis/
├── phase1c_targeted_distillation/
├── phase2_compression/
├── h100_training/
├── phase3-9_*/
└── post_mvp/
```

### New Structure (CURRENT)
```
Phase0_Dataset/docs/         # Phase 0 documentation
Phase1A_Base_Training/docs/  # Phase 1A + H100 guides
Phase1B_Failure_Analysis/docs/
Phase1C_Targeted_Distillation/docs/
Phase2_Compression/docs/
Phase3-9_[...]/docs/
```

## Migration Map

| Old Location | New Location | Status |
|--------------|--------------|--------|
| `docs/phase0_dataset/*` | `Phase0_Dataset/docs/` | ✅ Migrated |
| `docs/phase1a_2_0_full_precision/*` | `Phase1A_Base_Training/docs/` | ✅ Migrated |
| `docs/h100_training/*` | `Phase1A_Base_Training/docs/` | ✅ Migrated |
| `docs/phase1b_2_0_failure_analysis/*` | `Phase1B_Failure_Analysis/docs/` | ✅ Migrated |
| `docs/phase1c_targeted_distillation/*` | `Phase1C_Targeted_Distillation/docs/` | ✅ Migrated |
| `docs/PHASE1CD_QUICKSTART.md` | `Phase1C_Targeted_Distillation/docs/` | ✅ Migrated |
| `docs/AWS_SETUP_PHASE1CD.md` | `Phase1C_Targeted_Distillation/docs/` | ✅ Migrated |
| `docs/phase2_compression/*` | `Phase2_Compression/docs/` | ✅ Migrated |
| `docs/phase3-9_*/*` | `Phase3-9_[...]/docs/` | Empty folders (future use) |
| `docs/post_mvp/*` | Future phases | Not yet implemented |

## What to Keep in docs/

**Only project-wide documentation:**
- EXECUTION_PLAN.md
- IMPLEMENTATION_CHECKLIST.md
- CURRENT_STATUS.md
- technical_specification.md
- ISSUES_LOG.md
- general_guides/
- dev/
- archive/

**Phase-specific docs** now live with their respective phases.

## Rationale

**Old approach issues:**
1. Confusion between `Phase1A_2_0/` folder and `docs/phase1a_2_0_full_precision/`
2. Scripts in one place, documentation in another
3. Hard to find phase-related content
4. No clear ownership of files

**New approach benefits:**
1. One phase = one folder with everything (scripts, data, models, docs)
2. Self-contained phases
3. Clear navigation
4. Simplified structure

## Recovery

If you need to reference old documentation structure:
```bash
# View this archive
ls docs/archive_old_phase_structure/

# Compare with new structure
ls Phase0_Dataset/docs/
ls Phase1A_Base_Training/docs/
```

## Related Documentation

- `PROJECT_STRUCTURE.md` - Complete new structure overview
- `docs/PHASE_STRUCTURE_MIGRATION.md` - Detailed migration guide
- `docs/README.md` - Updated documentation index

---

**Do not use files from this archive.** Refer to `Phase*/docs/` for current documentation.
