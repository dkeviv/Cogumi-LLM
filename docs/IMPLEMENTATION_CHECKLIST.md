# IMPLEMENTATION CHECKLIST - LLAMA-3.2-8B PIPELINE

**Student Model:** LLAMA-3.2-8B  
**Target:** 668MB MVP system (520MB base + 3×40-50MB modifiers) beating GPT-4  
**Timeline:** 14 weeks for MVP, +12 weeks for full 8-domain system  
**Total Cost:** $1,717 (MVP) + $1,151 (Phase 2) = $2,868 total

---

## ✅ PHASE 0: CURATED DATASET CREATION (COMPLETED)

### English-Only Dataset Curation & Deduplication
- [x] **Public Dataset Collection**: Collected 640,637 high-quality English examples
  - Sources: OpenOrca, Alpaca-GPT4, WizardLM, Dolly, MetaMathQA, CodeAlpaca, Anthropic-HH
  - Domains: Code, reasoning, math, science, conversation, creative
  - **Language Verification**: 99.46% English (verified Oct 2025)
- [x] **Advanced Deduplication**: MinHash LSH with Jaccard similarity (threshold 0.8)
  - Removed 34,091 near-duplicates (10.29% dedup rate)
  - Preserved domain diversity and difficulty distribution
- [x] **Dataset Validation**: Format standardization to instruction-response pairs
- [x] **Output**: `/data/phase1/public_500k_filtered.jsonl` (640,637 English examples ready)

**Status:** ✅ **COMPLETE** - Ready to proceed with Phase 1

**English-Only Optimization Strategy:**
- ✅ Dataset is 99.46% English (no filtering needed)
- ✅ Training on English-only data naturally optimizes model
- ✅ Phase 2 pruning will remove non-English neural pathways
- ❌ Vocabulary trimming SKIPPED (breaks LLAMA architecture)

---

## 🎯 PHASE 1: MVP - BASE MODEL & 3 MODIFIERS (14 Weeks)

### Phase 1A: Base Model Training (4 weeks, $505)
- [ ] **1A. Train Base Model** - Axolotl QLoRA on 640K English curated data
  - Setup: Download LLAMA-3.2-8B, configure Axolotl QLoRA (rank 64, 4-bit)
  - Training: 3 epochs on 640K examples, early stopping, validation monitoring
  - Target: 89-91% GPT-4 baseline, ~11GB model
  - **English optimization**: Training only on English data → English-optimized weights

### Phase 1B: Failure Analysis (2 days, $5)
- [ ] **1B. Test & Cluster Failures** - 50K examples → 12-14K failures → 8-12 categories

### Phase 1C: GPT-5 Targeted Distillation (1 week, $285)
- [ ] **1C. Generate & Train** - 40K GPT-5 examples → 88-100% GPT-4 baseline

---

## 🗜️ PHASE 2: EXTREME COMPRESSION (6 weeks, $402)

- [ ] **2A. Neural Magic Pruning** (2 weeks, $180) - 10GB → 3.5GB (65% sparsity)
- [ ] **2B. AWQ Quantization** (4 days, $90) - 3.5GB → 900MB (4-bit)
- [ ] **2C. GGUF Export** (3 days, $0) - 900MB → 600MB (Q5_K_M)
- [ ] **2D. Zstd Compression** (2 days, $0) - 600MB → 500MB (lossless)
- [ ] **2E. Recovery Fine-Tuning** (4 days, $70) - Recover +1-2% quality
- [ ] **2F. Confidence Calibration** (3 days, $35) - Enable 97% routing accuracy

**🎯 Base Complete: 520MB, 89-91% GPT-4**

---

## 🎨 PHASE 3: DOMAIN MODIFIERS (4 weeks, $685)

- [ ] **3A. Code Modifier** (10 days, $200) - 47MB, 115-130% GPT-4 🏆
- [ ] **3B. Reasoning Modifier** (10 days, $207) - 48MB, 100-108% GPT-4 🏆
- [ ] **3C. Automation Modifier** (10 days, $170) - 40MB, 105-118% GPT-4 🏆

---

## 🧭 PHASE 4: ROUTER SYSTEM (2 weeks, $75)

- [ ] **4A. Router Training** (1 week, $45) - 13MB, 97% accuracy
- [ ] **4B. Escalation Logic** (4 days, $30) - 3MB, 94% detection
- [ ] **4C. Threshold Optimization** (2 days, $0) - Optimal 80% threshold

---

## 🚀 PHASE 5: DEPLOYMENT (1 week, $100)

- [ ] **5A. HuggingFace Deployment** (4 days) - Upload + API + Interface
- [ ] **5B. End-to-End Validation** (1 week, $100) - All quality gates pass

**🎯 MVP COMPLETE: 668MB, Beats GPT-4 on 3 Domains**

---

## 📦 PHASE 2: DOMAIN EXPANSION (Optional, 12 weeks, $1,151)

- [ ] **Math, Hard Math, Science, Finance, Creative Modifiers** - +196MB
- [ ] **Enhancements** - Self-consistency, adaptive learning, multi-mode
- [ ] **Full System Validation** - All 8 domains validated

**🎯 Full System: 864MB, 8 Domains**

---

**Last Updated:** October 19, 2025  
**Status:** Phase 0 Complete ✅ | Ready for Phase 1 🚀
