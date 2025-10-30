# EXECUTION PLAN - LLAMA-3.2-8B COGUMI-LLM# **COGUMI-LLM: COMPLETE EXECUTION PLAN**



**Version:** 2.0  **Project Goal:** 480MB base model + modifiers achieving 99-100% GPT-4 general reasoning, 108-113% GPT-4 coding  

**Timeline:** 14 weeks for MVP | +12 weeks for full system  **Total Cost:** $882-982 | **Timeline:** 10-12 weeks | **Final Size:** ~600-620MB (480MB base + 120-140MB modifiers)

**Budget:** $1,717 MVP | $2,868 total  

**Automation:** 93% via Claude 4.5 script generation  ---

**Status:** Phase 0 Complete ✅ | Starting Phase 1

## **PHASE 0: INFRASTRUCTURE & SETUP** ✅ **COMPLETED**

---

| Step | Task | Success Criteria | Duration | Cost |

## QUICK REFERENCE|------|------|------------------|----------|------|

| **0.1** | Setup & Verification | ✅ API keys verified<br/>✅ Deduplication tested (75% retention)<br/>✅ All dependencies verified<br/>✅ Lint errors fixed<br/>✅ API clients preserved | 6-8 hrs | $0 |

| Phase | Duration | Cost | Status |

|-------|----------|------|--------|---

| **Phase 0: Dataset** | Complete | $0 | ✅ DONE |

| **Phase 1: Base Training** | 4 weeks | $505 | ⏳ Next |## **PHASE 1: DATASET PREPARATION** ⏸️ **NEXT (Week 1-2)**

| **Phase 2: Compression** | 6 weeks | $402 | ⏳ Pending |

| **Phase 3: Modifiers** | 4 weeks | $685 | ⏳ Pending || Step | Task | Success Criteria | Duration | Cost |

| **Phase 4: Router** | 2 weeks | $75 | ⏳ Pending ||------|------|------------------|----------|------|

| **Phase 5: Deployment** | 1 week | $100 | ⏳ Pending || **1.1** | Download Public Datasets | ✅ OpenOrca 4.2M downloaded<br/>✅ Alpaca-GPT4 52K downloaded<br/>✅ WizardLM 143K downloaded<br/>✅ Dolly 15K downloaded<br/>✅ ShareGPT 90K downloaded<br/>✅ All files in `data/raw/` | 10-15 min | $0 |

| **TOTAL MVP** | **14 weeks** | **$1,717** | **6% Done** || **1.2** | Quality Scoring & Selection | ✅ English filter applied (100% English)<br/>✅ Top 350K OpenOrca selected<br/>✅ All 52K Alpaca selected<br/>✅ Top 80K WizardLM selected<br/>✅ All 15K Dolly selected<br/>✅ Top 50K ShareGPT selected<br/>✅ Total ~547K samples in `data/phase1/scored/` | 30-45 min | $0 |

| **1.3** | Deduplication | ✅ MinHash LSH @ 0.8 threshold applied<br/>✅ ~547K → ~500K unique samples<br/>✅ Output: `data/phase1/public_500k_filtered.jsonl` | 20-30 min | $0 |

---| **1.4** | Format Validation | ✅ All samples have instruction-response pairs<br/>✅ JSON schema valid<br/>✅ 100% English confirmed<br/>✅ Metadata integrity verified<br/>✅ Ready for training | 1 hr | $0 |



## ✅ PHASE 0: DATASET CREATION (COMPLETE)**Phase 1 Total:** 1.5-2 hrs | **$0** | **Success:** 500K validated English training samples



### Summary---

- ✅ 640K curated examples via multi-teacher distillation

- ✅ Advanced MinHash LSH deduplication (Jaccard 0.8)## **PHASE 2: BASE MODEL TRAINING** ❌ **BLOCKED BY PHASE 1 (Week 3-4)**

- ✅ Quality filtering via GPT-4-mini scoring (>7/10)

- ✅ Format standardization to instruction-response pairs| Step | Task | Success Criteria | Duration | Cost |

- ✅ Location: `/data/phase1/public_500k_filtered.jsonl`|------|------|------------------|----------|------|

| **2.1** | Setup Training Pipeline | ✅ PEFT, bitsandbytes, accelerate installed<br/>✅ Qwen 2.5 7B downloaded from HuggingFace<br/>✅ QLoRA 4-bit configured (rank 64, alpha 16)<br/>✅ Target modules set (q_proj, v_proj, k_proj, o_proj)<br/>✅ Training config validated | 3 hrs | $0 |

**No further action needed - proceed to Phase 1**| **2.2** | Train Base Model | ✅ Training completed on 500K samples<br/>✅ Loss converged (validation loss stable)<br/>✅ Best checkpoint saved (11GB)<br/>✅ Output: `models/qwen-2.5-7b-distilled/`<br/>✅ No training errors | 36-48 hrs (GPU)<br/>5-7 days (CPU) | $120-180<br/>(cloud)<br/>$0 (local) |

| **2.3** | Checkpoint Management | ✅ Best model checkpoint saved<br/>✅ 3-5 checkpoints retained<br/>✅ Intermediate checkpoints deleted<br/>✅ Disk space optimized | Auto | $3 |

---| **2.4** | Validate Base Model | ✅ **MMLU: 78-82%** (GPT-4: 86.4%)<br/>✅ **HumanEval: 58-62%** (GPT-4: 67.0%)<br/>✅ **BBH: 72-76%** (GPT-4: 83.1%)<br/>✅ **GSM8K: 86-88%** (GPT-4: 92.0%)<br/>✅ **Overall: 90-93% GPT-4**<br/>⚠️ If fails: iterate training config | 4 hrs | $0 |



## 🚀 PHASE 1: BASE MODEL TRAINING (4 Weeks, $505)**Phase 2 Total:** 36-48 hrs | **$123-183** | **Success:** 11GB model @ 90-93% GPT-4



### Week 0-1: Vocabulary Optimization & Setup---



#### Day 1: Environment Setup## **PHASE 3: BASE MODEL COMPRESSION** ❌ **BLOCKED BY PHASE 2 (Week 5)**

```bash

# 1. Install Axolotl**Compression Strategy:** Neural Magic SparseML + AWQ provides +1.5-2% better accuracy than magnitude pruning + INT8, with superior CPU performance through structured sparsity patterns optimized for M4 Pro and Apple Silicon. This approach yields 88-89% GPT-4 quality vs 87% with simpler methods, establishing a stronger foundation for modifier training at the same 480MB target size.

cd /Users/vivekdurairaj/Projects/Cogumi-LLM

git clone https://github.com/OpenAccess-AI-Collective/axolotl| Step | Task | Success Criteria | Duration | Cost |

cd axolotl|------|------|------------------|----------|------|

pip install -e .| **3.1** | Neural Magic Structured Pruning (60-65%) | ✅ SparseML 2:4 structured sparsity applied<br/>✅ 11GB → 3.85GB (60-65% pruned)<br/>✅ Quality: 88-90% GPT-4<br/>✅ CPU-optimized patterns (M4 Pro, Apple Silicon) | 5-6 hrs | $10-15 |

| **3.2** | AWQ Quantization (4-bit) | ✅ AutoAWQ activation-aware quantization<br/>✅ 3.85GB → 1.0GB<br/>✅ Quality: 88-89% GPT-4<br/>✅ Group-wise quantization (128 groups) | 2-3 hrs | $5-7 |

# 2. Install additional dependencies| **3.3** | Attention Compression (SVD) | ✅ PyTorch SVD rank-384 applied<br/>✅ 1.0GB → 850MB<br/>✅ Quality: 87-88% GPT-4<br/>✅ Attention mechanism intact | 2 hrs | $5 |

pip install sentence-transformers datasketch scikit-learn| **3.4** | GGUF Export (Q5_K_M) | ✅ llama.cpp conversion successful<br/>✅ 850MB → 510MB GGUF<br/>✅ Quality: 87-88% GPT-4 (no degradation)<br/>✅ GGUF format valid | 1 hr | $0 |

| **3.5** | Zstd Compression | ✅ Zstd compression applied<br/>✅ 510MB → 480MB final<br/>✅ Quality: 87-88% GPT-4<br/>✅ Lossless compression verified | 30 min | $0 |

# 3. Setup RunPod account| **3.6** | Validate Compressed Base | ✅ **Benchmarks: 87-88% GPT-4**<br/>✅ Ollama loads model successfully<br/>✅ Inference works (streaming functional)<br/>✅ M4 Pro: 40-50 tok/s<br/>✅ Memory: ≤2.0GB RAM<br/>⚠️ **DECISION:** Pass → Phase 4 \| Fail → Iterate Phase 3 | 4-6 hrs | $0 |

# - Go to runpod.io, create account

# - Add payment method**Phase 3 Total:** ~12 hrs | **$20-27** | **Success:** 480MB base @ 87-88% GPT-4, Ollama-compatible, CPU-optimized

# - Generate API key

# - Save to environment: export RUNPOD_API_KEY="your_key"---



# 4. Download LLAMA-3.2-8B base model## **PHASE 4: GENERAL REASONING MODIFIERS** ⚠️ **CRITICAL (Week 6)**

python scripts/download_llama.py --model meta-llama/Llama-3.2-8B --output models/llama-3.2-8b-base

```| Step | Task | Success Criteria | Duration | Cost |

|------|------|------------------|----------|------|

**Expected Time:** 4-6 hours  | **4.1** | Identify Failures | ✅ 50K diverse English tasks tested<br/>✅ ~5,500 failure cases identified<br/>✅ Failures documented with examples<br/>✅ Output: `data/phase4/failures_5500.jsonl` | 5 hrs | $0 |

**Cost:** $0  | **4.2** | Pattern Clustering (Llama 405B) | ✅ Groq Llama 405B Batch API used<br/>✅ 5,500 failures → 75 patterns<br/>✅ Patterns categorized by failure type<br/>✅ Output: `data/phase4/patterns_75.json` | 2 hrs | $11<br/>(batch) |

**Output:** Environment ready, base model downloaded (~16GB)| **4.3** | Train General Modifiers | ✅ 58.3K examples generated via cascaded teachers:<br/>   - 52K Llama 405B (89%)<br/>   - 4.5K ChatGPT-4 (8%)<br/>   - 1.8K ChatGPT-5 (3%)<br/>✅ 38 specialist LoRA adapters trained<br/>✅ 88MB modifier stack created<br/>✅ 568MB total (480MB base + 88MB)<br/>✅ **Target: 99-100% GPT-4 general**<br/>⚠️ If fails: iterate on data generation | ~40 hrs | $283-303<br/>(data+train) |



#### Day 2: Vocabulary Analysis**Phase 4 Total:** ~47 hrs | **$294-314** | **Success:** 568MB system @ 99-100% GPT-4 general

```bash

# Generate Claude 4.5 script for vocab analysis---

# Prompt: "Generate Python script to analyze token frequency in 10K English samples"

## **PHASE 5: CODING & AGENTIC MODIFIERS** ⚠️ **CRITICAL (Week 7-8)**

python src/phase1_base/vocab_analysis.py \

  --samples data/phase1/public_500k_filtered.jsonl \| Step | Task | Success Criteria | Duration | Cost |

  --num-samples 10000 \|------|------|------------------|----------|------|

  --output models/tokenizers/vocab_analysis.json| **5.1** | Identify Coding/Agentic Failures | ✅ HumanEval, MBPP, SWE-bench tested<br/>✅ Agentic workflows tested<br/>✅ ~2,800 failures identified (2,200 coding + 600 agentic)<br/>✅ Output: `data/phase5/failures_2800.jsonl` | 9 hrs | $0 |

| **5.2** | Train Coding Modifiers | ✅ 42 patterns identified (29 code + 13 agentic)<br/>✅ 24.5K examples generated via cascaded teachers:<br/>   - 17K Qwen3-Coder-480B (69%)<br/>   - 7.5K ChatGPT-5 (31%)<br/>✅ 23 specialist LoRA adapters trained (18 code + 5 agentic)<br/>✅ 80MB coding modifier stack created<br/>✅ 648MB total (480MB base + 88MB general + 80MB coding)<br/>✅ **Target: 108-113% GPT-4 coding**<br/>⚠️ If fails: iterate on data generation | ~35 hrs | $457-477<br/>(data+train) |

# Review analysis results| **5.3** | Validate Full System | ✅ **General: 99-100% GPT-4** (MMLU, BBH)<br/>✅ **Coding: 108-113% GPT-4** (HumanEval, MBPP)<br/>✅ **Agentic: Enhanced** (tool use, multi-step)<br/>✅ All benchmarks pass targets<br/>⚠️ **DECISION:** Pass → Phase 6 \| Fail → Iterate Phase 4-5 | 6-8 hrs | $0 |

cat models/tokenizers/vocab_analysis.json | jq '.top_25k_tokens | length'

```**Phase 5 Total:** ~51 hrs | **$457-477** | **Success:** 648MB system @ 99-100% general, 108-113% coding



**Expected Time:** 6 hours (including script generation)  ---

**Cost:** $0  

**Output:** Token frequency analysis, identify top 25K tokens## **PHASE 6: MODIFIER COMPRESSION** ❌ **BLOCKED BY PHASE 5 (Week 9)**



#### Day 3: Vocabulary Trimming| Step | Task | Success Criteria | Duration | Cost |

```bash|------|------|------------------|----------|------|

# Generate Claude 4.5 script for vocab trimming| **6.1** | Compress Modifiers | ✅ Zstd dictionary trained on modifier samples<br/>✅ 168MB modifiers → ~120-140MB compressed<br/>✅ Lossless compression verified<br/>✅ Quality maintained (99-100% general, 108-113% coding) | 16 min | $0 |

# Prompt: "Generate script to trim LLAMA tokenizer from 128K to 25K tokens with validation"| **6.2** | Package Final Model | ✅ 480MB base + ~120-140MB modifiers packaged<br/>✅ Total size: ~600-620MB<br/>✅ Metadata included<br/>✅ Dual-mode configuration embedded<br/>✅ Version info added | 15 min | $0 |



python src/phase1_base/vocab_trimming.py \**Phase 6 Total:** ~31 min | **$0** | **Success:** ~600-620MB packaged model ready for runtime

  --base-model models/llama-3.2-8b-base \

  --token-list models/tokenizers/vocab_analysis.json \---

  --target-size 25000 \

  --validation-samples 10000 \## **PHASE 7: DUAL-MODE RUNTIME SYSTEM** ❌ **BLOCKED BY PHASE 6 (Week 9)**

  --rollback-threshold 0.03 \

  --output models/tokenizers/trimmed_vocab| Step | Task | Success Criteria | Duration | Cost |

|------|------|------------------|----------|------|

# Validate trimmed vocabulary| **7.1** | Build Dual-Mode System | ✅ Mode selector implemented<br/>✅ **Speed Mode:** 45-55 tok/s @ 95-97% general, 100-105% coding<br/>✅ **Quality Mode:** 30-40 tok/s @ 99-100% general, 108-113% coding<br/>✅ Mode switching < 1 sec<br/>✅ Memory: ≤1.5GB (Speed), ≤2.5GB (Quality) | 8 hrs | $0 |

python src/phase1_base/validate_vocab.py \| **7.2** | Hardware Detection | ✅ CPU/GPU/Apple Silicon detection working<br/>✅ Auto-configuration for M4 Pro, Intel, AMD<br/>✅ Optimal settings applied per platform<br/>✅ Cross-platform compatibility verified | 3 hrs | $0 |

  --tokenizer models/tokenizers/trimmed_vocab \| **7.3** | Lazy Modifier Loader | ✅ On-demand modifier loading implemented<br/>✅ Task-based loading (general vs coding vs agentic)<br/>✅ Dynamic memory management working<br/>✅ Unused modifiers unloaded correctly | 4 hrs | $0 |

  --test-file data/phase1/public_500k_filtered.jsonl \| **7.4** | Benchmarking Suite | ✅ **Speed Mode validated:** 45-55 tok/s, 95-97% general, 100-105% coding<br/>✅ **Quality Mode validated:** 30-40 tok/s, 99-100% general, 108-113% coding<br/>✅ First token latency: <500ms<br/>✅ Memory within limits<br/>✅ All tests passing | 5 hrs | $0 |

  --num-samples 10000

```**Phase 7 Total:** 20 hrs | **$0** | **Success:** Dual-mode runtime validated, both modes meet targets



**Expected Time:** 8 hours  ---

**Cost:** $0  

**Output:** Trimmed vocabulary, validation report (~3.4GB savings)## **PHASE 8: OLLAMA INTEGRATION & CLI** ❌ **BLOCKED BY PHASE 7 (Week 10)**



**Validation Checks:**| Step | Task | Success Criteria | Duration | Cost |

- ✅ Tokenization coverage: 99.5%+ on validation set|------|------|------------------|----------|------|

- ✅ Perplexity increase: <3%| **8.1** | Ollama Integration | ✅ Modelfile created (dual-mode config)<br/>✅ Model loads in Ollama successfully<br/>✅ Both modes functional in Ollama<br/>✅ Streaming works correctly<br/>✅ Mode switching works in Ollama | 4 hrs | $0 |

- ✅ Round-trip accuracy: 99.9%+| **8.2** | CLI Development | ✅ `cogumi` command installed<br/>✅ Chat interface working (Rich UI)<br/>✅ Commands functional: /help, /mode, /save, /load, /stats, /exit<br/>✅ Streaming response handler working<br/>✅ Session persistence working<br/>✅ Mode indicator in UI | 8 hrs | $0 |

| **8.3** | Installation Script | ✅ `install.sh` created<br/>✅ OS detection working (macOS, Linux, Windows WSL)<br/>✅ Auto-installs Ollama if missing<br/>✅ Downloads model (~600-620MB)<br/>✅ Configures CLI tools<br/>✅ Verifies dual-mode functionality<br/>✅ One-command install works | 4 hrs | $0 |

### Week 1-2.5: Base Model Training (Phase 1A)

**Phase 8 Total:** 16 hrs | **$0** | **Success:** Full installation pipeline working, dual-mode CLI functional

#### Week 1, Day 1-2: Axolotl Configuration

```bash---

# Generate Claude 4.5 script for Axolotl config

# Prompt: "Generate Axolotl config for QLoRA training on LLAMA-3.2-8B with 640,637 English examples"## **PHASE 9: TESTING, DOCUMENTATION & RELEASE** ❌ **BLOCKED BY PHASE 8 (Week 11-12)**



# Edit configs/base_training.yaml| Step | Task | Success Criteria | Duration | Cost |

cat > configs/base_training.yaml << 'EOF'|------|------|------------------|----------|------|

base_model: models/llama-3.2-8b-base| **9.1** | Installation Testing | ✅ macOS (Apple Silicon + Intel): >95% success<br/>✅ Linux (Ubuntu, Debian, Fedora, Arch): >95% success<br/>✅ Windows WSL2: >95% success<br/>✅ Offline installation: works<br/>✅ Both modes functional across platforms | 1 week | $0 |

model_type: LlamaForCausalLM| **9.2** | Functional Testing | ✅ All commands work in both modes<br/>✅ Mode switching: <1 sec, no errors<br/>✅ Streaming: no gaps or buffering issues<br/>✅ Error handling: graceful recovery<br/>✅ Session save/load: data persists correctly<br/>✅ Edge cases: handled properly | 3-4 days | $0 |

tokenizer_type: AutoTokenizer| **9.3** | Performance Testing | ✅ **Speed Mode:** 45-55 tok/s, 95-97% general, 100-105% coding, ≤1.5GB RAM<br/>✅ **Quality Mode:** 30-40 tok/s, 99-100% general, 108-113% coding, ≤2.5GB RAM<br/>✅ First token: <500ms<br/>✅ Mode switch: <1 sec<br/>✅ Tested on M4 Pro, Intel, AMD | 2-3 days | $0 |

tokenizer_config: models/tokenizers/trimmed_vocab| **9.4** | User Acceptance Testing | ✅ 10-20 beta testers recruited<br/>✅ Feedback collected on both modes<br/>✅ Installation usability: positive<br/>✅ Mode switching: intuitive<br/>✅ Response quality: meets expectations<br/>✅ Bugs: identified and fixed<br/>✅ Iteration complete | 1 week | $0 |

| **9.5** | Documentation | ✅ README.md (overview, dual-mode, quick start)<br/>✅ INSTALL.md (detailed installation)<br/>✅ USAGE.md (commands, mode selection)<br/>✅ FAQ.md (mode comparison, when to use each)<br/>✅ TROUBLESHOOTING.md (common issues)<br/>✅ API.md (programmatic mode selection)<br/>✅ ARCHITECTURE.md (dual-mode design)<br/>✅ BENCHMARKS.md (performance comparison) | 1 week | $0 |

load_in_4bit: true| **9.6** | Package & Release | ✅ GitHub release v1.0.0 created<br/>✅ Changelog includes dual-mode features<br/>✅ Artifacts uploaded: cogumi-llm-full.gguf (~600-620MB), install.sh, Modelfile<br/>✅ PyPI package published (cogumi-llm CLI)<br/>✅ Homebrew formula created<br/>✅ APT/AUR packages created<br/>✅ Public announcement ready | 3-4 days | $0 |

adapter: lora

lora_r: 64**Phase 9 Total:** 3-4 weeks | **$0** | **Success:** Public release, >95% installation success, positive user feedback

lora_alpha: 128

lora_dropout: 0.05---

lora_target_modules:

  - q_proj## **COMPLETE EXECUTION SUMMARY**

  - k_proj

  - v_proj| Phase | Duration | Cost | Success Criteria | Status |

  - o_proj|-------|----------|------|-------------------|--------|

  - gate_proj| **Phase 0:** Setup | 6-8 hrs | $0 | Dependencies verified, deduplication working | ✅ COMPLETE |

  - up_proj| **Phase 1:** Dataset Prep | 1.5-2 hrs | $0 | 500K validated English samples | ⏸️ NEXT |

  - down_proj| **Phase 2:** Base Training | 36-48 hrs | $123-183 | 11GB model @ 90-93% GPT-4 | ❌ BLOCKED |

| **Phase 3:** Base Compression | ~12 hrs | $22 | 480MB base @ 86-88% GPT-4, Ollama-compatible | ❌ BLOCKED |

sequence_len: 2048| **Phase 4:** General Modifiers | ~47 hrs | $294-314 | 568MB @ 99-100% GPT-4 general | ❌ BLOCKED |

sample_packing: true| **Phase 5:** Coding Modifiers | ~51 hrs | $457-477 | 648MB @ 99-100% general, 108-113% coding | ❌ BLOCKED |

| **Phase 6:** Modifier Compression | ~31 min | $0 | ~600-620MB packaged, quality maintained | ❌ BLOCKED |

datasets:| **Phase 7:** Dual-Mode Runtime | 20 hrs | $0 | Both modes validated, targets met | ❌ BLOCKED |

  - path: data/phase1/public_500k_filtered.jsonl| **Phase 8:** Ollama & CLI | 16 hrs | $0 | Installation pipeline working | ❌ BLOCKED |

    type: completion| **Phase 9:** Testing & Release | 3-4 weeks | $0 | Public release, >95% success rate | ❌ BLOCKED |

| **TOTAL** | **10-12 weeks** | **$896-996** | **99-100% general, 108-113% coding, dual-mode ready** | **6% COMPLETE** |

optimizer: adamw_torch

lr_scheduler: cosine---

learning_rate: 5e-6

warmup_steps: 500## **CRITICAL SUCCESS VALIDATION PROTOCOL**

num_epochs: 3

gradient_accumulation_steps: 8### **🚨 ENFORCEMENT RULE: NO PHASE ADVANCE WITHOUT SUCCESS CRITERIA MET**

micro_batch_size: 4

gradient_checkpointing: true**For every task:**

1. ✅ **Execute task**

early_stopping_patience: 62. ✅ **Validate ALL success criteria** (use checklist above)

3. ✅ **Document validation results**

evaluation_strategy: steps4. ⚠️ **If ANY criterion fails → STOP and iterate**

eval_steps: 5005. ✅ **Only proceed to next task when ALL criteria pass**

save_steps: 1000

save_total_limit: 5**Validation Log Format:**

```

bf16: trueTask X.Y: [Task Name]

output_dir: data/checkpoints/phase1a_base✅ Criterion 1: [Result]

EOF✅ Criterion 2: [Result]

❌ Criterion 3: [Result] → FAILED

# Validate configuration→ ITERATION REQUIRED: [Corrective action]

axolotl validate configs/base_training.yaml```

```

**Phase Gates (Critical Checkpoints):**

**Expected Time:** 4 hours  - **Phase 1 → 2:** 500K validated samples ready

**Cost:** $0  - **Phase 2 → 3:** 11GB model achieves 90-93% GPT-4

**Output:** Validated Axolotl config- **Phase 3 → 4:** 480MB base achieves 86-88% GPT-4, Ollama-compatible

- **Phase 4 → 5:** General modifiers achieve 99-100% GPT-4 general

#### Week 1, Day 3 - Week 2.5: Training Execution- **Phase 5 → 6:** Coding modifiers achieve 108-113% GPT-4 coding

```bash- **Phase 6 → 7:** Compressed modifiers maintain quality

# 1. Launch RunPod instance- **Phase 7 → 8:** Both modes validated and meet performance targets

# - Go to runpod.io- **Phase 8 → 9:** Installation pipeline functional

# - Select "A100 40GB" GPU- **Phase 9 → Release:** All testing criteria met, documentation complete

# - Use PyTorch template

# - Configure: 200GB storage, Jupyter/SSH access---

# Cost: $1.89/hour

## **FUTURE ENHANCEMENT: INTELLIGENT FALLBACK LLM ROUTING (POST-MVP)**

# 2. Upload code and data to RunPod

rsync -avz --progress \### **Overview**

  /Users/vivekdurairaj/Projects/Cogumi-LLM/ \A planned enhancement to implement a runtime fallback system that intelligently selects different teacher LLMs (GPT-5, Claude 4.5, others) based on query type, enabling continuous improvement through failure-driven learning.

  runpod@<instance-ip>:/workspace/

### **Key Components**

# 3. SSH into RunPod and start training

ssh runpod@<instance-ip>| Component | Description | Success Criteria |

cd /workspace/Cogumi-LLM|-----------|-------------|------------------|

| **Architecture Design** | Define routing system architecture, schema for rules, integration with distillation pipeline | Architecture documented, routing logic designed |

# 4. Start training with monitoring| **Configuration System** | Implement YAML/JSON config for routing rules (e.g., GPT-5 for general, Claude 4.5 for coding) | Config system functional, rules can be defined/loaded |

axolotl train configs/base_training.yaml \| **Dynamic Model Selection** | Build query analyzer and model selector that routes to optimal teacher LLM | >95% routing accuracy, <50ms overhead |

  --wandb-project cogumi-llm \| **Logging & Monitoring** | Log all fallback cases with metadata, store responses for LoRA training, track costs per model | All fallback cases logged, responses stored |

  --wandb-run-name phase1a-base-training| **Extensible Framework** | Plugin pattern for adding new teacher LLMs without core code changes | New models can be added via adapters |

| **Distillation Integration** | Connect fallback logs to automatic LoRA adapter training pipeline | Fallback data automatically trains new adapters |

# 5. Monitor progress| **Testing & Validation** | Validate routing accuracy, cost tracking, performance impact, adapter training | System meets all criteria, positive feedback |

# - WandB dashboard: https://wandb.ai/your-project/cogumi-llm

# - Watch for validation loss plateau (early stopping)### **Benefits**

# - Expected: 28,000 steps over ~2.5 weeks- **Continuous Learning:** Automatically improve model from production failures

- **Cost Optimization:** Route to cheapest capable model for each query type

# 6. Training will auto-stop when validation loss plateaus for 3K steps- **Quality Improvement:** Use best-in-class teacher for each domain (coding, math, reasoning)

```- **Flexibility:** Easy to add new models and routing strategies

- **Data Collection:** Build high-quality training data from real user queries

**Expected Time:** 2.5 weeks (120 GPU-hours @ $1.89/hr)  

**Cost:** $220  ### **Implementation Timeline**

**Output:** Base model checkpoint in `data/checkpoints/phase1a_base/best/`- **Phase 10:** Post-v1.0 release (estimated 4-6 weeks after Phase 9)

- **Cost:** Variable (depends on fallback API usage)

**Monitoring Checklist:**- **Priority:** Medium (enhances existing system, not core MVP)

- [ ] WandB dashboard shows decreasing training loss

- [ ] Validation loss improves for first ~20K steps---

- [ ] No OOM errors (if OOM, reduce batch size)

- [ ] GPU utilization 85-95%**Document Version:** 1.0  

- [ ] Early stopping triggers around step 25K-28K**Last Updated:** January 2025  

**Status:** Phase 1 Ready to Start  

#### Week 2.5, Final Days: Merge & Validate**Next Milestone:** 500K dataset prepared and validated

```bash
# 1. Download checkpoint from RunPod
rsync -avz --progress \
  runpod@<instance-ip>:/workspace/Cogumi-LLM/data/checkpoints/phase1a_base/best/ \
  data/checkpoints/phase1a_base/

# 2. Merge LoRA weights into base model
python src/phase1_base/merge_lora.py \
  --base models/llama-3.2-8b-base \
  --adapter data/checkpoints/phase1a_base/best \
  --output models/phase1a_merged_10gb

# 3. Run validation benchmarks
python src/phase1_base/validate_base.py \
  --model models/phase1a_merged_10gb \
  --benchmarks mmlu humaneval gsm8k \
  --output results/phase1a_validation.json

# Expected results:
# MMLU: 60-66% (target: >60%)
# HumanEval: 45-53% (target: >45%)
# GSM8K: 55-62% (target: >55%)
```

**Expected Time:** 2 days  
**Cost:** $0  
**Output:** Merged 10GB base model, validation results

**Go/No-Go Decision:**
- ✅ If all benchmarks meet targets → Proceed to Phase 1B
- ❌ If any benchmark <target → Debug (check data quality, hyperparameters, training logs)

### Week 3-4: Failure Analysis & GPT-5 Distillation

#### Week 3, Day 1-2: Comprehensive Testing (Phase 1B)
```bash
# Generate Claude 4.5 script for comprehensive testing
# Prompt: "Generate script to test model on 50K diverse examples and identify failures"

# 1. Prepare test sets
python src/phase1_base/prepare_test_sets.py \
  --output data/test/comprehensive_50k.jsonl

# Test sets include:
# - Code: HumanEval, MBPP, LiveCodeBench (12K)
# - Reasoning: MMLU, BBH, ARC (15K)
# - Math: GSM8K, MATH (8K)
# - Automation: Tool-use scenarios (6K)
# - Others: Science, creative, etc. (9K)

# 2. Run testing
python src/phase1_base/test_base.py \
  --model models/phase1a_merged_10gb \
  --test-file data/test/comprehensive_50k.jsonl \
  --output results/phase1b_failures.jsonl \
  --batch-size 16 \
  --temperature 0.1

# This will take ~24 hours to process 50K examples
```

**Expected Time:** 2 days  
**Cost:** $0  
**Output:** 12-14K failures identified

#### Week 3, Day 3-5: Failure Clustering
```bash
# Generate Claude 4.5 script for failure clustering
# Prompt: "Generate script to cluster failures using Sentence-BERT + KMeans"

# 1. Embed failures
python src/phase1_base/embed_failures.py \
  --failures results/phase1b_failures.jsonl \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --output results/failure_embeddings.npy

# 2. Cluster failures
python src/phase1_base/cluster_failures.py \
  --embeddings results/failure_embeddings.npy \
  --k 10 \
  --output results/failure_clusters.json

# 3. Auto-label clusters with GPT-4-mini
python src/phase1_base/label_clusters.py \
  --clusters results/failure_clusters.json \
  --failures results/phase1b_failures.jsonl \
  --samples-per-cluster 20 \
  --output results/failure_patterns.json

# Cost: $0.30 for GPT-4-mini labeling
```

**Expected Time:** 3 days  
**Cost:** $5 (compute + GPT-4-mini)  
**Output:** 8-12 labeled failure patterns

#### Week 3-4: GPT-5 Targeted Data Generation (Phase 1C)

**Strategy:** Use GitHub Copilot to generate failure-aware examples at zero API cost
- Claude Sonnet 4.5 for code examples
- GPT-5 for reasoning/math/general examples
- Bidirectional training (doubles dataset to 80K)

##### Step 1: Split Failures by Domain (Day 1)
```bash
# Analyze failure clusters and split by code vs non-code
python src/phase1_base/split_failures.py \
  --patterns results/failure_patterns.json \
  --output-code data/phase1c/code_failures.jsonl \
  --output-other data/phase1c/other_failures.jsonl

# Expected split:
# Code failures: ~30% (4K failures) → 12K examples needed
# Other failures: ~70% (10K failures) → 28K examples needed
```

##### Step 2: Generate Code Examples via Copilot (Claude 4.5) (Day 2-3)
```bash
# Use GitHub Copilot Chat with Claude Sonnet 4.5
# Interactive prompt for EACH code failure cluster:

"""
You are generating training examples to fix specific code weaknesses.

**Failure Pattern:** {cluster_label}
**Failed Examples:**
{show 5 actual code failures from cluster}

**Task:** Generate {n} NEW code examples that:
1. Target this EXACT failure pattern
2. Are HARDER than the failed examples (increase complexity)
3. Include edge cases (null checks, boundary conditions, type mismatches)
4. Test deep algorithmic understanding, not surface patterns
5. Show complete working solutions with explanations

**Output Format:**
```json
{
  "instruction": "[challenging coding problem targeting this weakness]",
  "response": "[complete solution with step-by-step explanation]"
}
```

Generate examples one by one. After each, self-critique:
- Is it harder than the failures shown?
- Does it truly test the failure pattern?
- Are there more edge cases to add?
"""

# Process Copilot output → save to data/phase1c/code_12k.jsonl
# Cost: $0 (Copilot subscription)
```

**Expected Time:** 2 days (interactive with Copilot)
**Cost:** $0 (included in Copilot subscription)
**Output:** 12K code-focused examples

##### Step 3: Generate Other Examples via Copilot (GPT-5) (Day 4-5)
```bash
# Use GitHub Copilot Chat with GPT-5
# Interactive prompt for EACH non-code failure cluster:

"""
You are generating training examples to fix specific reasoning weaknesses.

**Failure Pattern:** {cluster_label}
**Failed Examples:**
{show 5 actual failures from cluster}

**Task:** Generate {n} NEW examples that:
1. Target this EXACT failure pattern
2. Are HARDER than the failed examples (increase difficulty)
3. Include corner cases and multi-step reasoning
4. Test deep understanding, not memorization
5. Show detailed step-by-step solutions

**Domain:** {math/reasoning/comprehension/etc based on cluster}

**Output Format:**
```json
{
  "instruction": "[challenging question targeting this weakness]",
  "response": "[detailed step-by-step solution with reasoning]"
}
```

Generate examples one by one. After each, self-critique:
- Is it significantly harder than the failures?
- Does it require multi-step reasoning?
- Are there ambiguities or tricks that test understanding?
"""

# Process Copilot output → save to data/phase1c/other_28k.jsonl
# Cost: $0 (Copilot subscription)
```

**Expected Time:** 2 days (interactive with Copilot)
**Cost:** $0 (included in Copilot subscription)
**Output:** 28K reasoning/math/other examples

##### Step 4: Create Bidirectional Training Data (Day 6)
```bash
# Generate reverse examples for bidirectional training
python src/phase1_base/create_bidirectional.py \
  --input data/phase1c/code_12k.jsonl data/phase1c/other_28k.jsonl \
  --output data/phase1c/bidirectional_80k.jsonl

# For each example:
# Forward: instruction → response
# Reverse: "Given this answer: {response}\n\nWhat was the question?" → instruction

# Benefits:
# - Doubles dataset to 80K examples
# - Improves reasoning in both directions
# - Better understanding of causality
```

**Expected Time:** 1 day
**Cost:** $0
**Output:** 80K bidirectional examples (40K forward + 40K reverse)

##### Step 5: Distillation Training (Day 7, ~4-5 hours)
```bash
# 1. Re-tokenize with bidirectional data
python Phase1A_2_0/scripts/pretokenize_dataset.py \
  --input data/phase1c/bidirectional_80k.jsonl \
  --output /tmp/tokenized_phase1c \
  --max_length 2048

# 2. Train Phase 1C adapter
python Phase1A_2_0/scripts/train_phase1a_optimized_h100.py \
  --model_name "models/phase1a_merged_10gb" \
  --use_pretokenized \
  --pretokenized_path "/tmp/tokenized_phase1c" \
  --torch_compile \
  --output_dir "data/checkpoints/phase1c_distilled" \
  --logging_dir "data/logs/phase1c" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --learning_rate 3e-6 \
  --dataloader_num_workers 4 \
  --save_steps 1000 \
  --logging_steps 10 \
  --save_total_limit 2

# Training time calculation:
# 80K examples / 8 batch size * 3 epochs = 30K steps
# 30K steps * 0.49s/step = 14,700s = 4.1 hours

# 3. Merge adapter to base
python src/phase1_base/merge_lora.py \
  --base models/phase1a_merged_10gb \
  --adapter data/checkpoints/phase1c_distilled/best \
  --output models/phase1c_enhanced_10gb

# 4. Validate enhanced base
python src/phase1_base/validate_base.py \
  --model models/phase1c_enhanced_10gb \
  --benchmarks mmlu humaneval gsm8k \
  --output results/phase1c_validation.json

# Expected results:
# MMLU: 70-80% (target: >70%, 88-100% GPT-4)
# HumanEval: 55-65% (target: >55%, 85-100% GPT-4)
# GSM8K: 65-75% (target: >65%, 87-100% GPT-4)
```

**Expected Time:** 4-5 hours (H100 SXM training)
**Cost:** $12.50 (5 hours × $2.50/hr)
**Output:** Enhanced 10GB base model (88-100% GPT-4 baseline)

**Phase 1C Summary:**
- ✅ 12K code examples via Copilot (Claude 4.5) - $0
- ✅ 28K other examples via Copilot (GPT-5) - $0
- ✅ 80K bidirectional examples total
- ✅ Training: 4-5 hours, $12.50
- ✅ **Total Phase 1C cost: $12.50** (vs $285 originally planned!)

**Phase 1 Complete Checklist:**
- ✅ Base model trained: 75-82% GPT-4 (Phase 1A)
- ✅ Failures identified and clustered: 12-14K in 8-12 categories (Phase 1B)
- ✅ 40K targeted examples generated via Copilot at $0 cost (Phase 1C)
- ✅ 80K bidirectional training data created (Phase 1C)
- ✅ Enhanced model trained: 88-100% GPT-4 (Phase 1C)
- ✅ All validation benchmarks met

**Phase 1 Total: 4 weeks, $517.50** ($505 Phase 1A + $12.50 Phase 1C)

---

## 🗜️ PHASE 2: EXTREME COMPRESSION (6 Weeks, $402)

### Week 5-6: Neural Magic Structured Pruning (Phase 2A)

#### Week 5, Day 1-3: Setup & Grid Search
```bash
# 1. Install Neural Magic
pip install llm-compressor

# 2. Generate Claude 4.5 script for pruning
# Prompt: "Generate script for Neural Magic structured pruning with grid search"

# 3. Prepare calibration data
python src/phase2_compression/prepare_calibration.py \
  --dataset data/phase1/public_500k_filtered.jsonl \
  --num-samples 10000 \
  --output data/calibration/diverse_10k.jsonl

# 4. Run grid search
python src/phase2_compression/pruning_grid_search.py \
  --model models/phase1c_enhanced_10gb \
  --sparsity-levels 0.60 0.65 0.70 \
  --calibration data/calibration/diverse_10k.jsonl \
  --output models/pruning_experiments/

# This runs 3 experiments in parallel, takes 2-3 days
```

**Expected Time:** 3 days  
**Cost:** $60 (RunPod GPU time)  
**Output:** 3 pruned models at different sparsity levels

#### Week 5, Day 4 - Week 6, Day 3: Gradual Pruning & Fine-Tuning
```bash
# 1. Select best sparsity level from grid search
python src/phase2_compression/select_best_sparsity.py \
  --experiments models/pruning_experiments/ \
  --metrics results/pruning_metrics.json \
  --output configs/optimal_sparsity.json

# 2. Run gradual pruning (2K steps over 3 days)
python src/phase2_compression/gradual_pruning.py \
  --model models/phase1c_enhanced_10gb \
  --config configs/optimal_sparsity.json \
  --schedule 0,0.1625,0.325,0.4875,0.65 \
  --steps 0,500,1000,1500,2000 \
  --calibration data/calibration/diverse_10k.jsonl \
  --output models/phase2a_sparse_3.5gb

# 3. Post-pruning fine-tuning (8 hours)
python src/phase2_compression/recovery_finetune.py \
  --model models/phase2a_sparse_3.5gb \
  --data data/calibration/diverse_10k.jsonl \
  --learning-rate 1e-6 \
  --hours 8 \
  --output models/phase2a_recovered_3.5gb

# 4. Validate
python src/phase2_compression/validate_pruning.py \
  --model models/phase2a_recovered_3.5gb \
  --benchmarks mmlu humaneval gsm8k \
  --output results/phase2a_validation.json

# Expected quality loss: -2 to -4% from Phase 1C
```

**Expected Time:** 7 days  
**Cost:** $120 (RunPod GPU time)  
**Output:** 3.5GB sparse model

### Week 7: AWQ Quantization (Phase 2B)

#### Week 7, Day 1-4: Quantization
```bash
# Generate Claude 4.5 script for AWQ quantization
# Prompt: "Generate script for AWQ 4-bit quantization with mixed-precision"

# 1. Prepare calibration samples
python src/phase2_compression/prepare_awq_calibration.py \
  --dataset data/phase1/public_500k_filtered.jsonl \
  --num-samples 2048 \
  --output data/calibration/awq_2k.jsonl

# 2. Run AWQ quantization
python src/phase2_compression/awq_quantize.py \
  --model models/phase2a_recovered_3.5gb \
  --calibration data/calibration/awq_2k.jsonl \
  --group-size 128 \
  --bits 4 \
  --output models/phase2b_quantized_900mb

# 3. Validate
python src/phase2_compression/validate_quantization.py \
  --model models/phase2b_quantized_900mb \
  --original models/phase2a_recovered_3.5gb \
  --benchmarks mmlu humaneval gsm8k \
  --output results/phase2b_validation.json

# Expected quality loss: -2 to -3% from Phase 2A
```

**Expected Time:** 4 days  
**Cost:** $90 (RunPod GPU time)  
**Output:** 900MB quantized model

### Week 8: GGUF Export & Zstd Compression (Phase 2C-D)

#### Week 8, Day 1-3: GGUF Export
```bash
# Generate Claude 4.5 script for GGUF conversion
# Prompt: "Generate script to convert PyTorch model to GGUF Q5_K_M format"

# 1. Convert to GGUF
python src/phase2_compression/convert_to_gguf.py \
  --model models/phase2b_quantized_900mb \
  --variant Q5_K_M \
  --output models/phase2c_gguf_600mb.bin

# 2. Validate GGUF
python src/phase2_compression/validate_gguf.py \
  --gguf models/phase2c_gguf_600mb.bin \
  --pytorch models/phase2b_quantized_900mb \
  --num-queries 100 \
  --token-agreement-threshold 0.95 \
  --output results/phase2c_validation.json
```

**Expected Time:** 3 days  
**Cost:** $0  
**Output:** 600MB GGUF model

#### Week 8, Day 4-5: Zstd Compression
```bash
# Generate Claude 4.5 script for Zstd compression
# Prompt: "Generate script for Zstd compression with dictionary training"

# 1. Train dictionary
python src/phase2_compression/train_zstd_dict.py \
  --model models/phase2c_gguf_600mb.bin \
  --sample-size 100MB \
  --dict-size 128KB \
  --output models/zstd_dict.bin

# 2. Compress
python src/phase2_compression/compress_with_zstd.py \
  --model models/phase2c_gguf_600mb.bin \
  --dictionary models/zstd_dict.bin \
  --level 10 \
  --output models/phase2d_compressed_500mb.bin.zst

# 3. Validate lossless compression
python src/phase2_compression/validate_compression.py \
  --compressed models/phase2d_compressed_500mb.bin.zst \
  --original models/phase2c_gguf_600mb.bin \
  --output results/phase2d_validation.json

# Should output: SHA-256 checksums match ✅
```

**Expected Time:** 2 days  
**Cost:** $0  
**Output:** 500MB compressed model

### Week 9-10: Recovery & Calibration (Phase 2E-F)

#### Week 9: Recovery Fine-Tuning
```bash
# Generate Claude 4.5 script for recovery fine-tuning
# Prompt: "Generate script to select hardest examples and enhance with GPT-5"

# 1. Measure perplexity on full dataset
python src/phase2_compression/measure_perplexity.py \
  --model models/phase2d_compressed_500mb.bin.zst \
  --dataset data/phase1/public_500k_filtered.jsonl \
  --output results/perplexity_scores.json

# 2. Select hardest 12K examples (top 2%)
python src/phase2_compression/select_hardest.py \
  --perplexity results/perplexity_scores.json \
  --top-percent 2 \
  --output data/phase2e/hardest_12k.jsonl

# 3. Enhance with GPT-5
python src/phase2_compression/enhance_with_gpt5.py \
  --examples data/phase2e/hardest_12k.jsonl \
  --model gpt-5 \
  --output data/phase2e/enhanced_12k.jsonl

# Cost: $70 for GPT-5

# 4. Conservative LoRA fine-tuning
python src/phase2_compression/recovery_lora.py \
  --base models/phase2d_compressed_500mb.bin.zst \
  --data data/phase2e/enhanced_12k.jsonl \
  --learning-rate 8e-7 \
  --epochs 2 \
  --rank 64 \
  --output models/phase2e_recovered_520mb

# 5. Validate recovery
python src/phase2_compression/validate_recovery.py \
  --model models/phase2e_recovered_520mb \
  --original models/phase1c_enhanced_10gb \
  --benchmarks mmlu humaneval gsm8k \
  --output results/phase2e_validation.json

# Expected: Recover +1-2% quality
```

**Expected Time:** 4 days  
**Cost:** $70 (GPT-5)  
**Output:** 520MB recovered base model

#### Week 10: Confidence Calibration
```bash
# Generate Claude 4.5 script for confidence calibration
# Prompt: "Generate script for temperature + Platt scaling calibration"

# 1. Generate calibration dataset
python src/phase2_compression/generate_calibration_data.py \
  --model models/phase2e_recovered_520mb \
  --num-queries 30000 \
  --collect-logits \
  --output data/phase2f/calibration_30k.jsonl

# 2. Label with quality scores
python src/phase2_compression/label_quality.py \
  --dataset data/phase2f/calibration_30k.jsonl \
  --scorer gpt-4-mini \
  --output data/phase2f/labeled_30k.jsonl

# Cost: $35 for GPT-4-mini

# 3. Train calibrators
python src/phase2_compression/train_calibration.py \
  --dataset data/phase2f/labeled_30k.jsonl \
  --methods temperature platt \
  --output models/calibrators/

# 4. Validate calibration
python src/phase2_compression/validate_calibration.py \
  --model models/phase2e_recovered_520mb \
  --calibrators models/calibrators/ \
  --test-set 5000 \
  --output results/phase2f_validation.json

# Expected: ECE <0.05, 97% routing accuracy
```

**Expected Time:** 3 days  
**Cost:** $35 (GPT-4-mini)  
**Output:** Calibrated confidence scores, ready for routing

**Phase 2 Complete Checklist:**
- ✅ Pruned to 3.5GB (65% sparsity)
- ✅ Quantized to 900MB (4-bit AWQ)
- ✅ Exported to 600MB (GGUF Q5_K_M)
- ✅ Compressed to 500MB (Zstd)
- ✅ Recovered to 520MB (+1-2% quality)
- ✅ Calibrated for accurate routing
- ✅ Final: 520MB base, 89-91% GPT-4

**Phase 2 Total: 6 weeks, $402**

---

## 🎨 PHASE 3: DOMAIN MODIFIERS (4 Weeks, $685)

**Note:** All three modifiers (Code, Reasoning, Automation) follow the same process. Below is the detailed plan for one modifier; repeat for others in parallel or sequence.

### General Modifier Pipeline (Reusable for All Domains)

#### Step 1: Test Base on Domain (2 days)
```bash
# Example for Code Modifier
python src/phase3_modifiers/test_domain.py \
  --model models/phase2e_recovered_520mb \
  --domain code \
  --benchmarks humaneval mbpp livecodebench \
  --num-tasks 12000 \
  --output results/phase3_code_failures.jsonl
```

#### Step 2: Tier 1 Generation (3 days, $65-70)
```bash
python src/phase3_modifiers/generate_tier1.py \
  --failures results/phase3_code_failures.jsonl \
  --teacher qwen-coder-480b \
  --num-examples 9000 \
  --output data/phase3_code/tier1_raw.jsonl

python src/phase3_modifiers/score_tier1.py \
  --examples data/phase3_code/tier1_raw.jsonl \
  --scorer gpt-4-mini \
  --threshold 7.0 \
  --output data/phase3_code/tier1_filtered.jsonl
```

#### Step 3: Test Tier 1 (4 hours)
```bash
python src/phase3_modifiers/test_tier1.py \
  --base-model models/phase2e_recovered_520mb \
  --tier1-data data/phase3_code/tier1_filtered.jsonl \
  --failures results/phase3_code_failures.jsonl \
  --output results/phase3_code_tier1_remaining.jsonl
```

#### Step 4: Tier 2 Generation (2 days, $50-55)
```bash
python src/phase3_modifiers/generate_tier2.py \
  --failures results/phase3_code_tier1_remaining.jsonl \
  --teacher deepseek-coder \
  --output data/phase3_code/tier2_filtered.jsonl
```

#### Step 5: Test Tier 2 (2 hours)
```bash
python src/phase3_modifiers/test_tier2.py \
  --base-model models/phase2e_recovered_520mb \
  --tier2-data data/phase3_code/tier2_filtered.jsonl \
  --failures results/phase3_code_tier1_remaining.jsonl \
  --output results/phase3_code_tier2_remaining.jsonl
```

#### Step 6: Tier 3 Generation (1 day, $75)
```bash
python src/phase3_modifiers/generate_tier3.py \
  --failures results/phase3_code_tier2_remaining.jsonl \
  --teacher gpt-5 \
  --output data/phase3_code/tier3_filtered.jsonl
```

#### Step 7: Train Modifier (1 week)
```bash
# Combine all tiers
cat data/phase3_code/tier1_filtered.jsonl \
    data/phase3_code/tier2_filtered.jsonl \
    data/phase3_code/tier3_filtered.jsonl \
    > data/phase3_code/combined_9k.jsonl

# Train LoRA adapter
python src/phase3_modifiers/train_modifier.py \
  --base models/phase2e_recovered_520mb \
  --data data/phase3_code/combined_9k.jsonl \
  --domain code \
  --rank 128 \
  --epochs 5 \
  --output models/phase3_code/code_modifier_260mb
```

#### Step 8: Compress Modifier (3 days, $25)
```bash
python src/phase3_modifiers/compress_modifier.py \
  --modifier models/phase3_code/code_modifier_260mb \
  --sparsity-levels 0.78 0.82 0.85 \
  --output models/phase3_code/code_modifier_47mb

python src/phase3_modifiers/validate_modifier.py \
  --base models/phase2e_recovered_520mb \
  --modifier models/phase3_code/code_modifier_47mb \
  --benchmarks humaneval mbpp livecodebench \
  --target-performance 1.15  # 115% GPT-4
  --output results/phase3_code_validation.json
```

### Modifier Schedule

**Week 11-12: Code Modifier** ($200)
- Days 1-2: Test base on code (12K tasks)
- Days 3-5: Tier 1 (Qwen-Coder, 9K examples)
- Days 6: Tier 2 (DeepSeek-Coder)
- Day 7: Tier 3 (GPT-5)
- Days 8-14: Train + compress
- **Output:** 47MB code modifier (115-130% GPT-4)

**Week 12-13: Reasoning Modifier** ($207)
- Follow same process with:
  - Tier 1: Groq Llama-405B FREE (12K examples)
  - Tier 2: GPT-4o ($75)
  - Tier 3: GPT-5 + COT ($95)
  - Rank: 112
- **Output:** 48MB reasoning modifier (100-108% GPT-4)

**Week 13-14: Automation Modifier** ($170)
- Follow same process with:
  - Tier 1: Claude-3.5 (8K examples, $65)
  - Tier 2: GPT-4o ($50)
  - Tier 3: GPT-5 ($55)
  - Rank: 96
- **Output:** 40MB automation modifier (105-118% GPT-4)

**Phase 3 Complete Checklist:**
- ✅ Code modifier: 47MB, 115-130% GPT-4
- ✅ Reasoning modifier: 48MB, 100-108% GPT-4
- ✅ Automation modifier: 40MB, 105-118% GPT-4
- ✅ All beat GPT-4 on respective domains

**Phase 3 Total: 4 weeks, $685**

---

## 🧭 PHASE 4: ROUTER SYSTEM (2 Weeks, $75)

### Week 15: Router Training (Phase 4A-B)

#### Week 15, Day 1-3: Collect Training Data
```bash
# Generate Claude 4.5 script for data collection
python src/phase4_router/collect_router_data.py \
  --base-model models/phase2e_recovered_520mb \
  --test-queries 35000 \
  --collect-confidence \
  --collect-correctness \
  --output data/phase4_router/training_35k.jsonl

# This runs base model on 35K queries, takes 2-3 days
```

**Expected Time:** 3 days  
**Cost:** $0  
**Output:** 35K labeled examples for router

#### Week 15, Day 4-7: Train Router
```bash
# Generate Claude 4.5 script for router training
python src/phase4_router/train_router.py \
  --data data/phase4_router/training_35k.jsonl \
  --architecture feedforward \
  --layers 3 \
  --hidden-dims 64 32 \
  --output models/router_13mb

python src/phase4_router/validate_router.py \
  --router models/router_13mb \
  --test-size 5000 \
  --output results/phase4a_validation.json

# Expected: 97% accuracy
```

**Expected Time:** 4 days  
**Cost:** $45 (compute)  
**Output:** 13MB router

#### Week 15, Day 8-11: Escalation Detector
```bash
# Generate Claude 4.5 script for dissatisfaction detection
python src/phase4_router/train_escalation.py \
  --data data/phase4_router/dissatisfaction_6k.jsonl \
  --base-model bert-base-uncased \
  --epochs 3 \
  --output models/escalation_110mb

python src/phase4_router/distill_escalation.py \
  --teacher models/escalation_110mb \
  --output models/escalation_3mb

# Expected: 94% detection accuracy
```

**Expected Time:** 4 days  
**Cost:** $30 (compute)  
**Output:** 3MB escalation detector

### Week 16: Threshold Optimization (Phase 4C)

#### Week 16, Day 1-2: A/B Testing
```bash
python src/phase4_router/optimize_threshold.py \
  --router models/router_13mb \
  --base-model models/phase2e_recovered_520mb \
  --modifiers models/phase3_*/  \
  --test-thresholds 0.75 0.80 0.85 \
  --test-size 5000 \
  --output results/phase4c_optimal_threshold.json

# Expected: 80% threshold optimal
```

**Expected Time:** 2 days  
**Cost:** $0  
**Output:** Optimal threshold configuration

**Phase 4 Complete Checklist:**
- ✅ Router trained: 13MB, 97% accuracy
- ✅ Escalation detector: 3MB, 94% detection
- ✅ Threshold optimized: 80% default
- ✅ Session memory implemented

**Phase 4 Total: 2 weeks, $75**

---

## 🚀 PHASE 5: DEPLOYMENT (1 Week, $100)

### Week 17: HuggingFace Deployment & Validation

#### Day 1-2: Upload to HuggingFace
```bash
# 1. Create HF repository
huggingface-cli login
huggingface-cli repo create cogumi-llm-mvp --type model

# 2. Upload all components
python src/phase5_deployment/upload_to_hf.py \
  --base models/phase2e_recovered_520mb \
  --modifiers models/phase3_*/ \
  --router models/router_13mb \
  --escalation models/escalation_3mb \
  --repo cogumi-llm-mvp

# Total upload: 671MB
```

**Expected Time:** 4 hours  
**Cost:** $0  
**Output:** Model on HuggingFace Hub

#### Day 2-3: Setup Inference API
```bash
# Generate Claude 4.5 script for HF Inference setup
python src/phase5_deployment/setup_inference_api.py \
  --repo cogumi-llm-mvp \
  --instance-type t4-gpu \
  --pricing serverless \
  --output configs/hf_inference.json
```

**Expected Time:** 1 day  
**Cost:** $0 (setup only, usage billed later)  
**Output:** REST API endpoint

#### Day 3-4: Create Gradio Interface
```bash
# Generate Claude 4.5 script for Gradio app
python src/phase5_deployment/create_gradio_app.py \
  --repo cogumi-llm-mvp \
  --features streaming history routing manual-override \
  --output src/phase5_deployment/gradio_app.py

# Deploy to HF Spaces
huggingface-cli space create cogumi-chat --type gradio
python src/phase5_deployment/deploy_gradio.py \
  --space cogumi-chat \
  --app src/phase5_deployment/gradio_app.py
```

**Expected Time:** 2 days  
**Cost:** $0  
**Output:** Public chat interface

#### Day 5: Monitoring Setup
```bash
python src/phase5_deployment/setup_monitoring.py \
  --grafana-config configs/grafana.yaml \
  --log-all-requests \
  --track-routing \
  --track-quality
```

**Expected Time:** 1 day  
**Cost:** $0  
**Output:** Grafana dashboard

#### Day 6-7: End-to-End Validation
```bash
# Automated quality gates
python src/phase5_deployment/run_quality_gates.py \
  --code-threshold 0.72 \
  --reasoning-threshold 0.70 \
  --automation-threshold 0.75 \
  --size-limit 650MB \
  --output results/phase5_quality_gates.json

# Human evaluation
python src/phase5_deployment/coordinate_human_eval.py \
  --num-users 100 \
  --tasks-per-user 20 \
  --output results/phase5_human_eval.json

# Performance benchmarking
python src/phase5_deployment/benchmark_performance.py \
  --platforms m4-pro rtx4090 a100 hf-t4 \
  --output results/phase5_performance.json
```

**Expected Time:** 2 days  
**Cost:** $100 (compute for validation)  
**Output:** Comprehensive validation report

**Go/No-Go Decision:**
- ✅ All quality gates pass → Deploy to production
- ❌ Any gate fails → Debug and re-validate

**Phase 5 Complete Checklist:**
- ✅ Models uploaded to HuggingFace
- ✅ Inference API live
- ✅ Gradio chat interface deployed
- ✅ Monitoring dashboard operational
- ✅ All quality gates passed
- ✅ Human evaluation >7.5/10
- ✅ Performance benchmarks met

**Phase 5 Total: 1 week, $100**

---

## 🎯 MVP COMPLETE

**Total Timeline:** 14 weeks (October 19 - January 18, 2026)  
**Total Cost:** $1,717  
**System Size:** 668MB (520MB base + 135MB modifiers + 13MB router)  
**Performance:** Beats GPT-4 on code, reasoning, automation  
**Inference:** 60+ tps on M4 Pro Mac, 80+ on RTX 4090

---

## 📦 OPTIONAL PHASE 2: DOMAIN EXPANSION (12 Weeks, $1,151)

**Note:** Only proceed after MVP validation and user feedback

### Weeks 18-29: Five Additional Modifiers

Each modifier follows the same 3-tier cascaded pipeline established in Phase 3. Reuse all scripts with domain-specific configurations.

**Week 18-19: Math Modifier** ($207)
- Tier 1: Qwen-Math (62%), Tier 2: DeepSeek-Math (23%), Tier 3: GPT-5 (15%)
- **Output:** 42MB, 92-102% GPT-4

**Week 20-21: Hard Math Modifier** ($219)
- Similar tiers for competition-level problems
- **Output:** 44MB, 98-110% GPT-4

**Week 22-23: Science Modifier** ($180)
- Tier 1: Llama-70B (65%), Tier 2: Gemma-27B (21%), Tier 3: GPT-5 (14%)
- **Output:** 36MB, 120-130% GPT-4

**Week 24-25: Finance Modifier** ($169)
- Tier 1: FinGPT (68%), Tier 2: InvestLM (19%), Tier 3: GPT-5 (13%)
- **Output:** 30MB, 115-125% GPT-4

**Week 26-27: Creative Modifier** ($185)
- Tier 1: Claude-3.5 (62%), Tier 2: GPT-4o (23%), Tier 3: GPT-5 (15%)
- **Output:** 44MB, 95-105% GPT-4

**Week 28: Enhancements** ($80)
- Self-consistency voting
- Adaptive threshold learning
- Multi-mode variants (Turbo/Balanced/Max)

**Week 29: Full System Validation** ($100)
- Test all 8 domains
- Update deployment
- Extended monitoring

**Phase 2 Total: 12 weeks, $1,151**

---

## 🎉 FULL SYSTEM COMPLETE

**Total Timeline:** 26 weeks (October 19 - April 2026)  
**Total Cost:** $2,868  
**System Size:** 864MB (520MB base + 8 modifiers + router)  
**Performance:** Beats GPT-4 on 6+ domains  
**Domains:** Code, Reasoning, Automation, Math, Hard Math, Science, Finance, Creative

---

## 📋 DAILY CHECKLIST TEMPLATE

Use this template for daily progress tracking:

```markdown
## Daily Progress: [Date]
**Phase:** [Current Phase]
**Week:** [X of Y]

### Today's Tasks
- [ ] Task 1
- [ ] Task 2
- [ ] Task 3

### Completed
- [x] Task A
- [x] Task B

### Blockers
- [Issue description and mitigation plan]

### Metrics
- GPU hours used: X/Y
- Budget spent: $X/$Y
- Time on track: Yes/No

### Tomorrow's Plan
- Next 3 tasks
- Resource requirements
- Expected deliverables

### Notes
[Any observations, learnings, or decisions made]
```

---

## 🚨 TROUBLESHOOTING GUIDE

### Common Issues & Solutions

**1. OOM During Training**
- Reduce batch size: 4 → 2
- Enable gradient checkpointing
- Use DeepSpeed Stage 2
- Switch to larger GPU (A100 80GB)

**2. Training Loss Not Decreasing**
- Check data quality (run validation script)
- Reduce learning rate by 2x
- Verify gradient accumulation working
- Check for NaN in weights

**3. Compression Degrades Quality >10%**
- Reduce sparsity: 65% → 60%
- Use more calibration samples
- Extend post-pruning fine-tuning
- Check if specific layers causing issues

**4. Router Accuracy <95%**
- Collect more training data (50K instead of 35K)
- Try deeper network (5 layers instead of 3)
- Balance classes better (undersample majority)
- Engineer better features

**5. Deployment Fails on HF**
- Check model size limits (5GB max for free)
- Verify all dependencies in requirements.txt
- Test locally with same Python version
- Check HF Space logs for errors

---

## 📞 SUPPORT & ESCALATION

**Technical Issues:**
- Axolotl GitHub: https://github.com/OpenAccess-AI-Collective/axolotl/issues
- Neural Magic Support: support@neuralmagic.com
- HuggingFace Community: https://huggingface.co/docs

**Budget Overruns:**
- Review grid search necessity (can skip for some experiments)
- Use spot instances on RunPod (30-50% cheaper)
- Reduce validation frequency
- Parallelize less, sequence more

**Timeline Delays:**
- Identify critical path tasks
- Parallelize independent work
- Use pre-trained checkpoints where available
- Consider hiring contractor for specific tasks

---

**Last Updated:** October 19, 2025  
**Next Review:** Week 1 (after vocabulary optimization)  
**Version:** 2.0 (LLAMA-3.2 Pipeline)  
**Contact:** [Your Email]
