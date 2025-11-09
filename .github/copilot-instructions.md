---
applyTo: '**'
---
Provide project context and coding guidelines that AI should follow when generating code, answering questions, or reviewing changes.


# GitHub Copilot Context - Cogumi-LLM Pipeline

## 🎯 SIMPLIFIED PIPELINE FLOW - ALWAYS FOLLOW THIS SEQUENCE

### **Llama-3.1-8B-Instruct→ 668MB System Beating GPT-4**

**Phase 0: Dataset Creation** ✅ **COMPLETE**

- Multi-teacher distillation: Llama-405B (40%), GPT-4o (35%), Qwen3-Coder-480B (25%)
- Quality filtering: GPT-4-mini scoring (>7/10 threshold)
- MinHash LSH deduplication: Jaccard 0.8, removed 150K duplicates
- Output: 600K curated examples at `/data/phase1/public_500k_filtered.jsonl`

---

### **Phase 1: Base Model Training** (4 weeks, $505)

**1.1 Vocabulary Trimming**

- Student model: Llama-3.1-8B-Instruct (8.3B parameters)
- Vocabulary optimization: 128K tokens → 25K tokens (saves 3.4GB)
- Method: Token frequency analysis on 10K English samples
- Validation: <3% perplexity increase, 99.5%+ coverage

**1.2 Base QLoRA Training (Phase 1A)**

- Framework: Axolotl with QLoRA (rank 64, 4-bit)
- Dataset: 600K curated examples from Phase 0
- Training: 3 epochs, ~28K steps, early stopping
- Duration: 2.5 weeks on A100 40GB
- Cost: $220 (120 GPU-hours @ $1.89/hr)
- Output: 10GB merged base model
- Performance: 75-82% GPT-4

**1.3 Failure Analysis (Phase 1B)**

- Comprehensive testing: 50K diverse examples across domains
- Failure identification: 12-14K failures detected
- Clustering: Sentence-BERT embeddings + KMeans (k=10)
- Auto-labeling: GPT-4-mini identifies 8-12 failure patterns
- Cost: $5

**1.4 GPT-5 Targeted Distillation (Phase 1C)**

- Teacher: GPT-5 (elite model for hardest cases)
- Data generation: 40K examples targeting failure patterns
- Quality filtering: GPT-4-mini (>8/10), yields 40K high-quality
- Training: 90% GPT-5 data + 10% original (prevent forgetting)
- Learning rate: 3e-6 (lower to avoid catastrophic forgetting)
- Duration: 5 days on A100
- Cost: $280 GPT-5 + $5 scoring
- Output: 10GB enhanced base model
- **Performance: 88-100% GPT-4** ✅

---

### **Phase 2: Extreme Compression** (6 weeks, $402)

**2.1 Neural Magic Structured Pruning (2 weeks, $180)**

- Technique: Neural Magic llm-compressor
- Target sparsity: 65% (remove 65% of weights)
- Method: Gradual pruning over 2K steps (0% → 16.25% → 32.5% → 48.75% → 65%)
- Calibration: 10K diverse samples
- Post-pruning recovery: 8 hours fine-tuning (lr 1e-6)
- Output: 3.5GB sparse model
- Quality loss: 2-4%

**2.2 AWQ 4-bit Quantization (1 week, $90)**

- Technique: AutoAWQ mixed-precision quantization
- Bits: 4-bit with group size 128
- Calibration: 2K samples
- Output: 900MB quantized model
- Quality loss: 2-3% (cumulative: 4-7%)

**2.3 GGUF Export (3 days)**

- Technique: llama.cpp GGUF format
- Variant: Q5_K_M (mixed 5-bit/6-bit)
- Validation: 95%+ token agreement with original
- Output: 600MB GGUF model
- Quality loss: 1-2% (cumulative: 5-9%)

**2.4 Zstd Lossless Compression (2 days)**

- Technique: Zstandard with dictionary training
- Dictionary: 128KB trained on 100MB sample
- Compression level: 10 (maximum)
- Validation: SHA-256 checksum verification
- Output: 500MB compressed model
- Quality loss: 0% (lossless)

**2.5 Recovery Fine-Tuning (1 week, $70)**

- Selection: Hardest 12K examples (top 2% perplexity)
- Enhancement: GPT-5 improves examples
- Training: Conservative LoRA (rank 64, lr 8e-7, 2 epochs)
- Output: 520MB recovered base + 20MB LoRA
- Quality improvement: +1-2%
- **Final: 520MB base, 89-91% GPT-4** ✅

**2.6 Confidence Calibration (3 days, $35)**

- Data collection: 30K queries with logits
- Labeling: GPT-4-mini quality scoring
- Methods: Temperature scaling + Platt scaling
- Validation: ECE <0.05, 97% routing accuracy
- Output: Calibrators for routing system

---

### **Phase 3: Domain Modifiers - 3-Tier Cascaded Teaching** (4 weeks, $685)

**3-Tier Strategy** (saves 61% cost vs single-teacher):

- **Tier 1 (60-70%):** Free/cheap models handle easy cases
- **Tier 2 (20-25%):** Mid-tier models (GPT-4o, DeepSeek) for moderate difficulty
- **Tier 3 (10-15%):** GPT-5 for hardest cases only

**Reusable Pipeline for Each Modifier:**

1. Test base on 12K domain tasks → identify failures
2. Generate Tier 1 data for failures (9K examples)
3. Test Tier 1 → identify remaining failures
4. Generate Tier 2 data for remaining
5. Test Tier 2 → identify remaining failures
6. Generate Tier 3 data with GPT-5
7. Train LoRA adapter on combined data
8. Compress modifier via pruning (260MB → 40-48MB)

**3.1 Code Modifier** (Week 11-12, $200, 47MB)

- Teachers: Qwen-Coder-480B (Tier 1), DeepSeek-Coder (Tier 2), GPT-5 (Tier 3)
- LoRA rank: 128
- Training: 12.5K examples (9K + 2K + 1.5K)
- Compression: 78-85% sparsity → 47MB
- **Performance: 115-130% GPT-4 on HumanEval, MBPP** ✅

**3.2 Reasoning Modifier** (Week 12-13, $207, 48MB)

- Teachers: Llama-405B FREE (Tier 1), GPT-4o (Tier 2), GPT-5+COT (Tier 3)
- LoRA rank: 112
- Training: 17K examples (12K + 3K + 2K)
- Compression: 78-85% sparsity → 48MB
- **Performance: 100-108% GPT-4 on MMLU, BBH** ✅

**3.3 Automation Modifier** (Week 13-14, $170, 40MB)

- Teachers: Claude-3.5 (Tier 1), GPT-4o (Tier 2), GPT-5 (Tier 3)
- LoRA rank: 96
- Training: 11.5K examples (8K + 2K + 1.5K)
- Compression: 78-85% sparsity → 40MB
- **Performance: 105-118% GPT-4 on tool-use tasks** ✅

**Modifier Compression Techniques:**

- Initial LoRA adapter: 260MB uncompressed
- Pruning: Test sparsity levels (78%, 82%, 85%)
- Target: 40-48MB per modifier
- Validation: Must exceed 115% GPT-4 for code, 100% for reasoning, 105% for automation

---

### **Phase 4: Router System** (2 weeks, $75)

**4.1 Router Training** (1 week, $45, 13MB)

- Architecture: 3-layer feedforward (input 128 → hidden 64,32 → output 4)
- Features: Confidence scores, query embeddings, session history
- Training data: 35K labeled examples (query → correct model decision)
- Validation: 5K holdout set
- **Performance: 97% routing accuracy** ✅
- Latency: <5ms on M4 Pro Mac

**4.2 Escalation Detector** (4 days, $30, 3MB)

- Purpose: Detect user dissatisfaction
- Base: BERT-base-uncased (110MB)
- Training: 6K dissatisfaction examples
- Distillation: LSTM (110MB → 3MB, 36.7x compression)
- **Performance: 94% detection accuracy** ✅
- Latency: <3ms

**4.3 Threshold Optimization** (2 days)

- A/B testing: 75%, 80%, 85% confidence thresholds
- Test size: 5K queries
- Optimal: 80% (balanced quality vs cost)
- Expected distribution: Base 45-55%, Code 20-25%, Reasoning 15-20%, Automation 10-15%

**Session Memory:**

- Storage: SQLite (lightweight persistence)
- Tracking: Last 5 queries, routing decisions, success/failure
- Learning: Improve routing from session history

---

### **Phase 5: Deployment & Validation** (1 week, $100)

**5.1 HuggingFace Upload** (Day 1-2)

- Repository: cogumi-llm-mvp
- Components: 520MB base + 135MB modifiers + 13MB router + 3MB escalation
- Total: 671MB

**5.2 Inference API Setup** (Day 2-3)

- Instance: T4 GPU (serverless)
- Features: Streaming responses, REST API
- Cost per query: ~$0.003

**5.3 Gradio Interface** (Day 3-4)

- Features: Chat, history, router visualization, manual override
- Deployment: HuggingFace Spaces (cogumi-chat)

**5.4 Monitoring Dashboard** (Day 5)

- Platform: Grafana
- Metrics: Query volume, routing distribution, quality scores, latency, cost

**5.5 Validation** (Day 6-7, $100)

- Automated quality gates: Code >72%, Reasoning >70%, Automation >75%
- Human evaluation: 100 users × 20 tasks, target >7.5/10
- Performance benchmarks: M4 Pro 60+ tps, RTX 4090 80+ tps, A100 120+ tps

---

## 📊 FINAL SYSTEM SPECIFICATIONS

**Total Size:** 668MB

- Base model: 520MB (89-91% GPT-4)
- Code modifier: 47MB (115-130% GPT-4)
- Reasoning modifier: 48MB (100-108% GPT-4)
- Automation modifier: 40MB (105-118% GPT-4)
- Router: 13MB (97% accuracy)
- Escalation detector: 3MB (94% accuracy)

**Runtime Memory:** 568MB max (520MB base + 48MB largest modifier)

**MVP Timeline:** 14 weeks (Phase 0 complete, Phases 1-5 pending)

**MVP Cost:** $1,717 total

- Phase 0: $0 (complete)
- Phase 1: $505
- Phase 2: $402
- Phase 3: $685
- Phase 4: $75
- Phase 5: $100

**Inference Performance:**

- M4 Pro Mac: 60+ tokens/sec
- RTX 4090: 80+ tokens/sec
- A100: 120+ tokens/sec
- HuggingFace T4: 40+ tokens/sec

---

## CRITICAL PROJECT CONTEXT - READ AFTER PIPELINE FLOW

### Current Status

- **Phase 0:** ✅ COMPLETE (600K dataset ready)
- **Phase 1-5:** ⏳ PENDING
- **Overall Progress:** 6% (Phase 0 done, infrastructure ready)

### Code Conventions

- Async/await for all API calls
- Rich progress bars for long operations
- Cost tracking in every API call
- MinHash LSH for deduplication
- English-only data generation
- **ALWAYS optimize for performance**: Use fastest algorithms, add progress indicators, profile bottlenecks
- **Hash algorithms**: Use xxhash (fastest) over MD5/SHA for non-cryptographic hashing
- **Progress bars**: Add for ANY operation >30 seconds expected runtime
- **Algorithm review**: Before implementing, check if faster alternatives exist

### Long-Running Operations (CRITICAL)

**For ANY script expected to run >5 minutes:**

1. **Progress Indicators (MANDATORY):**

   - Rich progress bars showing current item/total
   - Elapsed time and estimated completion
   - Current operation description
   - Percentage complete
2. **Debug Monitoring (MANDATORY):**

   - Log progress every N iterations (e.g., every 10K items)
   - Checkpoint files to track progress (can resume if interrupted)
   - Memory usage monitoring (warn if exceeds thresholds)
   - Intermediate statistics output
3. **Verification Hooks (MANDATORY):**

   - Sample output after first 100 items (verify correctness early)
   - Periodic validation checks (e.g., every 50K items)
   - Error rate monitoring (stop if error rate >threshold)
4. **Performance Logging:**

   - Items/second throughput
   - Estimated time remaining
   - Bottleneck identification (which stage is slowest)

**Example Pattern:**

```python
with Progress(...) as progress:
    task = progress.add_task(f"Processing {total} items", total=total)
    for i, item in enumerate(items):
        # Process item
        progress.advance(task)
  
        # Debug checkpoint every 10K
        if i % 10000 == 0:
            logger.info(f"Progress: {i}/{total} ({i/total:.1%}) - {throughput:.0f} items/sec")
```

**Rationale:** Long-running scripts must be monitorable and debuggable. Without progress indicators and checkpoints, impossible to diagnose issues or estimate completion.

### Test-First Approach for Large-Scale Operations (MANDATORY)

**Before running on full datasets:**

1. **Small Dataset Test (ALWAYS):**

   - Test with 100-1,000 samples first
   - Verify output correctness
   - Measure throughput (samples/sec)
   - Estimate full runtime: `total_samples / throughput`
   - Example: If 1K samples take 10 seconds → 674K samples = 1.87 hours
2. **Medium Dataset Test (if runtime >30 mins):**

   - Test with 10K-50K samples
   - Validate at scale (memory, performance)
   - Confirm linear scaling (no exponential slowdown)
   - Adjust algorithm if performance degrades
3. **Full Dataset Run:**

   - Only proceed after successful small/medium tests
   - Ensure progress indicators working
   - Monitor first 10 minutes for issues

**Test Pattern:**

```python
# Add test mode parameter
def process_data(input_file, output_file, test_mode=False):
    if test_mode:
        # Process only first 1000 samples
        data = data[:1000]
        output_file = output_file.replace('.jsonl', '_test.jsonl')
  
    # Main processing...
```

**Rationale:** Catching issues on 1K samples (10 seconds) vs 674K samples (2 hours) saves massive time. Always validate correctness and performance on small data first.

### CRITICAL: Always Read Files Before Editing (MANDATORY)

**BEFORE making ANY changes to a file:**

1. **Read the entire file first** using read_file tool
2. **Check for existing content** that might be duplicated
3. **Verify context** to understand current structure
4. **Then make targeted edits** using replace_string_in_file

**This applies to:**

- Code files (.py, .js, .ts, etc.)
- Documentation files (.md, .txt, etc.)
- Configuration files (.json, .yaml, etc.)
- Any file being modified

**Why this is critical:**

- Prevents duplicate code/content
- Avoids breaking existing functionality
- Maintains code quality and consistency
- Prevents messy, redundant files

**Example workflow:**

```
1. read_file(path) → understand current state
2. Plan changes based on what exists
3. replace_string_in_file() → make targeted edit
4. Verify no duplication introduced
```

**Red flag:** If you're about to edit a file without reading it first → STOP and read it.

## Script & Output Management (MANDATORY)

### Clean Up Old Outputs Before New Runs

**BEFORE running ANY script that generates output files:**

1. **Check for existing outputs** from previous runs
2. **Delete old outputs** to avoid confusion with new results
3. **Document cleanup** in run logs/comments

**Example workflow:**

```bash
# WRONG - runs script without cleanup, old data remains
python script.py

# CORRECT - cleans up first
rm -f output/*.jsonl
python script.py
```

**Common output locations:**

- `data/phase*/`: Dataset outputs
- `models/`: Model checkpoints
- `logs/`: Training logs
- `results/`: Evaluation results
- `*_test.jsonl`: Test run outputs

**Rationale:** Old outputs cause confusion - "Is this from the new run or old run?" Always start with clean slate.

### Legacy Script Management (MANDATORY)

**When creating an improved/replacement version of a script:**

1. **Add deprecation header to OLD script:**

   ```python
   """
   ⚠️ DEPRECATED - DO NOT USE ⚠️
   =================================

   This script is DEPRECATED and should NOT be used.

   **Replaced by:** path/to/new_script.py
   **Reason:** Brief explanation of why replaced
   **Performance Comparison:**
   - Old: [benchmark]
   - New: [benchmark]

   **Archive Date:** [Date]
   **Archived For:** Historical reference only

   =================================
   [Original docstring]
   """
   ```
2. **Archive old script:**

   - Copy to `src/utils/archive/` or appropriate archive folder
   - Keep original in place with deprecation header (for git history)
   - OR delete original and keep only archived copy
3. **Update documentation:**

   - Update technical_specification.md with replacement info
   - Update any references in other files
   - Add changelog entry in CURRENT_STATUS.md
4. **For superseded (still functional) scripts:**

   ```python
   """
   ⚠️ SUPERSEDED - USE NEWER VERSION ⚠️
   ========================================

   This script is SUPERSEDED by a better implementation.

   **Current Version:** path/to/better_script.py (RECOMMENDED)
   **This Version:** Still functional but slower/older

   **When to Use This:** [Specific use cases where old version is still valid]
   **When to Use New Version:** [When to prefer new version]

   ========================================
   [Original docstring]
   """
   ```

**Apply this to ALL file types:**

- Python scripts (.py)
- Jupyter notebooks (.ipynb)
- Configuration files (.json, .yaml)
- Documentation files (.md)
- Data files (if superseded by newer versions)

**Rationale:** Prevents using old/slow/deprecated code accidentally. Clear migration path. Preserves history for reference.

### Archive Folder Structure

```
src/
├── utils/
│   ├── archive/              # Archived utility scripts
│   │   ├── deduplication.py  # MD5 version (deprecated)
│   │   └── deduplication_optimized.py  # Sequential xxhash (superseded)
│   ├── deduplication_parallel.py  # CURRENT - use this
│   └── ...
```

**Red flags:**

- Running script without checking for old outputs → Clean up first
- Creating new version without deprecating old → Add deprecation header
- Multiple versions with no clear "current" indicator → Mark clearly
- No performance comparison in deprecation header → Add benchmarks

Activate venv for all terminals

## After every feature implementation or bug fix, run all tests and linters to make sure nothing is broken. Make sure to add tests for new features or bug fixes and run tests to make sure features implemented before are working - Do smart tests for regression - not the full regression that is time consuming.

## Create context comment for every file created at the top of the file created and update the context comment for every file modified.

## Create proper commented code - every function, class, method should have proper docstring comments.

## Follow PEP8 coding style guidelines for Python code.

## Follow best practices for Python code.

## Use type hints for function signatures and variable declarations.

## Include guard rails for error handling and edge cases.

## Include proper error handling and logging.

## **Naming** (STRICT):

- Variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- See `.naming-conventions.md` for specifics

## **Never Suggest**:

- Synchronous API calls
- Non-batch API usage
- Full precision training
- Multilingual data generation
- Libraries not in requirements.txt

## Before Generating Code, Check:

- [ ] Does it use batch API? (50% discount)
- [ ] Are there type hints?
- [ ] Is cost estimated first?
- [ ] Does it match naming conventions?
- [ ] Is it async if calling APIs?
- [ ] Are there guard rails (assertions)?
- [ ] Is it optimized? (fastest algorithm, no bottlenecks)
- [ ] Does it have progress indicators for long operations?
- [ ] Using xxhash instead of MD5/SHA for non-crypto hashing?

## Read These Files:

- `.copilot-context.md`: Full project context
- `.naming-conventions.md`: Naming standards
- `.adr/*.md`: Architecture decisions
- `.copilot-templates/*.py`: Code patterns

when making changes to requirements.txt, ensure compatibility with dependencies and if deleting any words/lines make you are not missing any context from the deleted lines

### Prompt appropriately after a feature is implemented to do git commit and git push to the remote repository

### After every todo task completion, ALWAYS check the Problems panel and fix all errors before marking the task as complete or moving to the next task. Use get_errors tool to check for compile errors, type errors, and linting issues. Fix all problems to maintain code quality.

### After every todo task completion, ALWAYS update .github/IMPLEMENTATION_CHECKLIST.md to reflect completed work. Mark tasks as complete (❌ → ✅), update status sections, and add implementation details. This keeps the checklist synchronized with actual progress.

### CRITICAL: Maintain .github/IMPLEMENTATION_CHECKLIST.md as Master Project Plan (MANDATORY)

**Purpose:** IMPLEMENTATION_CHECKLIST.md is the authoritative, detailed project plan that captures:
- Current state of all phases
- Completed vs pending tasks
- Status markers (✅/⏳/❌)
- Cost estimates and timelines
- Technical approach and decisions

**When to Update:**

- After completing ANY task (mark ✅)
- When pivoting strategies (update approach, remove obsolete steps)
- When discovering new requirements (add tasks)
- When updating cost/timeline estimates
- When changing technical decisions

**What to Update:**

- **Status markers:** Change ⏳ PENDING → ✅ COMPLETE (or ❌ → ✅)
- **Task details:** Add completion dates, actual costs, outputs
- **Approach sections:** Update methodology when pivots occur
- **Cost/timeline:** Update estimates with actuals
- **Dependencies:** Ensure prerequisite tasks marked correctly

**Critical Principle:** "This file is supposed to capture where we are and what was planned. If we pivot this has to be updated. This is like a project plan document that is detailed." - User emphasis, November 8, 2025

**Examples of Updates:**

- Task completion: "1C.3 Critique Batch Generation" ⏳ → ✅ COMPLETE
- Strategic pivot: Remove "bidirectional pairs" approach, update to "direct training"
- Cost update: "$40-45" → "$15-20" (after removing API generation)
- Approach change: Document rationale in section (e.g., "PIVOTED: No Bidirectional Pairs")

**Red Flags (DON'T):**

- ❌ Skip updates when completing tasks
- ❌ Leave obsolete approaches documented
- ❌ Update other docs but forget IMPLEMENTATION_CHECKLIST.md
- ❌ Mark complete without verifying all subtasks

**Quality Check:**

- Does status match reality? (✅ only if truly complete)
- Are costs/timelines current?
- Is technical approach accurate (no obsolete methods)?
- Are dependencies correctly marked?

### Maintain features.md to keep track of features implemented and pending features to be implemented. Keep it updated.

### Maintain technical implementation notes in the technical_notes.md file for future reference and knowledge sharing. Keep it updated with relevant information.

### Maintain technical_specification.md to document implementation details for all completed tasks. Update this file every time a todo list task is completed, capturing technical decisions, code structure, libraries used, and validation steps.

### Maintain docs/ISSUES_LOG.md for ALL bugs and issues (MANDATORY)

**When to Add Entry:**

- After fixing any bug that took >30 minutes to debug
- When discovering non-obvious root causes
- For any issue that might recur in similar contexts
- Performance issues where actual behavior differs from expected

**What to Include:**

- Issue description with user-visible symptoms
- Technical root cause with code snippets
- Solution applied with before/after code
- Files changed and line numbers
- Lesson learned and best practice to follow
- Links to related guidelines

**Format:**

```
## YYYY-MM-DD - Issue Title

**Issue:** User-visible problem description
**Root Cause:** Technical explanation with code
**Solution:** Fix applied with code snippets
**Files Changed:** List with line numbers
**Lesson Learned:** Key takeaway for future
**Related Guidelines:** Links to docs
```

**Why This Matters:**

- Creates searchable knowledge base of solved problems
- Prevents repeating same debugging work
- Documents non-obvious issues for future reference
- Helps onboard new developers
- Shows debugging patterns and solutions

**Example Use Cases:**

- "Progress bar stuck at 0%" → Check ISSUES_LOG for multiprocessing chunk size issues
- "Performance slower than expected" → Check scaling issues with small vs large datasets
- "Import errors in multiprocessing" → Check for circular import patterns

### Before updating anything as complete in the features.md , implementation_checklist double confirm if it is indeed implemented and tested properly.

### When updating a file, make sure to read the file to get the context , avoid any duplicate code or redundant code. Create optimal code with good commentary to help getting the context when you read the file. Make sure the commentary is crisp with minimal words but most meaningful

### Clear any temporary to keep the workspace clean. Maintain a separate folder for such teporary test and debug files

## Context Retention & Session Continuity

### CRITICAL: Always Check Current Architecture

**CURRENT PIPELINE (October 2025):** Llama-3.1-8B-Instruct with Multi-Phase Compression & Modifiers

- **Status:** ACTIVE - Phase 0 Complete, Phases 1-5 in progress
- **Student Model:** Llama-3.1-8B-Instruct(8.3B parameters)
- **Dataset:** 600K examples via multi-teacher distillation (Llama-405B, GPT-4o, Qwen3-Coder-480B)
- **Final Size:** 668MB (520MB base + 135MB modifiers + 13MB router)
- **Performance:** Beats GPT-4 on code (115-130%), reasoning (100-108%), automation (105-118%)
- **Cost:** $1,717 MVP (Phase 0: $0, Phase 1: $505, Phase 2: $402, Phase 3: $685, Phase 4: $75, Phase 5: $100)
- **Timeline:** 14 weeks total
- **Rationale:** Extreme compression (19.2x) with domain specialization beats monolithic models

### Before Making Any Changes

1. **Read CURRENT_STATUS.md FIRST** - Single source of truth for project status
2. **Check docs/IMPLEMENTATION_CHECKLIST.md** - Current phase and completed tasks
3. **Verify Current Architecture** - Llama-3.1-8B-Instruct with multi-phase compression
4. **Check Simplified Pipeline Flow** - Top of this file for exact sequence
5. **Review Recent Changelog** - Bottom of CURRENT_STATUS.md for latest decisions

### Key Context Files (Read Before Major Changes)

1. **docs/CURRENT_STATUS.md** - PRIMARY reference (Phase 0 complete, overall 6% progress)
2. **docs/IMPLEMENTATION_CHECKLIST.md** - Detailed task tracking across all 6 phases
3. **docs/EXECUTION_PLAN.md** - Step-by-step week-by-week execution guide
4. **docs/technical_specification.md** - Complete technical methodology and architecture
5. **README.md** - Project overview, quick start, benchmarks
6. **configs/*.yaml** - Training and compression configurations for each phase
7. ***Don't keep creating new markdowns without asking. It is critical to keep limited documents or tracking is harder and confusing***

### Documentation Guidelines (STRICT)

- **NEVER create new summary/overview MD files** - Consolidate into existing docs
- **Use CURRENT_STATUS.md as single source of truth** - Merge new content here
- **Changelog format:** 1 paragraph per major change (in CURRENT_STATUS.md)
- **Before creating new doc:** Check if content fits in existing file
- **Active docs target:** 4 core docs + README + phase READMEs
- **Archive completed work:** Old pipeline docs moved to docs/archive2/

### Session Continuity Checklist

When resuming work or starting new features:

- [ ] Read CURRENT_STATUS.md to understand current state
- [ ] Review Simplified Pipeline Flow (top of this file)
- [ ] Check IMPLEMENTATION_CHECKLIST.md for completed/pending tasks
- [ ] Review recent changelog entries (last 2-3 entries)
- [ ] Verify Phase 0 dataset location: `/data/phase1/public_500k_filtered.jsonl`
- [ ] Check if new work fits existing documentation structure
- [ ] Verify venv is activated before any Python operations
- [ ] Run tests/linters after changes to catch regressions

### Cost & Performance Targets (Current Pipeline)

- **Total Budget:** $1,717 (Phase 0-5 MVP)
- **Dataset Size:** 600K curated examples (Phase 0 complete)
- **Model Size:** 668MB total (520MB base + 135MB modifiers + 13MB router)
- **Performance:**
  - Base: 89-91% GPT-4
  - Code: 115-130% GPT-4 (HumanEval, MBPP)
  - Reasoning: 100-108% GPT-4 (MMLU, BBH)
  - Automation: 105-118% GPT-4 (ToolBench)
- **Training Time:** Phase 1: 4 weeks, Phase 2: 6 weeks, Phase 3: 4 weeks
- **Compression Ratio:** 19.2x (10GB → 520MB)

### Common Context Mistakes to Avoid

❌ **Don't assume old Qwen pipeline** - We're using Llama-3.1-8B-Instruct now
❌ **Don't skip vocabulary trimming** - 128K → 25K is Phase 1 step 1
❌ **Don't forget 3-tier cascading** - Cost optimization for modifiers
❌ **Don't create new summary docs** - Consolidate into existing 4 core docs
❌ **Don't forget Phase 0 is complete** - 600K dataset already ready
❌ **Don't use old compression targets** - New target is 520MB (not 480MB)
❌ **Don't skip calibration** - Phase 2F is critical for routing accuracy

### Phase-Specific Context

**Phase 0 (COMPLETE):** Dataset Creation ✅

- Multi-teacher distillation complete
- 600K curated examples ready
- Location: `/data/phase1/public_500k_filtered.jsonl`
- Quality: 8.2/10 average, 0% duplicates post-LSH

**Phase 1 (NEXT PRIORITY):** Base Model Training

- Vocabulary trimming: 128K → 25K tokens
- QLoRA training on 600K examples (Phase 1A)
- Failure analysis + clustering (Phase 1B)
- GPT-5 targeted distillation (Phase 1C)
- Output: 10GB enhanced base (88-100% GPT-4)

**Phase 2 (PENDING):** Compression

- Neural Magic pruning: 10GB → 3.5GB
- AWQ quantization: 3.5GB → 900MB
- GGUF export: 900MB → 600MB
- Zstd compression: 600MB → 500MB
- Recovery fine-tuning: 500MB → 520MB
- Confidence calibration for routing

**Phase 3 (PENDING):** Domain Modifiers

- Code modifier: 3-tier teaching, 47MB
- Reasoning modifier: 3-tier teaching, 48MB
- Automation modifier: 3-tier teaching, 40MB
- All use cascaded approach to save 61% cost

**Phase 4 (PENDING):** Router System

- Router: 13MB, 97% accuracy
- Escalation detector: 3MB, 94% accuracy
- Threshold optimization: 80% default

**Phase 5 (PENDING):** Deployment

- HuggingFace upload and Inference API
- Gradio chat interface
- Monitoring dashboard
- Comprehensive validation

### Quick Reference: Key Decisions

| Decision         | Choice                | Rationale                                      |
| ---------------- | --------------------- | ---------------------------------------------- |
| Student Model    | Llama-3.1-8B-Instruct | +14% params vs Qwen-7B, +2-3% English baseline |
| Data Source      | Multi-teacher (600K)  | Llama-405B + GPT-4o + Qwen-Coder diversity     |
| Deduplication    | MinHash LSH @ 0.8     | Removed 150K duplicates (20% of data)          |
| Base Training    | QLoRA 4-bit           | Cost-effective for 8B model                    |
| Compression      | 5-stage pipeline      | 19.2x compression (10GB → 520MB)              |
| Modifiers        | 3-tier cascaded       | 61% cost savings vs single-teacher             |
| Target Base      | 89-91% GPT-4          | Realistic after extreme compression            |
| Target Modifiers | 100-130% GPT-4        | Domain specialization beats monolithic         |
| Docs             | 4 core + README       | Minimal, maintainable structure                |
| Timeline         | 14 weeks MVP          | Phase 0 complete (6% progress)                 |

### Documentation Structure & Update Guidelines

**CURRENT_STATUS.md - High-Level Audit Trail:**

- Purpose: Searchable changelog of decisions for audit
- Format: Date + Decision + Rationale (1-2 sentences each)
- Update when: Major architecture/cost/dataset decisions made
- Keep: High-level only, easy to scan for past decisions

**technical_specification.md - Detailed Implementation:**

- Purpose: Complete technical details of what's implemented
- Format: Algorithms, thresholds, parameters, formulas, code structure
- Update when: Every todo task completion with full technical depth
- Keep: Very clear, precise, reproducible implementation details

**EXECUTION_PLAN.md - High-Level Task List:**

- Purpose: Project roadmap and phase overview
- Format: Phases → major milestones → success criteria
- Update when: Phase transitions or major scope changes
- Keep: Strategic overview, not detailed steps

**IMPLEMENTATION_CHECKLIST.md - Low-Level Task List:**

- Purpose: Granular task tracking with checkboxes
- Format: Nicely formatted checklist with status indicators
- Update when: Every subtask completion
- Keep: Current format (clean, scannable, checkbox-based)

### When to Update Each File

- **CURRENT_STATUS.md:** Add changelog entry for decisions (architecture, datasets, costs, approach changes)
- **technical_specification.md:** Document algorithms, thresholds, code structure after implementation. **CRITICAL: Review entire file after updates to ensure accuracy.**
- **EXECUTION_PLAN.md:** Update phase status and high-level milestones
- **IMPLEMENTATION_CHECKLIST.md:** Check off subtasks as completed
- **Never:** Create new summary/overview docs (consolidate into above structure)

### Technical Specification Requirements

- **Algorithm documentation**: Include algorithm name, complexity, why chosen, alternatives considered
- **Performance metrics**: Runtime, memory usage, throughput for large-scale operations
- **Optimization notes**: Document performance bottlenecks found and how they were resolved
- **Review protocol**: After ANY update, read entire technical_specification.md to verify accuracy

## Enforcement & Self-Check Protocol

### BEFORE Every Response, Ask Yourself:

1. **Pipeline Check:** Am I following the Simplified Pipeline Flow (see top of this file)?
2. **Phase Check:** Is Phase 0 complete? Are we starting Phase 1?
3. **Model Check:** Am I using Llama-3.1-8B-Instruct (NOT Qwen-7B)?
4. **Cost Check:** Am I using current budget ($1,717 MVP)?
5. **Compression Check:** Am I using 5-stage compression (Neural Magic → AWQ → GGUF → Zstd → Recovery)?
6. **Modifier Check:** Am I using 3-tier cascaded teaching for cost optimization?
7. **Docs Check:** Am I consolidating into 4 core docs (NOT creating new ones)?
8. **Git Check:** Did I prompt for git commit after implementing features?
9. **Context Check:** Did I read CURRENT_STATUS.md before making changes?

### COLLABORATION & DECISION MAKING

- Before implementing, actively suggest alternatives, clarify requirements, and ask "why" to understand the user's intent.
- Encourage brainstorming and collaborative discussion before finalizing a solution.
- Only proceed to implementation after confirming the best approach with the user.
- This collaborative process should be followed for all major decisions, as demonstrated with the public dataset selection.

### After Every Feature/Change:

- [ ] Run tests/linters
- [ ] Update relevant documentation (CURRENT_STATUS.md or IMPLEMENTATION_CHECKLIST.md)
- [ ] **Prompt user to commit and push changes**
- [ ] Verify changes match current architecture plan

### Checklist Display & Alignment

- Always display the current implementation checklist (from IMPLEMENTATION_CHECKLIST.md) at the start of each session.
- Keep the assistant's todo list fully aligned and synchronized with IMPLEMENTATION_CHECKLIST.md.

### After Every Todo Task Completion:

- [ ] **Validate ALL success criteria** from EXECUTION_PLAN.md
- [ ] Document validation results in validation log format
- [ ] **If ANY criterion fails → STOP, do NOT mark complete, iterate**
- [ ] Only mark todo complete when ALL success criteria pass
- [ ] Update todo with validation timestamp and results

**Validation Log Format for Every Task:**

```
Task X.Y: [Task Name]
✅ Criterion 1: [Result]
✅ Criterion 2: [Result]  
✅ Criterion 3: [Result]
→ ALL CRITERIA MET: Task complete
```

**Or if failure:**

```
Task X.Y: [Task Name]
✅ Criterion 1: [Result]
❌ Criterion 2: [Result] → FAILED
→ ITERATION REQUIRED: [Corrective action]
→ Task NOT complete until all criteria pass
```
