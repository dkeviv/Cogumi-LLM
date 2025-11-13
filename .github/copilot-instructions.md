---
applyTo: '**'
---
Provide project context and coding guidelines that AI should follow when generating code, answering questions, or reviewing changes.

## SESSION START PROTOCOL (MANDATORY)

**At the start of EVERY new session or conversation:**

1. ✅ **Read QUICK_START.md FIRST** - Fast context snapshot (1 page, updated after major changes)
2. ✅ **Read .github/Revised_complete_pipeline.md** (at least first 100 lines) for complete pipeline architecture
3. ✅ **Read docs/IMPLEMENTATION_CHECKLIST.md** to see current phase and completed tasks
4. ✅ **Scan docs/technical_specification.md** (last 50-100 lines) for recent implementation changes
5. ✅ **Display brief status** to user:
   ```
   📍 Current Phase: [Phase X]
   ✅ Last Completed: [Task description]
   ⏳ In Progress: [Current work]
   🎯 Next: [Next task]
   📋 Architecture: [Brief 1-line summary]
   ```

**This ensures context continuity across sessions and prevents working with outdated assumptions.**

---

## (CRITICAL) Context

Refer to the below file everytime to retain context: **".github/Revised_Complete_Pipeline.md"** -> This has the full context

---

## (CRITICAL) Must do checks for scripts(MANDATORY)

While creating training script and merge script, confirm with the developer on the precision , INT vs FP etc .

In the training scripts, include data validation as the first step and not proceed if data issues found.

When updating a file, make sure to read the file to get the context , avoid any duplicate code or redundant code. Create optimal code with good commentary to help getting the context when you read the file. Make sure the commentary is crisp with minimal words but most meaningful

Don't reuse a script from a different phase. If it can be reused create another copy with phase number included in the file name

---

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
   - Update IMPLEMENTATION_CHECKLIST.md if needed
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

## CRITICAL: Best Practices (MANDATORY)

* Activate venv for all terminals
* After every feature implementation or bug fix, run all tests and linters to make sure nothing is broken. Make sure to add tests for new features or bug fixes and run tests to make sure features implemented before are working - Do smart tests for regression - not the full regression that is time consuming.
* Create context comment for every file created at the top of the file created and update the context comment for every file modified.
* Create proper commented code - every function, class, method should have proper docstring comments.
* Follow PEP8 coding style guidelines for Python code.
* Follow best practices for Python code.
* Use type hints for function signatures and variable declarations.
* Include guard rails for error handling and edge cases.
* Include proper error handling and logging.
* when making changes to requirements.txt, ensure compatibility with dependencies and if deleting any words/lines make you are not missing any context from the deleted lines
* Prompt appropriately after a feature is implemented to do git commit and git push to the remote repository

## **Naming** (STRICT):

- Variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- See `.naming-conventions.md` for specifics

## **Never Suggest**:

- Synchronous API calls
- Non-batch API usage
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

- **.github/Revised_complete_pipeline.md**: Full project context and current pipeline architecture

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

### (CRITICAL) Files to maintain accurately (MANDATORY)

Before updating the documentations  review the code if it is indeed implemented and tested properly.

Maintain features.md to keep track of features implemented and pending features to be implemented. Keep it updated.

Maintain technical_specification.md to document implementation details for all completed tasks. Update this file every time a todo list task is completed, capturing technical decisions, code structure, libraries used, and validation steps.

**CRITICAL:** After ANY update to technical_specification.md, read the ENTIRE file to verify accuracy. technical_specification.md is supposed to capture what was actually implemented and HOW it was implemented. Keep it accurate and up-to-date.

### Maintain .github/FILE_REGISTRY.md for ALL code files and documentation (MANDATORY)

**Purpose:** Comprehensive inventory of all files organized by phase with purpose, methods, and outcomes.

**When to Update:**

- After creating new scripts or code files
- After deprecating/archiving old files
- After major refactoring or file moves
- After completing phases (update status markers)
- After strategic pivots or approach changes

**What to Update:**

- Add new files to appropriate phase section
- Update file purposes and methods used
- Move deprecated files to archive sections with notes
- Update status markers (✅/⏳/❌)
- Update file locations if moved/renamed

**Update Pattern:**

```bash
# 1. Read FILE_REGISTRY.md to understand current structure
# 2. Add/update entries in appropriate phase section
# 3. Include: file path, purpose, methods/algorithms, achieves
# 4. Update status markers and timestamps
# 5. Commit with descriptive message
```

**Why This Matters:**

- Single source of truth for all project files
- Easy navigation to find specific functionality
- Documents methods and algorithms used
- Tracks deprecated vs active files
- Helps onboard new developers
- Shows project structure evolution

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

## Context Retention & Session Continuity

### CRITICAL: Always Check Current Architecture

**Review the file ".github/Revised_complete_pipeline.md" for the current architecture**

### Before Making Any Changes

1. **Read .github/Revised_complete_pipeline.md FIRST** - Primary reference for current pipeline
2. **Check docs/IMPLEMENTATION_CHECKLIST.md** - Current phase and completed tasks
3. **Verify Current Architecture** - Refer to Revised_complete_pipeline.md for details
4. **Check technical_specification.md** - Implementation details and algorithms used

### Key Context Files (Read Before Major Changes)

1. **QUICK_START.md** - Fast context snapshot (updated after major changes, read this FIRST in new sessions!)
2. **.github/Revised_complete_pipeline.md** - PRIMARY reference for complete pipeline architecture
3. **docs/IMPLEMENTATION_CHECKLIST.md** - Detailed task tracking with checkboxes
4. **docs/technical_specification.md** - Complete technical methodology and implementation details
5. **README.md** - Project overview, quick start, benchmarks
6. ***Don't keep creating new markdowns without asking. It is critical to keep limited documents or tracking is harder and confusing***

### Documentation Guidelines (STRICT)

- **NEVER create new summary/overview MD files** - Consolidate into existing docs
- **Use Revised_complete_pipeline.md as primary reference** - Full pipeline architecture
- **Before creating new doc:** Check if content fits in existing file
- **Active docs target:** Keep limited documents for easier tracking
- **Archive completed work:** Move old/obsolete docs to archive folders

### Session Continuity Checklist

When resuming work or starting new features:

- [ ] Read Revised_complete_pipeline.md to understand current pipeline architecture
- [ ] Review technical_specification.md for implementation details
- [ ] Check IMPLEMENTATION_CHECKLIST.md for completed/pending tasks
- [ ] Check if new work fits existing documentation structure
- [ ] Verify venv is activated before any Python operations
- [ ] Run tests/linters after changes to catch regressions

### Documentation Structure & Update Guidelines

**Revised_complete_pipeline.md - Primary Architecture Reference:**

- Purpose: Complete pipeline architecture and phase definitions
- Format: Phases → steps → methods → expected outcomes
- Update when: Major architecture/approach changes or phase transitions
- Keep: Current architecture and methodology

**technical_specification.md - Detailed Implementation:**

- Purpose: Complete technical details of what's implemented
- Format: Algorithms, thresholds, parameters, formulas, code structure
- Update when: Every todo task completion with full technical depth
- Keep: Very clear, precise, reproducible implementation details

**IMPLEMENTATION_CHECKLIST.md - Task Tracking:**

- Purpose: Granular task tracking with checkboxes
- Format: Nicely formatted checklist with status indicators
- Update when: Every subtask completion
- Keep: Current format (clean, scannable, checkbox-based)

### When to Update Each File

- **Revised_complete_pipeline.md:** Update when pivoting strategies, changing architecture, or completing phases
- **technical_specification.md:** Document algorithms, thresholds, code structure after implementation. **CRITICAL: Review entire file after updates to ensure accuracy.**
- **IMPLEMENTATION_CHECKLIST.md:** Check off subtasks as completed, update status markers
- **Never:** Create new summary/overview docs (consolidate into above structure)

### Technical Specification Requirements

- **Algorithm documentation**: Include algorithm name, complexity, why chosen, alternatives considered
- **Performance metrics**: Runtime, memory usage, throughput for large-scale operations
- **Optimization notes**: Document performance bottlenecks found and how they were resolved
- **Review protocol**: After ANY update, read entire technical_specification.md to verify accuracy

## Enforcement & Self-Check Protocol

### BEFORE Every Response, Ask Yourself:

1. **Pipeline Check:** Am I following the pipeline in .github/Revised_complete_pipeline.md?
2. **Phase Check:** What phase are we in? Check actual implementation status
3. **Model Check:** Am I using the correct model as specified in Revised_complete_pipeline.md?
4. **Docs Check:** Am I consolidating into existing docs (NOT creating new ones)?
5. **Git Check:** Did I prompt for git commit after implementing features?
6. **Context Check:** Did I read Revised_complete_pipeline.md before making changes?

### CONTEXT VALIDATION (Every Session)

**Before responding to first user query, internally verify:**

- ✓ **What phase are we in?** (Read IMPLEMENTATION_CHECKLIST.md)
- ✓ **What's the current architecture?** (Read Revised_complete_pipeline.md)
- ✓ **What was the last change?** (Check technical_specification.md recent updates)
- ✓ **Any recent pivots?** (Look for "PIVOTED:" or "UPDATED:" markers in docs)

**If context unclear → Ask user to confirm current state before proceeding.**

**Red Flags:**
- Documents contradict each other → Ask user which is current
- Implementation doesn't match documentation → Verify actual state
- Major architecture changes not documented → Confirm with user

### COLLABORATION & DECISION MAKING

- Before implementing, actively suggest alternatives, clarify requirements, and ask "why" to understand the user's intent.
- Encourage brainstorming and collaborative discussion before finalizing a solution.
- Only proceed to implementation after confirming the best approach with the user.
- This collaborative process should be followed for all major decisions, as demonstrated with the public dataset selection.

### After Every Feature/Change:

- [ ] Run tests/linters
- [ ] Update relevant documentation (technical_specification.md or IMPLEMENTATION_CHECKLIST.md)
- [ ] **Prompt user to commit and push changes**
- [ ] Verify changes match current architecture plan in Revised_complete_pipeline.md

### Checklist Display & Alignment

- Always display the current implementation checklist (from IMPLEMENTATION_CHECKLIST.md) at the start of each session.
- Keep the assistant's todo list fully aligned and synchronized with IMPLEMENTATION_CHECKLIST.md.

### After Every Todo Task Completion:

- [ ] **Validate ALL success criteria** defined for the task
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
