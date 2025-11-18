# Phase 1D: Benchmark-Based Validation Strategy

**Date:** November 18, 2025  
**Phase:** 1D - Validation and Merge  
**Status:** ✅ Implementation Complete

---

## Context: Why Benchmark-Based Validation?

### The Problem with Extracted Test Sets

**Initial Approach (INVALID ❌):**
- Extract 202 examples from `training_data_clean.jsonl`
- Stratify by difficulty (easy/hard) and domain (8 domains)
- Use as "held-out" test set

**Critical Flaw Identified:**
- Model was trained on ALL 53,597 examples in training_data_clean.jsonl
- No examples were truly "held out" during training
- Extracted test set = subset of training data = **test set contamination**
- **Result:** Would test training accuracy, NOT generalization ability

**User Insight (November 18, 2025):**
> "didn't you create the test data from the training data"  
> "But did we really hold out?"

**Answer:** No, we didn't hold out any data. The model has seen ALL examples.

### The Solution: Independent Benchmark Datasets

**Corrected Approach (VALID ✅):**
- Use independent benchmark datasets downloaded previously
- These benchmarks were NEVER seen during training
- Provides **true generalization test** on unseen data
- Standard benchmarks allow comparison with published results

**Benchmarks Used:**
1. **DROP** (9,535 examples) - Reading comprehension, discrete reasoning
2. **GPQA** (1,319 examples) - Graduate-level science Q&A
3. **HumanEval** (164 examples) - Python code generation
4. **MATH** (1,319 examples) - Mathematical reasoning and problem solving
5. **MMLU** (14,042 examples) - Multitask language understanding (57 subjects)
6. **~~MGSM~~** (249 examples) - Multilingual grade-school math (excluded: empty format)

**Total Available:** 26,628 benchmark examples across 5 datasets

---

## Conversion Strategy

### Format Standardization

**Input:** Each benchmark has different schema
**Output:** Unified validation format compatible with `phase1_validate_maml.py`

**Target Schema:**
```json
{
  "prompt": "Question with instructions and context",
  "response": "Expected answer or solution",
  "metadata": {
    "difficulty": "easy|medium|hard",
    "domain": "domain_name",
    "source": "benchmark_name",
    "...": "benchmark-specific fields"
  }
}
```

### Conversion Details by Benchmark

#### 1. DROP (Reading Comprehension)
```python
# Original format
{
  "passage": "Long text passage...",
  "question": "Question about passage",
  "answers": ["Answer 1", "Answer 2"]
}

# Converted format
{
  "prompt": "Read the following passage and answer the question.\n\nPassage: {passage}\n\nQuestion: {question}\n\nAnswer:",
  "response": "{answers[0]}",  # Use first answer (most specific)
  "metadata": {
    "difficulty": "easy",
    "domain": "reading_comprehension",
    "source": "DROP",
    "num_answers": 2
  }
}
```

#### 2. GPQA (Graduate-Level Science)
```python
# Original format
{
  "question": "Graduate-level science question",
  "answer": "Detailed answer"
}

# Converted format
{
  "prompt": "Answer the following graduate-level science question.\n\nQuestion: {question}\n\nAnswer:",
  "response": "{answer}",
  "metadata": {
    "difficulty": "hard",  # Always hard (graduate-level)
    "domain": "science_qa",
    "source": "GPQA"
  }
}
```

#### 3. HumanEval (Code Generation)
```python
# Original format
{
  "task_id": "HumanEval/0",
  "prompt": "def has_close_elements(...):\n    \"\"\"Docstring\"\"\"\n",
  "canonical_solution": "    # Function body",
  "entry_point": "has_close_elements"
}

# Converted format
{
  "prompt": "Complete the following Python function.\n\n{prompt}",
  "response": "{canonical_solution}",
  "metadata": {
    "difficulty": "medium",
    "domain": "code_generation",
    "source": "HumanEval",
    "task_id": "HumanEval/0",
    "entry_point": "has_close_elements"
  }
}
```

#### 4. MATH (Mathematical Reasoning)
```python
# Original format
{
  "problem": "Math problem text",
  "solution": "Step-by-step solution",
  "level": "Level 1",
  "type": "Algebra"
}

# Converted format
{
  "prompt": "Solve the following math problem. Show your work.\n\nProblem: {problem}\n\nSolution:",
  "response": "{solution}",
  "metadata": {
    "difficulty": "medium|hard",  # Based on level
    "domain": "mathematics",
    "source": "MATH",
    "level": "Level 1",
    "type": "Algebra"
  }
}
```

#### 5. MMLU (Multitask Understanding)
```python
# Original format
{
  "question": "Multiple choice question",
  "choices": ["A option", "B option", "C option", "D option"],
  "answer": 1,  # Index of correct answer
  "subject": "abstract_algebra"
}

# Converted format
{
  "prompt": "Answer the following multiple choice question from {subject}.\n\nQuestion: {question}\n\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n\nAnswer:",
  "response": "B. {choices[1]}",  # Format: "Letter. Answer text"
  "metadata": {
    "difficulty": "medium|hard",  # Based on subject
    "domain": "multitask_understanding",
    "source": "MMLU",
    "subject": "abstract_algebra",
    "answer_index": 1
  }
}
```

### Difficulty Estimation Rules

**GPQA & MATH:**
- GPQA: Always "hard" (graduate-level questions)
- MATH: "medium" for Level 1-2, "hard" for Level 3+

**HumanEval:**
- Always "medium" (code generation requires reasoning)

**MMLU:**
- "hard" for technical subjects (algebra, physics, CS, law, etc.)
- "medium" for general subjects

**DROP:**
- Always "easy" (reading comprehension, straightforward)

---

## Sampling Strategy

### Parameters
- **Default:** 100 examples per benchmark
- **Total:** ~500 examples for validation
- **Stratified:** Proportional sampling by difficulty level

### Why 100 per Benchmark?

**Rationale:**
1. **Balanced:** Equal representation across domains
2. **Efficient:** ~500 examples processable in 10-15 minutes
3. **Comprehensive:** Covers diverse capabilities (reading, math, code, science, multitask)
4. **Statistical:** Large enough for reliable metrics

**Special Cases:**
- **HumanEval:** Only 164 total examples (sample all 100)
- **MGSM:** Excluded (empty format in dataset)

### Stratification Details

For each benchmark:
1. Group examples by difficulty (easy/medium/hard)
2. Calculate proportion for each difficulty
3. Sample proportionally from each group
4. Ensure minimum 1 example per difficulty level

**Example (MATH benchmark):**
- Total: 1,319 examples
- Easy (Level 1-2): 30% → 30 examples
- Hard (Level 3+): 70% → 70 examples
- Sampled: 100 total

---

## Validation Test Set Statistics

### Final Composition (100 per benchmark)

**By Source:**
```
DROP:        100 examples (reading comprehension)
GPQA:        100 examples (science Q&A)
HumanEval:   100 examples (code generation)
MATH:        100 examples (mathematical reasoning)
MMLU:        100 examples (multitask understanding)
─────────────────────────────────────────────────
Total:       500 examples
```

**By Difficulty:**
```
Easy:        100 examples (20%)
Medium:      179 examples (36%)
Hard:        221 examples (44%)
─────────────────────────────────────────────────
Total:       500 examples
```

**By Domain:**
```
reading_comprehension:      100 examples
science_qa:                 100 examples
code_generation:            100 examples
mathematics:                100 examples
multitask_understanding:    100 examples
─────────────────────────────────────────────────
Total:                      500 examples
```

### Why This Distribution?

**Difficulty Distribution:**
- **Easy (20%):** Baseline performance check
- **Medium (36%):** Core capability validation
- **Hard (44%):** Generalization stress test

**Rationale:** ANIL-MAML is trained for few-shot adaptation. Hard examples test its ability to generalize to challenging, unseen tasks.

**Domain Balance:**
- Equal representation across 5 distinct capability areas
- Tests breadth of generalization (not just one domain)
- Matches few-shot learning goal: adapt to diverse tasks

---

## Implementation

### Scripts Created

#### 1. `convert_benchmarks_to_test.py`
**Purpose:** Convert benchmark datasets to validation format

**Features:**
- Parse 6 benchmark formats (DROP, GPQA, HumanEval, MATH, MGSM, MMLU)
- Standardize to unified schema
- Stratified sampling by difficulty
- Rich progress tracking

**Usage:**
```bash
# Convert with default settings (100 per benchmark)
python scripts/convert_benchmarks_to_test.py

# Custom sample size
python scripts/convert_benchmarks_to_test.py --samples-per-benchmark 50

# All examples (no sampling)
python scripts/convert_benchmarks_to_test.py --samples-per-benchmark -1

# Custom paths
python scripts/convert_benchmarks_to_test.py \
    --input-dir data/benchmarks \
    --output-file data/benchmarks/validation_test.jsonl
```

**Output:** `data/benchmarks/validation_test.jsonl` (500 examples)

#### 2. Updated `vastai_validate_and_merge.sh`
**Changes:**
- Removed test set extraction from training data (Step 1)
- Use `validation_test.jsonl` instead
- Updated step numbers (5 steps → 4 steps)
- Added check for test file existence

**New Workflow:**
1. ✅ Validate LoRA model on benchmarks
2. ✅ Merge LoRA weights
3. ✅ Validate merged model
4. ✅ Package results

#### 3. Updated `PHASE1D_VALIDATION_MERGE_GUIDE.md`
**Changes:**
- Added explanation of benchmark-based approach
- Updated prerequisites (benchmark datasets instead of training data)
- Added Step 3: Convert benchmarks before running workflow
- Updated expected outputs (500 examples, 5 benchmarks)

---

## Validation Process (On Vast.ai)

### Step 1: Convert Benchmarks (Local or Vast.ai)

```bash
cd /workspace
source venv/bin/activate

# Convert benchmarks to validation format
python scripts/convert_benchmarks_to_test.py --samples-per-benchmark 100
```

**Output:**
```
═══════════════════════════════════════════════════════════════
   Benchmark to Validation Test Converter   
═══════════════════════════════════════════════════════════════

  DROP: Sampled 100 from 9535 examples
  GPQA: Sampled 100 from 1319 examples
  HumanEval: Sampled 100 from 164 examples
  MATH: Sampled 100 from 1319 examples
  MGSM: Using all 0 examples (< 100 requested)
  MMLU: Sampled 100 from 14042 examples

Writing 500 examples to data/benchmarks/validation_test.jsonl

✓ Successfully created validation test set
✓ Total examples: 500
```

### Step 2: Run Validation Workflow

```bash
chmod +x scripts/vastai_validate_and_merge.sh
bash scripts/vastai_validate_and_merge.sh
```

**What Happens:**
1. Load LoRA model (base + adapter)
2. Compute perplexity on 500 benchmark examples
3. Merge LoRA weights into base model (~7GB standalone)
4. Validate merged model on same benchmarks
5. Compare results (should be identical within 1e-4)
6. Package for download

### Expected Results

**Pre-Merge (LoRA Model):**
```
Loss:       0.02-0.05 (expected)
Perplexity: 2.5-5.0 (expected)
Examples:   500 (5 benchmarks)
```

**Post-Merge (Merged Model):**
```
Loss:       0.02-0.05 (identical to LoRA)
Perplexity: 2.5-5.0 (identical to LoRA)
Examples:   500 (same benchmarks)
```

**Comparison:**
```
Loss Δ:     < 1e-4 (negligible)
PPL Δ:      < 1e-4 (negligible)
Status:     ✅ IDENTICAL (merge successful)
```

---

## Advantages of Benchmark-Based Validation

### 1. True Generalization Test
- **Independent data:** Never seen during training
- **Real test:** Measures ability to generalize, not memorize
- **Uncontaminated:** No train/test overlap

### 2. Standard Benchmarks
- **Comparable:** Can compare with published results
- **Diverse:** Covers multiple capability domains
- **Established:** Well-studied, high-quality datasets

### 3. Comprehensive Coverage
- **5 domains:** Reading, science, code, math, multitask
- **3 difficulty levels:** Easy, medium, hard
- **500 examples:** Statistically significant sample

### 4. Few-Shot Learning Alignment
- **ANIL-MAML Goal:** Adapt to diverse tasks with few examples
- **Benchmark Diversity:** Tests adaptation across domains
- **Difficulty Range:** Tests adaptation to varying complexity

---

## Comparison: Extracted vs Benchmark Validation

| Aspect | Extracted Test Set ❌ | Benchmark Test Set ✅ |
|--------|----------------------|----------------------|
| **Data Source** | Training data subset | Independent benchmarks |
| **Contamination** | Contaminated (model trained on it) | Clean (never seen) |
| **What It Tests** | Training accuracy | True generalization |
| **Domain Diversity** | 8 domains (all from training) | 5 diverse domains (unseen) |
| **Difficulty** | Easy (194), Hard (8) | Easy (100), Med (179), Hard (221) |
| **Size** | 202 examples | 500 examples |
| **Comparability** | Not comparable (custom) | Comparable (standard benchmarks) |
| **Value** | Limited (overfitted) | High (real generalization) |

---

## Lessons Learned

### Critical Insight: Test Set Contamination

**Problem:**
- Easy to assume "stratified sampling from training data = held-out test set"
- Actually: Model was trained on ALL training data
- Result: No examples were truly "held out"

**Detection:**
- User asked: "didn't you create the test data from the training data"
- Follow-up: "But did we really hold out?"
- Answer: No, we didn't!

**Solution:**
- Use independent datasets for validation
- Never test on data seen during training
- Benchmark datasets provide clean, unseen test data

### Best Practice: Always Verify Test Set Independence

**Checklist:**
- [ ] Is test set truly held out? (not just sampled)
- [ ] Was it excluded from training? (check training data)
- [ ] Can we use independent benchmarks? (preferred)
- [ ] What does this test measure? (accuracy vs generalization)

---

## Files Modified/Created

### Created Files
1. **scripts/convert_benchmarks_to_test.py** (600+ lines)
   - Converts 6 benchmark formats to unified validation format
   - Stratified sampling by difficulty
   - Rich progress tracking and statistics

### Updated Files
1. **scripts/vastai_validate_and_merge.sh**
   - Removed test set extraction (Step 1)
   - Use benchmark test set instead
   - Updated step numbers (5→4)

2. **docs/PHASE1D_VALIDATION_MERGE_GUIDE.md**
   - Added benchmark-based approach explanation
   - Updated prerequisites
   - Added benchmark conversion step

### Created Documentation
1. **docs/PHASE1D_BENCHMARK_VALIDATION.md** (this file)
   - Complete explanation of benchmark strategy
   - Conversion details for each benchmark
   - Comparison with extracted approach
   - Lessons learned

---

## Next Steps

### Immediate (On Vast.ai)
1. ✅ Upload `convert_benchmarks_to_test.py` to Vast.ai
2. ✅ Run benchmark conversion (creates validation_test.jsonl)
3. ⏳ Run validation workflow (LoRA model on benchmarks)
4. ⏳ Merge LoRA weights
5. ⏳ Validate merged model
6. ⏳ Download results and merged model

### Documentation
1. ⏳ Update technical_specification.md with validation results
2. ⏳ Document benchmark performance (perplexity by domain)
3. ⏳ Add example generations from each benchmark

### Future Phases
1. ⏸️ Use same benchmarks for Phase 2 validation (compression)
2. ⏸️ Use same benchmarks for Phase 3 validation (modifiers)
3. ⏸️ Track performance across phases (degradation analysis)

---

## Success Criteria

### Validation Success
- [ ] LoRA model perplexity < 10 on benchmarks
- [ ] Merged model perplexity identical to LoRA (Δ < 1e-4)
- [ ] Generation quality reasonable across all domains
- [ ] No errors or crashes during validation

### Merge Success
- [ ] Merged model size ~7GB (base model size)
- [ ] Post-merge validation identical to pre-merge
- [ ] Forward pass works correctly
- [ ] Can load and generate with merged model

---

**Conclusion:** Benchmark-based validation provides a true test of generalization ability. By using independent datasets (DROP, GPQA, HumanEval, MATH, MMLU), we avoid test set contamination and measure real performance on unseen data. This approach is more rigorous, comparable, and aligned with few-shot learning goals.
