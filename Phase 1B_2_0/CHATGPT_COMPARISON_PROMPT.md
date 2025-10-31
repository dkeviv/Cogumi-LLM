# ChatGPT Prompt for Phase 1B Step 3 Comparison

## Instructions for You:
1. Upload these two files to ChatGPT:
   - `test_dataset_20k.jsonl` (27MB - reference answers)
   - `model_outputs_20k.jsonl` (71MB - model generations)
2. Copy-paste the prompt below
3. Wait for ChatGPT to analyze and return results

---

## PROMPT FOR CHATGPT:

I need you to compare model outputs against reference answers to identify failures.

**CONTEXT:**
- I trained a Llama-3.1-8B model on 600K examples (Phase 1A)
- This model should achieve 75-82% GPT-4 performance
- I'm testing it on 20,000 diverse examples to find weaknesses
- Goal: Identify real failures (expected: 24-28% failure rate, ~5,000-6,000 failures)
- Purpose: These failures will be clustered to guide Phase 1C targeted training

**FILES:**
- `test_dataset_20k.jsonl`: Contains `instruction`, `response` (reference answer), `category`, `quality_score`
- `model_outputs_20k.jsonl`: Contains `id`, `instruction`, `reference`, `model_output`, `category`

**YOUR TASK:**
Compare each model output (20,000 total) against its reference answer and determine PASS or FAIL.

**EVALUATION CRITERIA (in priority order):**

1. **CORRECTNESS** (Most Important)
   - Is the core answer/information correct?
   - For code: Is the logic correct? Does it solve the problem?
   - For math: Is the final answer correct? Is the reasoning valid?
   - For QA: Does it answer the question accurately?
   - For reasoning: Is the conclusion sound and justified?

2. **COMPLETENESS**
   - Does the model address ALL key points from the instruction?
   - Are major aspects covered (even if phrased differently)?
   - Note: Missing 1-2 minor details is OK, missing major points is FAIL

3. **ACCURACY**
   - Are facts correct (no hallucinations)?
   - Is logic sound (no reasoning errors)?
   - Is syntax correct (for code)?
   - Are calculations correct (for math)?

4. **RELEVANCE**
   - Does the output stay on topic?
   - Does it follow the instruction's intent?
   - Is it addressing the right question?

**CRITICAL GUIDELINES - WHAT COUNTS AS PASS:**
- ✅ Verbose but correct outputs = PASS (detailed explanations are GOOD, not bad)
- ✅ Different phrasing = PASS (semantic equivalence matters, not exact wording)
- ✅ Minor style differences = PASS (formatting, verbosity, structure variations OK)
- ✅ Alternative valid approaches = PASS (for code/math, multiple solutions exist)
- ✅ Additional helpful context = PASS (extra explanation is not a failure)

**CRITICAL GUIDELINES - WHAT COUNTS AS FAIL:**
- ❌ Wrong answer or incorrect conclusion
- ❌ Missing major required information (not minor details)
- ❌ Factual errors or clear hallucinations
- ❌ Off-topic or completely misunderstood instruction
- ❌ Broken/incorrect code syntax
- ❌ Wrong mathematical calculations or final answer
- ❌ Nonsensical, repetitive garbage text
- ❌ Truncated output (ends mid-sentence incomplete)

**IMPORTANT NUANCES:**

For **instruction-following tasks** (like "rewrite in past perfect tense"):
- If instruction explicitly requires a specific format/tense/style, must follow it
- But if output is linguistically valid AND accomplishes the goal = PASS
- Example: "she wrote" vs "she had written" - if instruction says "past perfect", must use "had written"

For **open-ended tasks** (explanations, essays, reasoning):
- Focus on correctness of core ideas, not length or style
- Multiple valid answers exist - accept all that are correct

For **code tasks**:
- Different implementations are OK if logic is correct
- Focus on: Does it work? Is logic sound? 
- Minor syntax issues OK if intent is clear

For **math tasks**:
- Final answer must be correct
- Reasoning steps can vary if they lead to right answer
- Small notation differences OK

**OUTPUT FORMAT:**

Return a JSON file with this structure:
```json
{
  "summary": {
    "total_examples": 20000,
    "passes": <count>,
    "failures": <count>,
    "pass_rate": <percentage>,
    "failure_rate": <percentage>,
    "by_category": {
      "code": {"total": X, "passes": Y, "failures": Z, "pass_rate": P},
      "math": {"total": X, "passes": Y, "failures": Z, "pass_rate": P},
      "reasoning": {"total": X, "passes": Y, "failures": Z, "pass_rate": P},
      "qa": {"total": X, "passes": Y, "failures": Z, "pass_rate": P},
      "other": {"total": X, "passes": Y, "failures": Z, "pass_rate": P},
      "creative": {"total": X, "passes": Y, "failures": Z, "pass_rate": P}
    }
  },
  "all_results": [
    {
      "id": 0,
      "category": "other",
      "status": "FAIL",
      "reason": "Model incorrectly states 'she wrote' is past perfect when it should be 'she had written'",
      "confidence": 0.95
    },
    {
      "id": 1,
      "category": "code",
      "status": "PASS",
      "reason": "Correctly explains virtualization benefits and provides valid C# code",
      "confidence": 0.90
    },
    ...continue for all 20,000...
  ],
  "failure_analysis": {
    "common_patterns": [
      "Pattern 1: Instruction-following issues (X failures)",
      "Pattern 2: Incomplete code (Y failures)",
      "Pattern 3: Wrong math calculations (Z failures)",
      ...
    ],
    "most_problematic_categories": ["category1", "category2", ...],
    "sample_failures": [
      {"id": X, "issue": "brief description"},
      ...10 diverse failure examples...
    ]
  }
}
```

**VALIDATION CHECKS:**
- Expected pass rate: 72-76% (we need ~14,000-15,000 passes)
- If you get <60% or >90% pass rate, something is wrong - review criteria
- Code category typically has lowest pass rate
- Math category should have clear right/wrong answers

**PROCESS:**
1. Load both JSONL files
2. For each example (id 0-19,999):
   - Match by ID between test and model files
   - Compare model_output to reference
   - Apply evaluation criteria strictly but fairly
   - Record PASS/FAIL with reason and confidence
3. Generate summary statistics by category
4. Identify common failure patterns
5. Return complete JSON results

**EXPECTED TIMELINE:**
- This is 20,000 comparisons - may take 5-10 minutes
- Show progress every 2,000 examples
- If it's taking too long, you can:
  - Sample 2,000 random examples first to estimate
  - Then extrapolate to full 20K

Begin the comparison and return the complete JSON results.
