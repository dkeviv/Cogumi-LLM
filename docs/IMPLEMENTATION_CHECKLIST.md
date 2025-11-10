# IMPLEMENTATION CHECKLIST

**Purpose:** Track all implementation tasks for Cogumi-LLM pipeline. Mark tasks as ✅ COMPLETE, ⏳ PENDING, or ❌ NOT STARTED. Update after every major change.

---

## Phase 1C: Targeted Distillation (EOS+CoT)

### Step 1: Prepare 10K Dataset with EOS
- ✅ Split 10K judged examples into passing (7,862) and failing (2,138) sets
- ✅ Add EOS token to passing examples
- ✅ Prepare failing examples for GPT-4.1 correction
- ✅ Validate output structure and counts

### Step 2: GPT-4.1 Gold Standard Corrections
- ✅ Generate gold standard corrections for 2,138 failing examples using GPT-4.1 internal LLM
- ✅ Use integrated Chain-of-Thought format (<thinking> DRAFT/CRITIQUE/REVISED </thinking>, <answer> ... <|end_of_text|>)
- ✅ Save to `Phase1C_Targeted_Distillation/data/phase1c_gpt4_corrections_cot.jsonl`
- ✅ Validate file format, sample outputs, EOS coverage

### Step 3: Merge and Validate Complete Dataset
- ⏳ Merge passing and corrected examples into unified training set
- ⏳ Validate merged dataset for structure, adaptive_max_tokens, EOS

### Step 4: Retrain with EOS+CoT Focus
- ⏳ Retrain base model on 10K EOS+CoT dataset
- ⏳ Monitor training, validate outputs, benchmark quality

---

**Last Updated:** November 10, 2025
