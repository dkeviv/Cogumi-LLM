# Technical Specification: Phase 1C EOS+CoT Gold Standard Corrections

**Context:**
- Phase 1C of Cogumi-LLM pipeline: Targeted Distillation with EOS and Chain-of-Thought (CoT) reasoning.
- Step 2: Generate gold standard corrections for 2,138 failing examples using GPT-4.1 internal LLM.

---

## Data Sources
- Input: `Phase1C_Targeted_Distillation/data/phase1c_failures_for_gpt4.jsonl` (2,138 failing examples)
- Output: `Phase1C_Targeted_Distillation/data/phase1c_gpt4_corrections_cot.jsonl` (2,138 gold standard corrections)

## Correction Format
- Each example contains:
  - `<thinking>` block: DRAFT, CRITIQUE, REVISED reasoning
  - `<answer>` block: Final polished answer
  - `<|end_of_text|>` EOS token
- All corrections follow integrated CoT format as specified in pipeline instructions.

## Algorithm & Methodology
- For each failing example:
  1. Analyze the failure pattern (off-topic, keyword list, empty, wrong entity, etc.)
  2. Generate a DRAFT correction based on the reference and context
  3. Critique the draft, identifying specific flaws and missing elements
  4. Revise the answer to address critique and match gold standard quality
  5. Compose final answer in `<answer>` block, append EOS token
- All reasoning and answers generated using GPT-4.1 internal LLM (no heuristics)
- Adaptive max_tokens calculated per example (metadata preserved)

## Validation Steps
- File structure: JSONL, one example per line
- Each entry contains required fields and blocks
- EOS token present in every answer
- Sample outputs reviewed for format and quality
- 100% coverage of failing examples

## Implementation Notes
- No external API calls; all corrections generated internally
- File saved to correct location for downstream merging
- Checklist and documentation updated after completion

## Next Steps
- Merge passing and corrected examples for unified training set
- Retrain base model with EOS+CoT dataset

**Last Updated:** November 10, 2025
