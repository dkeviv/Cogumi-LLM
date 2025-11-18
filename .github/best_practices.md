One-Pager: Best Practices for Multi-Teacher + Easy/Hard Training (Single SFT Run)

1. Dataset Schema
   Use a consistent, normalized, teacher-agnostic structure:

{
  "input": "...",
  "difficulty": "EASY | HARD",
  "draft": "string | null",
  "thinking": "string | null",
  "final": "string",
  "teacher": "gpt4o | claude | qwen | etc"
}

2. Mode Tags in the Prompt
   Explicit mode signals prevent reasoning bleed and stabilize behavior.

HARD example:
<difficulty: HARD>
`<draft>`
{draft}
`</draft>`

<thinking>
{thinking}
</thinking>

<final>
{final}
</final>

EASY example:
<difficulty: EASY>
`<final>`
{final}
`</final>`

3. Multi-Teacher Normalization
   Normalize all teacher outputs into:

- one CoT structure
- one draft style
- one final answer style

4. Weighted Token Sampling
   Hard examples = 300–1500 tokens.
   Easy examples = 5–30 tokens.
   Use weighted sampling so batches contain ~60% EASY tokens and 40% HARD tokens.
5. Loss Masking
   EASY → loss only on `<final>`
   HARD → loss on `<draft>`, `<thinking>`, `<final>`
6. Create training data with meta data. For easy, no thinking data , for hard have thinking data. Need to have explicit EOS . Everything should be explicit
7. "prompt": "...",
   "draft":".."
   "thinking":"..."
   "response": "...",
   "metadata": {
   "harm_category": "none",
   "task_type": "coding",
   "complexity": "low",
   "requires_reasoning": false
   }
   }
