# .adr/002-always-use-batch-api.md

## Status: ACCEPTED

## Context
Groq and OpenAI offer batch API with 50% discount.
Immediate API is 2× more expensive.

## Decision
ALWAYS use batch API for data generation.
NEVER use immediate API unless real-time required.

## Cost Impact
With batch: $263-288 (Phase 1)
Without batch: $526-576 (Phase 1)
Savings: $263-288 per phase

## Code Pattern
```python
# CORRECT:
batch = await client.batches.create(
    input_file=data,
    completion_window="24h"
)

# WRONG:
response = await client.completions.create(...)  # 2× cost!
```