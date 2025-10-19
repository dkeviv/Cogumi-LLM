# .adr/001-use-async-everywhere.md

## Status: ACCEPTED

## Context
API calls to Groq, OpenAI, Together.ai can be parallelized.
Sequential calls would take 10× longer.

## Decision
ALL API calls MUST use async/await pattern.
ALL data generation MUST use asyncio.gather for parallelization.

## Consequences
- Faster data generation (3-4 hours vs 30+ hours)
- More complex code (async/await everywhere)
- Better cost efficiency (can use batch API)

## Code Pattern
```python
# CORRECT:
async def generate_data():
    tasks = [client.generate(p) for p in patterns]
    results = await asyncio.gather(*tasks)

# WRONG:
def generate_data():
    results = [client.generate(p) for p in patterns]  # Sequential!
```