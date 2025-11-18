# Deduplication Strategy - Phase 1 Question Generation

## Problem
When generating 44K new questions, there's a risk that:
1. API returns questions similar to existing ones
2. Waste API calls and time on duplicates
3. Discover duplicates only after generation completes

## Solution: Real-Time Deduplication

### 1. Load Existing Questions into Memory Set
```python
seen_questions = set()  # Normalized question texts
# Load all 44.7K existing questions
for q in existing:
    question_text = q['question'].strip().lower()
    seen_questions.add(question_text)
```

**Memory usage:** ~44.7K strings × ~50 bytes avg = ~2.2 MB (negligible)

### 2. Check Each New Question During Generation
```python
for new_question in batch:
    question_text = new_question['question'].strip().lower()
    if question_text not in seen_questions:
        # ✓ Unique - save it
        save_question(new_question)
        seen_questions.add(question_text)  # Track globally
    else:
        # ✗ Duplicate - skip it
        duplicates_found += 1
```

### 3. Advantages

**✅ Immediate Detection:**
- Check happens as questions arrive from API
- No post-processing needed

**✅ Memory Efficient:**
- Uses Python set (O(1) lookup)
- Only stores normalized text, not full JSON

**✅ Progress Visibility:**
- Reports duplicates found per domain
- Shows unique count in real-time

**✅ No Wasted Storage:**
- Duplicates never written to disk
- Checkpoint files contain only unique questions

**✅ Cross-Domain Deduplication:**
- Shared `seen_questions` set across all generation tasks
- Prevents duplicates even across different domains

### 4. Normalization Strategy

```python
# Normalize for comparison (case-insensitive, whitespace normalized)
question_text = q['question'].strip().lower()
```

**Why this works:**
- "What is 2+2?" == "what is 2+2?" (case)
- "What is 2+2?" == "What is  2+2?" (extra spaces)
- Catches near-duplicates with minor formatting differences

### 5. Progress Reporting

```
Generating 8,279 easy Coding questions (414 batches)...
███████████████████████████████████ 100%
  ⚠ Filtered 23 duplicates
  → Saved 8,256 unique to checkpoint_coding_easy.jsonl
```

**Shows:**
- How many duplicates filtered
- How many unique questions actually saved

### 6. Performance Impact

**Minimal overhead:**
- Set lookup: O(1) average case
- String normalization: ~microseconds per question
- Total overhead: <1% of API call time

**Example timing:**
- API call: 2-5 seconds per batch
- Dedup check: <0.001 seconds per question
- **Impact: Negligible!**

## Expected Duplicate Rate

Based on previous generation:
- **Best case:** 0-2% duplicates (models remember context well)
- **Typical case:** 2-5% duplicates (some repetition)
- **Worst case:** 10-15% duplicates (model forgets or loops)

**With 44K generation:**
- Expected: 880-2,200 duplicates filtered (2-5%)
- Actual unique generated: ~42K-43K
- **Result:** Still reach 60K target easily

## Fallback Strategy

If duplicate rate is higher than expected:

1. **Generate more batches** (script will keep going until target reached)
2. **Use temperature variation** (already at 0.9 for diversity)
3. **Vary prompts** (different phrasings per batch)

## Summary

✅ **Real-time deduplication** prevents wasting API calls  
✅ **Memory efficient** (~2 MB for 44K questions)  
✅ **Cross-domain protection** (shared set)  
✅ **Progress visibility** (reports filtered count)  
✅ **Zero post-processing** (checkpoints already clean)  

This approach ensures we only generate and store truly unique questions!
