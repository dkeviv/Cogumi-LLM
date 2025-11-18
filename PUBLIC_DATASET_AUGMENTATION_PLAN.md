# Public Dataset Augmentation Plan

## 🎯 Objective
Augment our 47,442 synthetic questions with FREE public datasets to:
- Fill domain gaps (Common Sense, Instruction, Reading, Summarization)
- Add diversity and real-world examples
- Reach 60K total questions
- Maintain 60/40 token balance

## 📊 Current State

### Existing Synthetic Dataset
```
Total: 47,442 questions
- Easy: 46,775 (98.6%)
- Hard: 667 (1.4%)

Domain Distribution:
  Coding:           6,861
  Math:             6,861
  Tool Use:         6,861
  Reasoning:        6,861
  Reading:          4,999  ⚠️ GAP: 1,862
  Summarization:    4,999  ⚠️ GAP: 1,862
  Common Sense:     5,000  ⚠️ GAP: 1,861
  Instruction:      5,000  ⚠️ GAP: 1,861

Token Balance: 58.4% / 41.6% ✓
```

### Gap to 60K Target
- **Need:** 12,558 more questions
- **Focus:** Under-represented domains (4 domains need ~1,861 each)

## 📚 Proposed Public Datasets

### Domain: Coding (Target: +2,000)
| Dataset | Size | Questions | Cost | Notes |
|---------|------|-----------|------|-------|
| CodeSearchNet | 2M+ | 2,000 | $0 | Function documentation → "Write a function that..." |
| HumanEval | 164 | 164 | $0 | Hand-written programming problems |
| MBPP | 1,000 | 500 | $0 | Basic Python problems |

**Total Coding: 2,664 questions**

### Domain: Math (Target: +2,000)
| Dataset | Size | Questions | Cost | Notes |
|---------|------|-----------|------|-------|
| GSM8K | 8.5K | 2,000 | $0 | Grade school math word problems |
| MATH | 12.5K | 1,500 | $0 | Competition-level math |
| MetaMathQA | 395K | 500 | $0 | Augmented math problems |

**Total Math: 4,000 questions**

### Domain: Reasoning (Target: +2,000)
| Dataset | Size | Questions | Cost | Notes |
|---------|------|-----------|------|-------|
| MMLU | 15.9K | 2,000 | $0 | Multiple-choice across 57 subjects |
| ARC-Challenge | 2.6K | 1,500 | $0 | Science exam questions |
| HellaSwag | 70K | 500 | $0 | Commonsense reasoning about events |

**Total Reasoning: 4,000 questions**

### Domain: Reading (Target: +2,000) ⭐ PRIORITY
| Dataset | Size | Questions | Cost | Notes |
|---------|------|-----------|------|-------|
| SQuAD 2.0 | 150K | 2,000 | $0 | Reading comprehension questions |
| DROP | 96K | 1,500 | $0 | Discrete reasoning over paragraphs |
| CoQA | 127K | 500 | $0 | Conversational question answering |

**Total Reading: 4,000 questions**

### Domain: Tool Use (Target: +2,000)
| Dataset | Size | Questions | Cost | Notes |
|---------|------|-----------|------|-------|
| ToolBench | 16K+ | 1,500 | $0 | Real-world API usage |
| Gorilla API Bench | 1.6K | 1,600 | $0 | API call generation |

**Total Tool Use: 3,100 questions**

### Domain: Common Sense (Target: +2,000) ⭐ PRIORITY
| Dataset | Size | Questions | Cost | Notes |
|---------|------|-----------|------|-------|
| CommonsenseQA | 12K | 2,500 | $0 | Commonsense reasoning questions |
| PIQA | 21K | 2,500 | $0 | Physical commonsense reasoning |

**Total Common Sense: 5,000 questions**

### Domain: Summarization (Target: +2,000) ⭐ PRIORITY
| Dataset | Size | Questions | Cost | Notes |
|---------|------|-----------|------|-------|
| CNN/DailyMail | 312K | 2,500 | $0 | News article summarization |
| XSum | 227K | 2,500 | $0 | Single-sentence summaries |

**Total Summarization: 5,000 questions**

### Domain: Instruction (Target: +2,000) ⭐ PRIORITY
| Dataset | Size | Questions | Cost | Notes |
|---------|------|-----------|------|-------|
| Natural Instructions | 193K | 2,500 | $0 | Diverse instruction following |
| Alpaca | 52K | 5,000 | $0 | Instruction-following dataset |

**Total Instruction: 7,500 questions**

## 📈 Projected Results

### Total Public Dataset Questions: ~39,264

### After Augmentation (47,442 + 39,264 = 86,706)
```
Will need to balance down to 60K target:

Option 1: Keep all lacking domains, trim excess
  - Reading:        4,999 + 4,000 = 8,999 → 7,500 (trim 1,499)
  - Summarization:  4,999 + 5,000 = 9,999 → 7,500 (trim 2,499)
  - Common Sense:   5,000 + 5,000 = 10,000 → 7,500 (trim 2,500)
  - Instruction:    5,000 + 7,500 = 12,500 → 7,500 (trim 5,000)
  - Coding:         6,861 + 2,664 = 9,525 → 7,500 (trim 2,025)
  - Math:           6,861 + 4,000 = 10,861 → 7,500 (trim 3,361)
  - Tool Use:       6,861 + 3,100 = 9,961 → 7,500 (trim 2,461)
  - Reasoning:      6,861 + 4,000 = 10,861 → 7,500 (trim 3,361)

Final: 60,000 questions (7,500 per domain - perfectly balanced)
```

## 💡 Key Advantages

### 1. **Cost: $0** (vs $416+ for generating answers)
- All datasets are FREE on HuggingFace
- No API costs

### 2. **Quality: High**
- Professionally curated datasets
- Used for benchmarking major models
- Diverse, real-world examples

### 3. **Diversity: Maximum**
- 15+ different data sources
- Mix of synthetic + real-world
- Various question styles and formats

### 4. **Speed: Fast**
- Download in 10-15 minutes (parallel)
- No generation time needed
- Immediate availability

### 5. **Token Balance: Maintained**
- Public datasets typically "easy" difficulty
- Will maintain 60/40 token balance
- Can adjust difficulty labels if needed

## ⚠️ Considerations

### 1. **Format Differences**
- Public datasets have various formats
- Need conversion to our schema
- Some datasets already have answers (bonus!)

### 2. **Duplicate Risk**
- Must deduplicate against existing questions
- Use MinHash LSH as before
- Expected: 5-10% duplicates

### 3. **Difficulty Labeling**
- Most public datasets → "easy" difficulty
- Can analyze complexity and relabel some as "hard"
- Or keep all as "easy" (we have enough hard questions)

### 4. **Answer Generation**
- Public datasets may already have answers
- Can use existing answers (saves cost!)
- Or regenerate with our teachers for consistency

## 🚀 Implementation Plan

### Step 1: Download Public Datasets (30 minutes)
```bash
python scripts/phase1_augment_with_public_datasets.py
```
- Downloads 15+ datasets from HuggingFace
- Converts to our JSON format
- Real-time deduplication against existing
- Output: `questions_augmented_with_public.jsonl` (~86K questions)

### Step 2: Balance to 60K (5 minutes)
```bash
python scripts/phase1_balance_final_60k.py
```
- Balance all domains to 7,500 each
- Maintain token balance
- Random shuffle for batch mixing
- Output: `questions_final_60k.jsonl` (60,000 questions)

### Step 3: Validate (Optional, $9)
```bash
python scripts/phase2_validate_questions.py
```
- Quality check with GPT-4-mini
- Filter low-quality questions
- Can skip if we trust public datasets

### Step 4: Generate/Use Answers
**Option A: Use Existing Answers (FREE)**
- Many public datasets have answers
- Extract and format to our schema
- Cost: $0

**Option B: Generate with Teachers ($416)**
- Generate fresh answers with GPT-4o-mini + Claude
- Consistent quality and format
- Cost: $416

### Step 5: Train Models ($39)
- Same as before
- MAML training on 60K balanced dataset
- Cost: $39

## 💰 Cost Comparison

### Original Plan (Synthetic Only)
```
Questions:     47,442 (synthetic)
Validation:    $9
Answers:       $416 (GPT-4o-mini + Claude)
Training:      $39
TOTAL:         $464
```

### Augmented Plan (Synthetic + Public)
```
Questions:     60,000 (47K synthetic + 13K public)
Download:      $0
Validation:    $9 (optional, can skip for public data)
Answers:       $0 (use existing) OR $535 (regenerate all)
Training:      $39
TOTAL:         $39-48 (use existing) OR $574-583 (regenerate)
```

### Cost Savings Scenarios

**Scenario 1: Use Existing Answers**
- Cost: $39-48 (89-90% savings!)
- Quality: Good (vetted public datasets)
- Risk: Format inconsistency

**Scenario 2: Hybrid (existing + regenerate hard)**
- Cost: ~$200 (57% savings)
- Quality: Best (consistent format)
- Risk: Low

**Scenario 3: Regenerate All**
- Cost: $574-583 (23% more expensive)
- Quality: Highest (fully consistent)
- Risk: None

## 🎯 Recommendation

### **Use Augmented Plan + Hybrid Answers**

**Why:**
1. **Better dataset:** 60K vs 47K questions
2. **More diversity:** Synthetic + real-world
3. **Lower cost:** $200 vs $464 (57% savings)
4. **Balanced domains:** 7,500 per domain (perfect)
5. **Proven quality:** Benchmarked public datasets

**Approach:**
1. Download public datasets (15 mins, $0)
2. Balance to 60K (5 mins, $0)
3. Use existing answers for public data ($0)
4. Generate answers for synthetic hard questions only ($414 → $200)
5. Train models ($39)

**Total: ~$239 (48% savings vs original $464)**

## 📋 Next Steps

1. **Review this plan** - Confirm approach
2. **Run augmentation script** - Download datasets
3. **Balance to 60K** - Create final training set
4. **Decide on answers** - Use existing vs regenerate
5. **Proceed to training** - Phase 1 complete!

---

**Ready to proceed? This gives us a better dataset at lower cost!** 🚀
