# Phase 3: Domain Modifiers

**Duration:** 4 weeks  
**Cost:** $685  
**Status:** ⏳ Pending Phase 2 completion

## Overview
Create three specialized domain modifiers using cascaded 3-tier teacher strategy, each targeting specific weak areas of the base model.

## Modifier Strategy

### 3-Tier Cascaded Teaching
1. **Tier 1 (60-70%):** Free/cheap models handle easy cases
2. **Tier 2 (20-25%):** GPT-4o/DeepSeek for moderate difficulty
3. **Tier 3 (10-15%):** GPT-5 for hardest cases
- **Cost savings:** 61% vs single-teacher approach

### Modifier Pipeline (Reusable)
1. Test base model on 12K domain tasks → identify failures
2. Generate Tier 1 data for failures (9K examples)
3. Test Tier 1 → identify remaining failures
4. Generate Tier 2 data for remaining failures
5. Test Tier 2 → identify remaining failures
6. Generate Tier 3 data with GPT-5
7. Train LoRA adapter on combined data (rank 96-128)
8. Compress modifier via pruning (260MB → 40-48MB)

## Modifiers

### Code Modifier (Week 11-12, $200, 47MB)
**Performance:** 115-130% GPT-4 on coding tasks
- **Teachers:** Qwen-Coder-480B (Tier 1), DeepSeek-Coder (Tier 2), GPT-5 (Tier 3)
- **Benchmarks:** HumanEval, MBPP, LiveCodeBench
- **LoRA rank:** 128
- **Target domains:** Python, JavaScript, debugging, code review

### Reasoning Modifier (Week 12-13, $207, 48MB)
**Performance:** 100-108% GPT-4 on reasoning tasks
- **Teachers:** Llama-405B FREE (Tier 1), GPT-4o (Tier 2), GPT-5+COT (Tier 3)
- **Benchmarks:** MMLU, BBH, ARC, logical reasoning
- **LoRA rank:** 112
- **Target domains:** Logic, common sense, science reasoning

### Automation Modifier (Week 13-14, $170, 40MB)
**Performance:** 105-118% GPT-4 on automation tasks
- **Teachers:** Claude-3.5 (Tier 1), GPT-4o (Tier 2), GPT-5 (Tier 3)
- **Benchmarks:** Tool-use, API calls, workflow orchestration
- **LoRA rank:** 96
- **Target domains:** Task planning, API integration, workflows

## Scripts

### Testing & Analysis
- `test_domain.py` - Test base on domain benchmarks
- `test_tier1.py` - Validate Tier 1 data effectiveness
- `test_tier2.py` - Validate Tier 2 data effectiveness

### Data Generation
- `generate_tier1.py` - Generate data with cheap teachers
- `generate_tier2.py` - Generate data with mid-tier teachers
- `generate_tier3.py` - Generate data with GPT-5

### Training & Compression
- `train_modifier.py` - Train LoRA adapter (260MB)
- `compress_modifier.py` - Prune to 40-48MB
- `validate_modifier.py` - Benchmark against GPT-4

## System Architecture
```
Base Model (520MB, 89-91% GPT-4)
├── Code Modifier (47MB, 115-130% GPT-4)
├── Reasoning Modifier (48MB, 100-108% GPT-4)
└── Automation Modifier (40MB, 105-118% GPT-4)

Total: 655MB (520 + 135 modifiers)
Router: 13MB
Grand Total: 668MB
```

## Hot-Swappable Design
- Only base + 1 modifier loaded at a time
- Router switches modifiers based on query
- Memory footprint: 520MB + 48MB = 568MB max

## Expected Outcomes
- **3 specialized modifiers** (135MB total)
- **All beat GPT-4** in respective domains
- **Efficient routing** via calibrated confidence scores

## Next Steps
After Phase 3, proceed to **Phase 4: Router** to build intelligent routing system.

See `docs/EXECUTION_PLAN.md` for modifier-specific timelines.
