# Phase 4: Router System

**Duration:** 2 weeks  
**Cost:** $75  
**Status:** ⏳ Pending Phase 3 completion

## Overview
Build intelligent routing system that decides when to use base model vs specialized modifiers, with escalation detection for dissatisfied users.

## Components

### Phase 4A: Router Training (1 week, $45, 13MB)
**Goal:** 97% accuracy routing system
- Collect 35K labeled examples (query → correct model decision)
- Train 3-layer feedforward network (64 → 32 dims)
- Features: confidence scores, query embeddings, historical performance
- **Output:** 13MB router model

### Phase 4B: Escalation Detector (4 days, $30, 3MB)
**Goal:** 94% detection of dissatisfaction
- Detect phrases like "you don't understand", "that's wrong"
- Train on 6K dissatisfaction examples
- Distill BERT-base (110MB) → 3MB lightweight model
- **Output:** 3MB escalation detector

### Phase 4C: Threshold Optimization (2 days, $0)
**Goal:** Optimize base vs modifier decision boundary
- A/B test thresholds: 75%, 80%, 85%
- Evaluate on 5K queries
- Balance quality vs cost
- **Optimal:** 80% confidence threshold

## Architecture

### Routing Flow
```
User Query
    ↓
Router (13MB) → Confidence score (0-1)
    ↓
If confidence > 80%:
    → Base Model (520MB) → Response
Else:
    → Select Modifier (Code/Reasoning/Automation)
    → Load Modifier (40-48MB)
    → Modified Model → Response
    ↓
Escalation Detector (3MB)
    ↓
If dissatisfaction detected:
    → Switch to alternative modifier
    → Or escalate to human
```

### Session Memory
- Track last 5 queries + responses per user
- Remember successful/failed modifier choices
- Improve routing accuracy over conversation
- Stored in SQLite (lightweight persistence)

## Scripts

### Router Training
- `collect_router_data.py` - Run base on 35K queries, label correctness
- `train_router.py` - Train 3-layer feedforward network
- `validate_router.py` - Test on 5K holdout set

### Escalation Detection
- `train_escalation.py` - Train BERT on dissatisfaction examples
- `distill_escalation.py` - Distill 110MB → 3MB

### Threshold Optimization
- `optimize_threshold.py` - A/B test 75%, 80%, 85%
- `analyze_tradeoffs.py` - Quality vs cost analysis

### Session Management
- `session_memory.py` - SQLite-based session tracking
- `update_routing.py` - Learn from session history

## Routing Strategies

### Default Mode (80% threshold)
- **Base usage:** 45-55% of queries
- **Code modifier:** 20-25%
- **Reasoning modifier:** 15-20%
- **Automation modifier:** 10-15%

### Turbo Mode (85% threshold)
- Prefer base model for speed
- Base usage: 60-70%
- Lower cost, slightly lower quality

### Max Quality Mode (70% threshold)
- Prefer modifiers for quality
- Base usage: 30-40%
- Higher cost, maximum quality

## Expected Outcomes
- **97% routing accuracy**
- **94% escalation detection**
- **16MB total (router + escalation)**
- **Adaptive learning** from user feedback

## Performance Metrics
- **Routing latency:** <5ms on M4 Pro Mac
- **Memory overhead:** 16MB
- **Correct modifier selection:** 97%
- **False escalation rate:** <6%

## Next Steps
After Phase 4, proceed to **Phase 5: Deployment** to launch on HuggingFace.

See `docs/EXECUTION_PLAN.md` for detailed implementation steps.
