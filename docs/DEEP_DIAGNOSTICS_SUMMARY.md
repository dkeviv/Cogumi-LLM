# Deep Diagnostic Tools - Summary

## 🎯 Purpose

Created comprehensive diagnostic tools to understand **WHY** failures happen, not just count them.

## 📁 Files Created

### 1. **notebooks/Benchmark_Diagnostic_Analysis.ipynb**
Comprehensive notebook with:
- **Section 1**: Benchmark Results Analysis
  - Overall performance summary tables
  - 4 visualizations (scores, W/L/T, win rates, tie frequencies)
  - Deep dives: MATH (70% ties), CODE (competitive), CREATIVITY (small sample)
  - Overall patterns and insights
  
- **Section 1B**: Deep Diagnostics
  - Answer extraction validation with pattern testing
  - Consistency analysis framework
  - Benchmark file analysis (if downloaded from Vast.ai)
  - Generated diagnostic script creation
  - Results analysis when diagnostic_results.json available
  
- **Section 2**: Dataset Composition Analysis
  - Source distribution analysis
  - Category inference (math/code/reasoning/etc)
  - Training order analysis (first/middle/last 10K)
  - Visualization of distribution
  - Correlation with benchmark performance

### 2. **scripts/deep_diagnostic.py**
Executable diagnostic script for Vast.ai:
- Tests 10 MATH problems × 10 runs = 100 generations
- Tests 10 CODE problems × 10 runs = 100 generations
- Measures consistency rate (how often same answer appears)
- Measures unique answer count (variance)
- Saves detailed results to diagnostic_results.json
- Duration: ~30-60 minutes on GPU
- Purpose: Prove that inconsistency causes ties

## 🔬 What Gets Diagnosed

### Answer Extraction Issues
- Tests multiple regex patterns for math answers (####, \boxed{}, "answer:", etc.)
- Tests code block extraction (```python, def patterns)
- Tests multiple choice extraction (A/B/C/D patterns)
- Validates extraction on sample responses

### Consistency Patterns
- Same prompt → 10 different generations
- Measure how much answers vary
- Calculate consistency rate (% of runs with same answer)
- Count unique answers (2 = consistent, 10 = random)
- **Hypothesis**: Low consistency explains 70% ties in MATH

### Dataset Composition
- Count examples by source (Alpaca, MetaMathQA, CodeAlpaca, etc.)
- Infer categories (math/code/reasoning/knowledge)
- Analyze training order (early/middle/late)
- **Hypothesis**: Math early → forgotten (explains 41% vs 90% knowledge)

### Failure Mode Analysis
- Load benchmark JSON if available
- Extract individual test results
- Analyze TIE cases (why did judge call it a tie?)
- Analyze LOSS cases (what went wrong?)
- Look for patterns in failures

## 📊 Expected Insights

### If Consistency is LOW (<60%):
✅ **Confirms hypothesis**: Inconsistency causes ties
- Math 70% ties = model CAN solve but answers vary
- Solution: Greedy decoding + self-consistency training
- Expected: 41% → 70-80% after training

### If Consistency is HIGH (>80%):
⚠️ **Alternative explanation**: Formatting differences
- Model is consistent but format differs from expected
- Solution: Response format standardization
- May need fewer examples, just format training

### Dataset Order Insights:
- If math is early (first 10K) and knowledge is late (last 10K)
  - Supports catastrophic forgetting hypothesis
  - Solution: Interleaved training or periodic review
- If math is uniformly distributed
  - Forgetting is NOT the issue
  - Confirms inconsistency as primary cause

## 🚀 Usage Workflow

### Step 1: Benchmark Analysis (Local)
```bash
# Open notebook
jupyter lab notebooks/Benchmark_Diagnostic_Analysis.ipynb

# Run Section 1 cells (benchmarks analysis)
# No GPU needed, analyzes the results we already have
```

### Step 2: Run Deep Diagnostics (Vast.ai)
```bash
# Upload script to Vast.ai
scp scripts/deep_diagnostic.py vastai:/workspace/scripts/

# Run on GPU
python /workspace/scripts/deep_diagnostic.py

# Wait ~30-60 minutes
# Download results
scp vastai:/workspace/diagnostic_results.json ./
```

### Step 3: Analyze Diagnostics (Local)
```bash
# Place diagnostic_results.json in project root
# Run Section 1B cells in notebook
# Will show:
#   - Consistency rates per problem
#   - Unique answer counts
#   - Validation of hypothesis
#   - Actionable conclusions
```

### Step 4: Dataset Analysis (Local)
```bash
# Run Section 2 cells in notebook
# Analyzes data/phase1/public_500k_filtered.jsonl
# Shows:
#   - Composition by source/category
#   - Training order visualization
#   - Correlation with performance
```

## 💡 Key Questions Answered

1. **Why does MATH have 70% ties?**
   - Diagnostic will show consistency rate
   - If <60% → Answers vary each time → Explains ties
   - If >80% → Formatting issue, not consistency

2. **Why does CODE score 58% despite 1.5:1 win ratio?**
   - Diagnostic will show code consistency
   - If <60% → Code structure varies → Need training
   - If >80% → Edge cases, need more examples

3. **Is catastrophic forgetting happening?**
   - Dataset analysis will show training order
   - If math early + knowledge late → Yes
   - If uniform distribution → No

4. **What's the actual root cause?**
   - Diagnostics prove: Inconsistency vs capability vs forgetting
   - Guides next steps: Training vs format vs data ordering

## 🎯 Next Actions Based on Results

### If Diagnostic Shows Low Consistency:
1. ✅ Run self-consistency distillation (already created)
2. Generate with greedy decoding (do_sample=False)
3. Train on filtered deterministic examples
4. Expected: MATH 41% → 70-80%, CODE 58% → 75-80%

### If Diagnostic Shows High Consistency:
1. Focus on response format standardization
2. May need fewer examples (just format training)
3. Check extraction patterns more carefully
4. Consider prompt engineering

### If Dataset Shows Poor Math Representation:
1. Add more math examples to training
2. Use interleaved training (prevent forgetting)
3. Consider periodic review of early examples
4. May explain 41% performance directly

## 📈 Success Metrics

After running diagnostics, we'll know:
- **Consistency rate**: <60% = problem, >80% = good
- **Unique answers**: <3 = consistent, >5 = random
- **Dataset composition**: Math %, Code %, Order
- **Root cause**: Confirmed with data, not hypothesis

This enables **data-driven decisions** instead of guessing!
