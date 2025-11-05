# Phase 1A Quality Validation Guide

Before proceeding to Phase 2 compression, you **must** validate that your Phase 1A model meets quality targets. Here are three methods to confirm quality:

---

## ✅ **Method 1: Quick Quality Check (15 minutes)**

**Best for:** Fast sanity check before running full benchmarks

### What It Does
Tests the model on 10 hand-picked questions across different domains (math, code, reasoning, factual QA).

### How to Run

```bash
# After training completes and model is merged
cd /Users/vivekdurairaj/Projects/Cogumi-LLM

# If model on HuggingFace
python scripts/quick_quality_check.py \
  --model-path YOUR_USERNAME/cogumi-llm-phase1a \
  --num-samples 10 \
  --verbose

# If model is local
python scripts/quick_quality_check.py \
  --model-path models/llama-3.1-8b-phase1a-merged \
  --num-samples 10 \
  --verbose
```

### Expected Output

```
Overall Pass Rate: 8/10 (80%)
Overall Score: 75%

By Domain:
  math            80% (2/2 passed)
  code            75% (2/2 passed)
  reasoning       70% (1/2 passed)
  factual         90% (2/2 passed)
  problem_solving 80% (1/2 passed)

✓ EXCELLENT - Model quality looks good! Ready for Phase 2.
```

### Interpretation
- **≥70% overall score**: ✓ Good quality, proceed to Phase 2
- **50-70% overall score**: ⚠ Acceptable but review training logs
- **<50% overall score**: ✗ Poor quality, debug training

---

## ✅ **Method 2: Standard Benchmarks (2-4 hours) [RECOMMENDED]**

**Best for:** Accurate quality measurement against GPT-4

### What It Does
Runs your model on standardized benchmarks:
- **MMLU** (1000 samples): General knowledge across 57 subjects
- **GSM8K** (500 samples): Grade school math reasoning
- **HumanEval** (164 samples): Python code generation (simplified)

### How to Run

```bash
cd /Users/vivekdurairaj/Projects/Cogumi-LLM

# Run all benchmarks
python scripts/run_benchmarks.py \
  --model-path YOUR_USERNAME/cogumi-llm-phase1a \
  --benchmarks all \
  --output benchmark_results.json

# Or run specific benchmarks
python scripts/run_benchmarks.py \
  --model-path YOUR_USERNAME/cogumi-llm-phase1a \
  --benchmarks mmlu gsm8k \
  --num-samples-mmlu 1000 \
  --num-samples-gsm8k 500
```

### Expected Output

```
MMLU Benchmark (General Knowledge)
  Accuracy: 79.2% (792/1000)
  Target: 78-82%
  GPT-4: 80%
  vs GPT-4: 99.0%

GSM8K Benchmark (Math Reasoning)
  Accuracy: 87.4% (437/500)
  Target: 86-88%
  GPT-4: 75%
  vs GPT-4: 116.5%

OVERALL SUMMARY
  MMLU:    79.2% (Target: 78-82%, 99.0% of GPT-4)
  GSM8K:   87.4% (Target: 86-88%, 116.5% of GPT-4)

Overall: 92.3% of GPT-4 baseline

✓ EXCELLENT - Exceeds target! Ready for Phase 2.
```

### Target Metrics

| Benchmark | Target Range | GPT-4 Baseline | Min Required |
|-----------|--------------|----------------|--------------|
| **MMLU** | 78-82% | 80% | ≥75% |
| **GSM8K** | 86-88% | 75% | ≥80% |
| **Overall** | 90-93% GPT-4 | 100% | **≥87% GPT-4** |

### Interpretation
- **≥90% GPT-4**: ✓ Excellent! Exceeds targets
- **87-90% GPT-4**: ✓ Good - Meets minimum for Phase 2
- **80-87% GPT-4**: ⚠ Below target - review but may proceed
- **<80% GPT-4**: ✗ Too low - debug training before Phase 2

### Results File
Results saved to `benchmark_results.json`:
```json
{
  "mmlu": {
    "accuracy": 0.792,
    "vs_gpt4": "99.0%",
    "target_range": "78-82%"
  },
  "gsm8k": {
    "accuracy": 0.874,
    "vs_gpt4": "116.5%",
    "target_range": "86-88%"
  },
  "overall": {
    "vs_gpt4_percentage": 92.3,
    "target_range": "90-93%",
    "status": "✓ PASS"
  }
}
```

---

## ✅ **Method 3: Manual Testing in Colab (10 minutes)**

**Best for:** Quick qualitative assessment during/after training

### How to Run

Add this cell to your `Phase1A_Training_Colab.ipynb` after training:

```python
# Test the merged model
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load merged model
model = AutoModelForCausalLM.from_pretrained(
    "models/llama-3.1-8b-phase1a-merged",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("models/llama-3.1-8b-phase1a-merged")

# Test questions
test_prompts = [
    "Write a Python function to calculate factorial",
    "What is the capital of France?",
    "If a train travels 120 miles in 2 hours, what is its speed?",
    "Explain why the sky is blue in simple terms"
]

for prompt in test_prompts:
    inputs = tokenizer(
        f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        return_tensors="pt"
    ).to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nQ: {prompt}")
    print(f"A: {response.split('assistant')[-1].strip()}\n")
```

### What to Look For

✓ **Good Signs:**
- Answers are coherent and relevant
- Code is syntactically correct
- Math calculations are accurate
- Explanations are logical

✗ **Bad Signs:**
- Gibberish or repetitive text
- Code with syntax errors
- Wrong math answers
- Nonsensical explanations

---

## 🎯 **Decision Tree: Should You Proceed to Phase 2?**

```
Did you run benchmarks?
├─ YES
│  ├─ Overall ≥87% GPT-4?
│  │  ├─ YES → ✅ PROCEED TO PHASE 2
│  │  └─ NO
│  │     ├─ 80-87% GPT-4? → ⚠️ REVIEW (may proceed but expect lower Phase 2 quality)
│  │     └─ <80% GPT-4? → ❌ DEBUG TRAINING (check logs, data quality, config)
│  │
└─ NO
   ├─ Quick check passed (≥70%)?
   │  ├─ YES → ⏭️ RUN FULL BENCHMARKS (Method 2) before Phase 2
   │  └─ NO → ❌ DEBUG TRAINING
   │
   └─ Manual testing looks good?
      ├─ YES → ⏭️ RUN FULL BENCHMARKS (Method 2) before Phase 2
      └─ NO → ❌ DEBUG TRAINING
```

---

## 🔍 **Troubleshooting Poor Quality**

If your model scores **<87% GPT-4**, check these:

### 1. Check Training Logs
```bash
# Look at final loss
grep "loss" logs/training_phase1a.log | tail -20

# Expected: Final loss 0.5-0.7
# If loss >1.0: Training didn't converge properly
```

### 2. Verify Dataset Quality
```bash
# Check English purity
python scripts/verify_dataset.py \
  --dataset data/phase1/public_500k_filtered.jsonl \
  --sample-size 10000

# Expected: ≥99% English
# If <95%: Non-English data degraded quality
```

### 3. Check Model Size
```bash
# Merged model should be ~11GB (float16)
du -h models/llama-3.1-8b-phase1a-merged/

# If significantly smaller: Model may be corrupted
```

### 4. Review Training Config
- **Learning rate**: Should be 5e-6 (if too high: unstable; too low: underfit)
- **Epochs**: Should complete 3 epochs (if stopped early: underfit)
- **Batch size**: Should be 4-8 per GPU (if too small: unstable gradients)
- **LoRA rank**: Should be 64 (if too low: insufficient capacity)

### 5. Common Issues

| Issue | Symptoms | Fix |
|-------|----------|-----|
| **Training stopped early** | Loss >1.0, low accuracy | Resume training for more epochs |
| **Low-quality dataset** | Nonsensical responses | Re-filter data, increase quality threshold |
| **Wrong prompt format** | Model doesn't follow instructions | Check chat template matches LLAMA-3.1 format |
| **OOM during training** | Gradient checkpointing off | Enable gradient checkpointing, reduce batch size |
| **Model forgot base knowledge** | Zero factual accuracy | Learning rate too high, reduce to 3e-6 |

---

## 📊 **What Happens If Quality Is Low?**

### Option 1: Re-train with Adjustments
- Lower learning rate: 5e-6 → 3e-6
- Increase LoRA rank: 64 → 128
- Train longer: 3 epochs → 4 epochs
- Filter dataset more strictly: >7/10 → >8/10

### Option 2: Proceed Anyway (Not Recommended)
- Phase 2 compression will **preserve** existing quality
- If Phase 1 = 80% GPT-4 → Phase 2 = ~70% GPT-4 (87% retention)
- Final quality may be too low for production use

### Option 3: Switch to Smaller Dataset for Testing
- Use 50K samples for faster iteration
- Debug training pipeline
- Scale back to 640K once working

---

## ✅ **Validation Checklist Before Phase 2**

Before starting Phase 2 compression, ensure:

- [ ] Training completed all 3 epochs
- [ ] Final training loss ≤0.8 (ideally 0.5-0.7)
- [ ] LoRA adapters merged successfully
- [ ] Model generates coherent responses (manual test)
- [ ] Quick quality check ≥70% (if not running full benchmarks)
- [ ] **Full benchmarks ≥87% GPT-4** (recommended)
- [ ] Model uploaded to HuggingFace or saved to Drive
- [ ] Training logs saved for reference

---

## 🚀 **Next Steps After Validation**

Once quality is confirmed:

1. **Upload model to HuggingFace** (recommended)
   ```bash
   huggingface-cli login
   python scripts/upload_to_hf.py --model-path models/llama-3.1-8b-phase1a-merged
   ```

2. **Open Phase 2 Colab notebook**
   - Upload `notebooks/Phase2_Compression_Colab.ipynb` to Colab
   - Use HuggingFace model ID as input
   - Run all cells (8-10 hours)

3. **Expected Phase 2 Output**
   - Size: 480MB (97% reduction from 11GB)
   - Quality: 87-89% of Phase 1 quality
   - Example: If Phase 1 = 92% GPT-4 → Phase 2 = 80% GPT-4

---

## 📚 **Additional Resources**

- **Training notebook**: `notebooks/Phase1A_Training_Colab.ipynb`
- **Compression guide**: `docs/PHASE2_QUICK_START.md`
- **Full execution plan**: `docs/EXECUTION_PLAN.md`
- **Technical specs**: `docs/technical_specification.md`

---

**Remember:** Quality validation is critical! Don't skip benchmarks - they determine if Phase 2 compression will be successful. 🎯
