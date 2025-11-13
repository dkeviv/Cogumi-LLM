# Cogumi-LLM - Phase 1A Restart

**Goal:** Train Llama-3.1-8B-Instruct to beat GPT-4o-mini on all benchmarks

## 🎯 Target Performance

| Benchmark | GPT-4o-mini | Llama-3.1-8B | Target |
|-----------|-------------|--------------|--------|
| MMLU      | 82.0%       | 73.0%        | >82%   |
| GPQA      | 40.2%       | 32.8%        | >40%   |
| HumanEval | 87.2%       | 72.6%        | >87%   |
| MATH      | 70.2%       | 51.9%        | >70%   |
| MGSM      | 87.0%       | 68.9%        | >87%   |
| DROP      | 79.7%       | 59.5%        | >80%   |

## 📊 Strategy

**60K Balanced Dataset:**
- 10K per benchmark (MMLU, GPQA, HumanEval, MATH, MGSM, DROP)
- Each benchmark: 5K easy (GPT-4o-mini) + 5K hard (Claude Sonnet 4 + self-critique)
- No catastrophic forgetting - perfectly balanced distribution

**Cost:** ~$366 (dataset) + $180 (training) = **$546 total**

## 📁 Structure

```
data/
  raw/                    # Source benchmark data
  phase1_balanced/        # 60K curated examples
models/
  llama-3.1-8b-instruct/  # Base model
scripts/
  gpu_setup.sh            # GPU environment setup
configs/
  training.yaml           # Training configuration
docs/
  **Final Updated Pipeline.md  # Complete pipeline reference
```

## 🚀 Quick Start

```bash
# 1. Setup environment
./scripts/gpu_setup.sh

# 2. Download Llama-3.1-8B-Instruct
python3 scripts/download_llama31.py

# 3. Generate balanced dataset (60K examples)
python3 scripts/generate_balanced_dataset.py

# 4. Train with QLoRA
python3 scripts/train_phase1a.py

# 5. Benchmark all metrics
python3 scripts/benchmark_all.py
```

## 📚 Documentation

- **Master Pipeline:** `.github/**Final Updated Pipeline.md` (1204 lines)
- **Implementation Guide:** Coming soon
- **Benchmarking Guide:** Coming soon

## 🔄 Progress

- [ ] Environment setup
- [ ] Download Llama-3.1-8B-Instruct
- [ ] Generate 60K balanced dataset
- [ ] Train Phase 1A
- [ ] Benchmark and validate
