# vLLM Setup for Phase 1E

## Install vLLM

```bash
# Install vLLM (requires CUDA)
pip install vllm

# Verify installation
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
```

## Usage

```bash
# Full generation (53,597 examples, ~30-60 minutes on H200)
python scripts/phase1e_generate_teacher_outputs_vllm.py \
    --model_path models/phase1_maml_lora_v2/merged \
    --input_file data/phase1/answers/training_data_clean.jsonl \
    --output_file data/phase1e/teacher_outputs_53k.jsonl \
    --max_tokens 1024

# Test with 1000 examples first (~1-2 minutes)
python scripts/phase1e_generate_teacher_outputs_vllm.py \
    --model_path models/phase1_maml_lora_v2/merged \
    --input_file data/phase1/answers/training_data_clean.jsonl \
    --output_file data/phase1e/teacher_outputs_test.jsonl \
    --max_examples 1000 \
    --max_tokens 1024
```

## Performance Comparison

| Method | Batch Size | Time (53K examples) | Throughput | Speedup |
|--------|-----------|---------------------|------------|---------|
| **vLLM** | Auto-optimized | **30-60 min** | **50-100 ex/s** | **Baseline** |
| transformers (batch=200) | 200 | 3-4 hours | 4-5 ex/s | **5-10× slower** |
| transformers (batch=100) | 100 | 3-4 hours | 4-5 ex/s | **5-10× slower** |
| transformers (sequential) | 1 | 50+ hours | 0.3 ex/s | **50× slower** |

## Why vLLM is Faster

1. **Paged Attention** - Efficient KV cache management (no memory waste)
2. **Continuous Batching** - Process requests as they complete (no waiting for slowest)
3. **Optimized CUDA Kernels** - Hand-tuned for maximum GPU utilization
4. **Automatic Batch Sizing** - Dynamically adjusts batch size for optimal throughput

## Expected Output

```
⏳ Generating batch 1/54 (1000 examples)...
✅ Batch 1/54 - [1000 done, 87.3 ex/s]
⏳ Generating batch 2/54 (1000 examples)...
✅ Batch 2/54 - [2000 done, 92.1 ex/s]
...

Generation Complete!
Statistics:
  Total examples: 53597
  Generated: 53597
  Failed: 0
  Total time: 45.2 minutes
  Average throughput: 19.8 examples/second
  Total tokens generated: 25,450,123
```

## Troubleshooting

### OOM (Out of Memory)

```bash
# Reduce GPU memory utilization
--gpu_memory_utilization 0.8  # Default is 0.9
```

### Import Error

```bash
# Make sure CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall vLLM
pip uninstall vllm -y
pip install vllm
```

### Slower than Expected

Check GPU utilization:
```bash
watch -n 1 nvidia-smi
```

Should show 90-100% GPU utilization during generation.
