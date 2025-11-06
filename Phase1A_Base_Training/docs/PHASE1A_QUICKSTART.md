# Phase 1A Training - Quick Start Guide

## Overview
Train Llama-3.1-8B-Instruct on 640,637 curated examples using QLoRA on Vast.ai H100 GPU.

**Duration**: ~3 hours  
**Cost**: ~$10 (3 hours × $3/hour)  
**GPU**: H100 80GB HBM3 (CUDA 12.8)  
**Framework**: HuggingFace Transformers + TRL + Unsloth (NOT Axolotl)  

---

## Setup Steps

### 1. Rent H100 GPU on Vast.ai

1. Visit: https://vast.ai/
2. Sign up and add payment method
3. Search for: `cuda_vers >= 12.8 gpu_name = H100 disk_space >= 100`
4. Sort by **$/hr** (lowest first)
5. Select instance (~$2.50-3.50/hour)
6. Click **Rent** → Select **On-demand**
7. Choose **Jupyter Notebook** template

### 2. Upload Notebook & Dataset

1. Once instance starts, click **Open** → JupyterLab
2. Upload `notebooks/H100_Training_Clean.ipynb`
3. Create folder: `/data/Cogumi-LLM/`
4. Upload dataset: `public_500k_filtered.jsonl` to `/data/Cogumi-LLM/`

### 3. Get HuggingFace Token

1. Visit: https://huggingface.co/settings/tokens
2. Click **New token** → Create read token
3. Accept Llama license: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
4. Copy token for Cell 10 in notebook

### 4. Run Training (16 Cells)

**Cell-by-cell execution** (recommended):

1. **Cell 1**: Review golden dependency table
2. **Cells 2-8**: Installation & verification (~5-10 min)
3. **Cell 9**: Verify H100 detected
4. **Cell 10**: HuggingFace authentication
5. **Cell 11**: Upload dataset (if not done manually)
6. **Cell 12**: Verify dataset loaded (640K examples)
7. **Cell 13**: Check disk space (>50GB free needed)
8. **Cell 14**: Create `train.py` with optimizations
9. **Cell 15**: Review training parameters
10. **Cell 16**: Start training (~3 hours)

**Or run all**: Kernel → Restart Kernel and Run All Cells

### 5. Monitor Training

Training runs for ~3 hours with live output in Cell 16:
```
{'loss': 2.421, 'learning_rate': 5e-06, 'epoch': 0.05}
  5%|▌         | 1500/30000 [05:00<2:55:00,  5.00 it/s]
GPU: 99% | Mem: 40.2GB/80GB | Temp: 68°C
```

**Expected**:
- 5-12 it/s (variable by example length)
- 99-100% GPU utilization
- ~40GB memory usage
- Training completes in ~3 hours

---

## Files & Directories

### Local Files (Before Upload)
- `notebooks/H100_Training_Clean.ipynb` - Production notebook (16 cells)
- `data/phase1/public_500k_filtered.jsonl` - Dataset (640K examples, 870MB)

### Vast.ai Files (Created During Setup)
- `/workspace/golden-venv/` - Virtual environment with golden dependencies
- `/workspace/golden_dynamic_setup_full.sh` - Dependency installation script
- `/data/Cogumi-LLM/train.py` - Generated training script (Cell 14)
- `/data/Cogumi-LLM/public_500k_filtered.jsonl` - Uploaded dataset

### Training Output Files
- `/data/Cogumi-LLM/checkpoints/` - Saved every 1000 steps
- `/data/Cogumi-LLM/checkpoints/checkpoint-30000/` - Final checkpoint
- `/data/Cogumi-LLM/checkpoints/checkpoint-30000/adapter_model.safetensors` - LoRA weights (~400MB)
- Training logs in Cell 16 output

---

## Training Monitoring

### Live Output (Cell 16)
Training shows real-time progress:
```
{'loss': 2.421, 'grad_norm': 1.234, 'learning_rate': 5e-06, 'epoch': 0.05}
  5%|▌         | 1500/30000 [05:00<2:55:00,  5.00 it/s]
{'loss': 2.156, 'grad_norm': 0.987, 'learning_rate': 4.95e-06, 'epoch': 0.10}
 10%|█         | 3000/30000 [10:00<2:45:00,  5.45 it/s]
```

### Key Metrics to Watch
- **it/s (iterations/sec)**: Should be 5-12 (variable by example length)
- **GPU Utilization**: Should be 99-100%
- **Memory**: Should be ~40GB / 80GB
- **Loss**: Should decrease steadily (2.8 → 1.2)
- **Temperature**: Should be <80°C

### Expected Loss Curve
- **Epoch 1** (0-10K steps): Loss 2.8 → 1.6 (~1 hour)
- **Epoch 2** (10K-20K steps): Loss 1.6 → 1.3 (~1 hour)
- **Epoch 3** (20K-30K steps): Loss 1.3 → 1.2 (~1 hour)
- **Target**: Final loss 1.18-1.22

---

## Troubleshooting

### Problem: "CUDA version mismatch"
**Solution**: Must use PyTorch 2.8.0+cu128 (golden dependency set):
```bash
# Cell 2 installs correct version
pip install torch==2.8.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
```

### Problem: "Out of Memory (OOM)"
**Solution**: Reduce batch size in Cell 14 train.py:
```python
per_device_train_batch_size=16,  # Changed from 32
gradient_accumulation_steps=4,   # Changed from 2
```

### Problem: "Training very slow (<1 it/s)"
**Solution**: Missing FastLanguageModel.for_training() call:
```python
# CRITICAL: Add this line after get_peft_model()
model = FastLanguageModel.for_training(model)  # Enables Flash Attention 2
```
Without this: 0.5 it/s @ 35% GPU  
With this: 5-12 it/s @ 100% GPU (10-24× speedup)

### Problem: "Dataset not found"
**Solution**: Check dataset path in Cell 14:
```python
# Must match your upload location
dataset = load_dataset("json", 
    data_files="/data/Cogumi-LLM/public_500k_filtered.jsonl", 
    split="train")
```

### Problem: "Dependency conflicts"
**Solution**: Use golden dependency set from Cell 1:
- PyTorch 2.8.0+cu128
- bitsandbytes 0.48.1
- xformers 0.0.32.post2
- transformers 4.57.1
- Unsloth 2025.10.8

Do NOT upgrade packages without testing.

---

## Expected Timeline

| Phase | Steps | Duration | Status Check |
|-------|-------|----------|--------------|
| Setup (Cells 1-13) | - | 5-10 min | Dependencies installed, GPU verified |
| Epoch 1 | 0-10K | ~1 hour | Loss 2.8→1.6, 5-12 it/s |
| Epoch 2 | 10K-20K | ~1 hour | Loss 1.6→1.3, 99-100% GPU |
| Epoch 3 | 20K-30K | ~1 hour | Loss 1.3→1.2, consistent speed |
| **Total** | **30K** | **~3 hours** | **LoRA adapter saved (400MB)** |

**Note**: it/s varies by example length (5-12 range is normal with packing enabled)

---

## After Training Completes

### 1. Download LoRA Adapter
```bash
# In JupyterLab terminal or new notebook cell
cd /data/Cogumi-LLM/checkpoints/checkpoint-30000
tar -czf adapter.tar.gz adapter_*
# Download via JupyterLab file browser (right-click → Download)
```

### 2. Test Model (Optional)
Add test cell to notebook:
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/data/Cogumi-LLM/checkpoints/checkpoint-30000",
    max_seq_length=1024,
    load_in_4bit=True,
)

prompt = "What is the capital of France?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

### 3. Run Phase 1B Benchmarking
Evaluate model quality against GPT-4:
```bash
# Download benchmark script
bash scripts/run_phase1b_benchmark.sh YOUR_OPENAI_KEY
```

Expected score: **85-90%** (compared to GPT-4 baseline)

### 4. Decide Next Phase
- **Score ≥90%**: Skip Phase 1C, proceed to Phase 2 (Compression)
- **Score 85-90%**: Optional Phase 1C (10K targeted examples)
- **Score <85%**: Required Phase 1C (40K targeted examples from GPT-5)

### 5. Phase 2: Compression
Reduce model size 16GB → 600MB (96% compression)

---

## Cost Breakdown

| Item | Cost | Notes |
|------|------|-------|
| Vast.ai H100 | $3/hour × 3 hours | $9-10 for training |
| HuggingFace | Free | Token and model hosting |
| Dataset | Free | Already created |
| **Total** | **~$10** | One-time training cost |

**Comparison to alternatives**:
- RunPod H100: $3.89/hr × 3 hrs = $11.67
- Lambda H100: $4.50/hr × 3 hrs = $13.50
- Colab Pro+ A100: $50/month (36-48 hours training)

**Vast.ai is cheapest** and 10-15× faster than A100

---

## Key Optimizations (Why H100 is 10-15× Faster)

1. **FastLanguageModel.for_training()** - Enables Flash Attention 2
   - Without: 0.5 it/s @ 35% GPU
   - With: 5-12 it/s @ 100% GPU

2. **Sequence Length 1024** (vs 2048)
   - 4× faster attention computation (O(n²))
   - Less padding waste with packing enabled

3. **Packing Enabled** - Multiple examples per sequence
   - Eliminates 30-40% padding waste
   - Increases effective batch size

4. **10 Data Workers + Prefetch 4**
   - Parallel data loading
   - CPU preprocessing concurrent with GPU training

5. **Batch Size 32 + Gradient Accumulation 2**
   - Large effective batch (64) for stable gradients
   - Optimal for H100 80GB memory

6. **H100 Hardware**
   - 4th-gen tensor cores
   - 3× faster than A100 for attention operations
   - 80GB vs 40GB (larger batches possible)

---

## Next Steps

1. ✅ Complete Phase 1A training (this guide) - **~3 hours, $10**
2. ⏳ Phase 1B: Automated GPT-4 benchmarking - **~1 hour, $5-10**
3. ⏳ Phase 1C (optional): Targeted distillation if score <90%
4. ⏳ Phase 2: Compression (600MB base model) - **~8 hours, $25**
5. ⏳ Phase 3: Domain modifiers (code, reasoning, automation)
6. ⏳ Phase 4: Router system
7. ⏳ Phase 5: Deployment

---

## Support & Documentation

**Notebook**: `notebooks/H100_Training_Clean.ipynb` (16 cells, production-ready)  
**Benchmark**: `scripts/automated_gpt4_benchmark.py` (Phase 1B evaluation)  
**Technical Spec**: `docs/technical_specification.md` (updated Oct 2025)  
**Issues**: https://github.com/dkeviv/Cogumi-LLM/issues

---

**Last Updated**: October 22, 2025  
**Version**: 2.0 (H100 Implementation)  
**Status**: ✅ Tested & Verified on Vast.ai H100
