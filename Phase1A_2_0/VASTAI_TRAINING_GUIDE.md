# Phase 1A 2.0 - Vast.ai H100 Training Guide

## 🎯 Complete Step-by-Step Guide for Vast.ai H100 Training

**Goal**: Train Phase 1A 2.0 in 8-12 hours for $20-30 using optimal configuration

**Key Optimizations**:
- ✅ Use local NVMe storage (not persistent) for 20-30% I/O speedup
- ✅ Pre-compiled wheels (5-10 min install vs 30-40 min)
- ✅ Optimized training config (4 workers, batch 4, grad_accum 2)
- ✅ Pre-tokenization (optional 5-10% speedup)

---

## 📋 Prerequisites

### Before Starting
1. **Vast.ai account** with sufficient balance ($25-30 recommended)
2. **GitHub repository** with Phase1A_2_0 and Phase0 folders pushed
   - Repository: `https://github.com/dkeviv/Cogumi-LLM.git`
   - Contains: Phase0 (dataset), Phase1A_2_0 (training scripts), configs, docs
3. **Git access** on Vast.ai instance (usually pre-installed)

### Required Instance Specs
- **GPU**: H100 80GB (single GPU)
- **RAM**: 64GB minimum (80GB+ recommended)
- **Storage**: 150GB+ NVMe/SSD (NOT persistent storage)
- **CUDA**: 12.1 or 12.4
- **OS**: Ubuntu 22.04 with PyTorch template
- **Expected Cost**: $2.00-2.50/hr × 10-12 hours = $20-30

---

## 🚀 Step-by-Step Instructions

### Phase 1: Instance Setup (10 minutes)

#### Step 1.1: Select Instance on Vast.ai
```
Filter Settings:
- GPU: H100 80GB (single GPU, NOT multi-GPU)
- CUDA: 12.1+
- RAM: 64GB+
- Disk Space: 150GB+ NVMe/SSD
- Sort by: $/hr (lowest first)
```

**Important**: 
- ✅ Select instance with **NVMe or SSD** (fast local storage)
- ❌ Avoid "Network Storage" or "Persistent Storage" (slow I/O)

#### Step 1.2: Launch Instance
```
Image: pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
OR: nvidia/cuda:12.1.0-devel-ubuntu22.04
```

**SSH into Instance**:
```bash
ssh -p <PORT> root@<HOST_IP> -L 8080:localhost:8080
```

#### Step 1.3: Verify Hardware
```bash
# Check GPU
nvidia-smi
# Expected: H100 80GB, CUDA 12.1+

# Check local storage (should be NVMe/SSD)
df -h /workspace
# Expected: 150GB+ fast local storage

# Check CUDA version
nvcc --version
# Expected: CUDA 12.1 or 12.4
```

---

### Phase 2: Repository Setup (5 minutes)

#### Step 2.1: Navigate to Workspace
```bash
cd /workspace
```

**Important**: `/workspace` is typically the fast local NVMe storage on Vast.ai

#### Step 2.2: Clone Repository with Git
```bash
# Clone the Cogumi-LLM repository
git clone https://github.com/dkeviv/Cogumi-LLM.git
cd Cogumi-LLM

# Verify repository cloned successfully
ls -la
# Expected: Phase0/, Phase1A_2_0/, data/, docs/, scripts/, etc.
```

#### Step 2.3: Verify Phase1A_2_0 Structure
```bash
# Check Phase1A_2_0 folder structure
ls -la Phase1A_2_0/
# Expected output:
# scripts/           - Training scripts
# data/              - Dataset folder
# docs/              - Documentation
# README.md          - Phase overview
# PROJECT_STATUS.md  - Status summary
# VASTAI_TRAINING_GUIDE.md - This guide

# Verify all required files present
ls -la Phase1A_2_0/scripts/
# Expected: train_phase1a_optimized_h100.py, requirements-stable-precompiled.txt, etc.
```

#### Step 2.4: Download Dataset (REQUIRED)
```bash
# Navigate to data folder
cd Phase1A_2_0/data

# Download dataset from GitHub releases
wget https://github.com/dkeviv/Cogumi-LLM/releases/download/v1.0-dataset/public_500k_filtered.jsonl

# If wget not available, use curl:
# curl -L -o public_500k_filtered.jsonl https://github.com/dkeviv/Cogumi-LLM/releases/download/v1.0-dataset/public_500k_filtered.jsonl

# Verify download
ls -lh public_500k_filtered.jsonl
# Expected: ~870MB (600K examples)

# Verify line count
wc -l public_500k_filtered.jsonl
# Expected: 600000 lines

# Verify dataset format (optional)
head -1 public_500k_filtered.jsonl | python -m json.tool
# Expected: Valid JSON with "instruction" and "response" fields
```

**Important**: The dataset is NOT included in the git repository due to size limits. You must download it separately.

---

### Phase 3: Environment Setup (5-10 minutes)

#### Step 3.1: Create Virtual Environment
```bash
cd /workspace/Cogumi-LLM/Phase1A_2_0/scripts

# Create venv
python3.10 -m venv venv_phase1a_2_0

# Activate venv
source venv_phase1a_2_0/bin/activate

# Verify Python version
python --version
# Expected: Python 3.10.x
```

#### Step 3.2: Upgrade pip
```bash
pip install --upgrade pip setuptools wheel
```

#### Step 3.3: Install Dependencies in Stages (5-10 minutes)

**CRITICAL**: Install in 4 stages to avoid Flash Attention build errors

**Stage 1: Install PyTorch first (required by Flash Attention)**
```bash
# Install PyTorch 2.3.1 with CUDA 12.1 support
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch installed and importable
python -c "import torch; print(f'✅ PyTorch {torch.__version__} installed')"
# Expected: ✅ PyTorch 2.3.1+cu121 installed
```

**Stage 2: Install psutil (required by Flash Attention's setup.py)**
```bash
# Flash Attention's setup.py imports psutil during build
pip install psutil==5.9.8

# Verify psutil installed
python -c "import psutil; print('✅ psutil installed')"
# Expected: ✅ psutil installed
```

**Stage 3: Install Flash Attention with --no-build-isolation**
```bash
# CRITICAL: Use --no-build-isolation flag
# This allows Flash Attention's setup.py to import torch and psutil
pip install flash-attn==2.5.8 --no-build-isolation \
    --extra-index-url https://flashattn.github.io/whl/cu121/torch2.3/

# Verify Flash Attention installed
python -c "import flash_attn; print('✅ Flash Attention installed')"
# Expected: ✅ Flash Attention installed
```

**Stage 4: Install everything else**
```bash
# Now install all other dependencies
pip install -r requirements-stable-precompiled.txt

# This will install:
# - Transformers 4.43.3 (10 sec)
# - xformers 0.0.26.post1 (required by Unsloth) (20 sec)
# - Unsloth July-2024 (30 sec)
# - NumPy 1.26.4 (5 sec)
# - All other dependencies (2-3 min)

# Expected total time: 5-10 minutes for all 4 stages
```

**Why 4 stages?**
1. **Flash Attention IMPORTS torch AND psutil** during setup.py execution (not just checks if installed)
2. **pip's build isolation** creates a clean environment that doesn't have access to installed packages
3. **--no-build-isolation** flag disables this isolation, allowing Flash Attention to import dependencies
4. **This flag cannot be specified in requirements.txt**, must be done manually

**Common Installation Issues & Solutions:**

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: No module named 'torch'` | Flash Attention can't find torch in build isolation | Use `--no-build-isolation` flag (Stage 3) |
| `ModuleNotFoundError: No module named 'psutil'` | Flash Attention needs psutil during build | Install psutil first (Stage 2) |
| `ModuleNotFoundError: No module named 'xformers'` | Unsloth requires xformers for speedup | Already fixed in requirements (includes xformers 0.0.26.post1) |
| `ResolutionImpossible: huggingface-hub==0.23.0` | transformers 4.43.3 requires >=0.23.2 | Already fixed in requirements (uses 0.23.4) |
| `git checkout -q 2024.7 did not match` | Unsloth uses month-name tags | Already fixed in requirements (uses July-2024) |

**If installation fails**: Pull latest from git (`git pull origin main`) - these issues have been fixed.

#### Step 3.4: Verify Installation
```bash
python verify_h100_environment.py

# Expected output:
# ================================================================================
# ENVIRONMENT VERIFICATION COMPLETE
# ================================================================================
# 
# Configuration Score: 95/100 (EXCELLENT)
# 
# ✅ All critical components installed
# ✅ H100 GPU detected
# ✅ Flash Attention available
# ✅ Ready for training
```

**If score < 90%**: Check output for specific issues and fix before proceeding

---

### Phase 4: Data Optimization (15-20 minutes)

#### Step 4.1: Copy Dataset to Local NVMe (CRITICAL for 20-30% speedup)
```bash
# Copy to /tmp (fast local storage, NOT network storage)
cp ../data/public_500k_filtered.jsonl /tmp/dataset.jsonl

# Verify copy
ls -lh /tmp/dataset.jsonl
# Expected: ~870MB
```

**Why this matters**: 
- ✅ `/tmp` is on local NVMe (fast I/O)
- ❌ Network/persistent storage is slow (bottleneck)
- 📊 Result: 20-30% I/O speedup

#### Step 4.2: Optional - Pre-tokenize Dataset (10-15 minutes, 5-10% speedup)
```bash
# Pre-tokenize to local storage
python pretokenize_dataset.py \
    --input /tmp/dataset.jsonl \
    --output /tmp/tokenized_dataset \
    --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Expected: 10-15 minutes, ~1.2GB output

# Verify tokenized dataset
ls -lh /tmp/tokenized_dataset/
# Expected: dataset_info.json, state.json, data-00000-of-00001.arrow
```

**Decision**: 
- ✅ **Recommended**: Pre-tokenize for 5-10% speedup
- ⚠️ **Skip if time-constrained**: Training will tokenize on-the-fly

---

### Phase 5: Training (8-12 hours)

#### Step 5.1: Start Training

**Option A: Without Pre-tokenization**
```bash
# Start training
python train_phase1a_optimized_h100.py \
    --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --dataset_path "/tmp/dataset.jsonl" \
    --output_dir "/workspace/Cogumi-LLM/Phase1A_2_0/data/checkpoints" \
    --logging_dir "/workspace/Cogumi-LLM/Phase1A_2_0/data/logs" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --dataloader_num_workers 4 \
    --save_steps 2000 \
    --logging_steps 10 \
    --save_total_limit 5 \
    --torch_compile

# Expected: 8-12 hours
# Cost: $20-30 @ $2.50/hr
```

**Option B: With Pre-tokenization (5-10% faster)**
```bash
# Start training with pre-tokenized data
python train_phase1a_optimized_h100.py \
    --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --use_pretokenized \
    --dataset_path "/tmp/tokenized_dataset" \
    --output_dir "/workspace/Cogumi-LLM/Phase1A_2_0/data/checkpoints" \
    --logging_dir "/workspace/Cogumi-LLM/Phase1A_2_0/data/logs" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --dataloader_num_workers 4 \
    --save_steps 2000 \
    --logging_steps 10 \
    --save_total_limit 5 \
    --torch_compile

# Expected: 7.5-11 hours (5-10% faster)
# Cost: $19-28 @ $2.50/hr
```

#### Step 5.2: Monitor Training

**Terminal 1 (Training Output)**:
```bash
# Training will show progress:
# Step 100/28000 | Loss: 2.456 | LR: 2.5e-5 | Time: 1.2s/step
# Step 200/28000 | Loss: 2.123 | LR: 2.5e-5 | Time: 1.2s/step
```

**Terminal 2 (GPU Monitor)** - Open new SSH session:
```bash
watch -n 1 nvidia-smi

# Expected:
# GPU Util: 95-100%
# Memory: 70-75GB / 80GB
# Power: 600-700W
```

**Terminal 3 (Training Logs)** - Optional:
```bash
tail -f /workspace/Cogumi-LLM/Phase1A_2_0/data/logs/training.log
```

#### Step 5.3: Checkpoints
Training will save checkpoints every 2000 steps:
```
Phase1A_2_0/data/checkpoints/
├── checkpoint-2000/    # After 2,000 steps (~1 hour)
├── checkpoint-4000/    # After 4,000 steps (~2 hours)
├── checkpoint-6000/    # After 6,000 steps (~3 hours)
...
└── checkpoint-28000/   # Final checkpoint (~10 hours)
```

---

### Phase 6: Post-Training - Merge & Save (30 minutes)

#### Step 6.1: Merge Adapter to Base Model
```bash
# After training completes, merge adapter
python merge_adapter_fullprecision.py \
    --base_model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --adapter_path "/workspace/Cogumi-LLM/Phase1A_2_0/data/checkpoints/checkpoint-28000" \
    --output_path "/workspace/Cogumi-LLM/Phase1A_2_0/data/merged/merged_phase1a_v2.0" \
    --device "cuda"

# Expected: 10-15 minutes
# Output: ~10GB merged model
```

#### Step 6.2: Validate Merged Model
```bash
# Quick validation
python ../../scripts/validate_merged_model.py \
    --model_path "/workspace/Cogumi-LLM/Phase1A_2_0/data/merged/merged_phase1a_v2.0" \
    --num_samples 100

# Expected results:
# MATH: 6-8% wins, 68-72% ties
# CODE: 46-50% wins, 18-22% ties
# Overall: 89-91% GPT-4 equivalent
```

#### Step 6.3: Download Results to Local
```bash
# On your local machine, run:
# Download merged model
scp -P <PORT> -r root@<HOST_IP>:/workspace/Cogumi-LLM/Phase1A_2_0/data/merged ./

# Download training logs
scp -P <PORT> -r root@<HOST_IP>:/workspace/Cogumi-LLM/Phase1A_2_0/data/logs ./

# Download best checkpoint (optional)
scp -P <PORT> -r root@<HOST_IP>:/workspace/Cogumi-LLM/Phase1A_2_0/data/checkpoints/checkpoint-28000 ./
```

---

## 📊 Expected Timeline & Costs

| Phase | Task | Time | Cost @ $2.50/hr |
|-------|------|------|-----------------|
| 1 | Instance setup | 10 min | ~$0.40 |
| 2 | Repository setup | 5 min | ~$0.20 |
| 3 | Environment install | 10 min | ~$0.40 |
| 4 | Data optimization | 20 min | ~$0.85 |
| 5 | **Training** | **8-12 hours** | **$20-30** |
| 6 | Merge & download | 30 min | ~$1.25 |
| **Total** | **End-to-end** | **9-13 hours** | **$22-33** |

---

## ✅ Success Criteria

### During Training
- [ ] GPU utilization: 95-100%
- [ ] Training loss: Decreasing steadily
- [ ] Time per step: ~1.2 seconds
- [ ] Gradient norms: Stable (not exploding)
- [ ] No OOM errors
- [ ] Checkpoints saving every 2000 steps

### After Training
- [ ] Training completed all 3 epochs (~28,000 steps)
- [ ] Final loss: <0.5
- [ ] Merged model size: ~10GB
- [ ] MATH validation: 70% ties or better
- [ ] CODE validation: 48% wins or better
- [ ] No merge corruption (coherent text generation)

---

## 🔧 Troubleshooting

### Issue 1: OOM Error
**Symptoms**: `CUDA out of memory` error

**Solution**:
```bash
# Reduce batch size
python train_phase1a_optimized_h100.py \
    --per_device_train_batch_size 2 \  # Changed from 4
    --gradient_accumulation_steps 4 \   # Changed from 2 (keep same effective batch)
    ...
```

### Issue 2: Slow I/O
**Symptoms**: Low GPU utilization (<80%), slow steps

**Solution**:
```bash
# Verify dataset is on local NVMe, not network storage
ls -lh /tmp/dataset.jsonl

# If not, copy again
cp /workspace/Cogumi-LLM/Phase1A_2_0/data/public_500k_filtered.jsonl /tmp/dataset.jsonl
```

### Issue 3: Flash Attention Import Error
**Symptoms**: `ModuleNotFoundError: No module named 'flash_attn'`

**Solution**:
```bash
# Reinstall Flash Attention with pre-compiled wheel
pip install flash-attn==2.5.8 --no-build-isolation \
    --extra-index-url https://flashattn.github.io/whl/cu121/torch2.3/
```

### Issue 4: Training Too Slow
**Symptoms**: >1.5 seconds per step

**Check**:
```bash
# 1. Verify GPU utilization
nvidia-smi
# Should be 95-100%

# 2. Verify dataset location
ls -lh /tmp/dataset.jsonl
# Should be on fast local storage

# 3. Check num_workers
# Should be 4 (optimal for H100)

# 4. Verify torch.compile is enabled
# Check training script output for "Compiling model..."
```

---

## 📝 Complete Training Command Template

```bash
#!/bin/bash
# Complete training script for copy-paste

# Navigate to scripts folder
cd /workspace/Cogumi-LLM/Phase1A_2_0/scripts

# Activate environment
source venv_phase1a_2_0/bin/activate

# Copy dataset to fast local storage (CRITICAL)
cp ../data/public_500k_filtered.jsonl /tmp/dataset.jsonl

# Optional: Pre-tokenize (5-10% speedup)
# python pretokenize_dataset.py \
#     --input /tmp/dataset.jsonl \
#     --output /tmp/tokenized_dataset \
#     --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Start training
python train_phase1a_optimized_h100.py \
    --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --dataset_path "/tmp/dataset.jsonl" \
    --output_dir "/workspace/Cogumi-LLM/Phase1A_2_0/data/checkpoints" \
    --logging_dir "/workspace/Cogumi-LLM/Phase1A_2_0/data/logs" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --dataloader_num_workers 4 \
    --save_steps 2000 \
    --logging_steps 10 \
    --save_total_limit 5 \
    --learning_rate 2e-4 \
    --warmup_steps 100 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --torch_compile \
    --bf16 \
    --gradient_checkpointing

echo "Training complete!"
echo "Next: Merge adapter and validate results"
```

---

## 🎯 Quick Reference Card

**Setup**:
```bash
cd /workspace/Cogumi-LLM/Phase1A_2_0/scripts
python3.10 -m venv venv_phase1a_2_0
source venv_phase1a_2_0/bin/activate
pip install -r requirements-stable-precompiled.txt
python verify_h100_environment.py
```

**Data Prep**:
```bash
cp ../data/public_500k_filtered.jsonl /tmp/dataset.jsonl
```

**Train**:
```bash
python train_phase1a_optimized_h100.py \
    --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --dataset_path "/tmp/dataset.jsonl" \
    --output_dir "/workspace/Cogumi-LLM/Phase1A_2_0/data/checkpoints" \
    --logging_dir "/workspace/Cogumi-LLM/Phase1A_2_0/data/logs"
```

**Merge**:
```bash
python merge_adapter_fullprecision.py \
    --base_model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --adapter_path "/workspace/Cogumi-LLM/Phase1A_2_0/data/checkpoints/checkpoint-28000" \
    --output_path "/workspace/Cogumi-LLM/Phase1A_2_0/data/merged/merged_phase1a_v2.0"
```

---

**Status**: ✅ Ready for Vast.ai H100 Training - Expected: 8-12 hours, $20-30 cost
