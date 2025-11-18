# Vast.ai Training Quickstart Guide

**Phase 1 MAML + LoRA Training on H100 with Flash Attention 2**

Last Updated: November 17, 2025 (Hyperparameters corrected for MAML at scale)

---

## Overview

This guide walks you through training the Cogumi-LLM Phase 1 model on Vast.ai using H100 GPUs with Flash Attention 2 (pre-built wheel - no compilation).

**⚠️ CRITICAL UPDATE (November 17, 2025):**
Initial training revealed hyperparameters were incorrectly scaled for 8B models. Learning rates were 40-100× too high, causing loss divergence. Updated to MAML best practices for large language models.

**Expected Metrics (v2 - Corrected):**
- **Training Time:** 3-4 hours (with corrected hyperparameters)
- **Cost:** $8-10 total (H100 at ~$2.20/hour)
- **Dataset:** 53,597 examples (52,916 easy + 681 hard) - cleaned format
- **Output:** ~14GB BF16 model with LoRA adapters
- **Setup Time:** ~5 minutes (pre-built wheels, no compilation)

**Key Changes from v1:**
- Inner LR: 5e-3 → 3e-5 (100× lower, prevents gradient explosion)
- Outer LR: 2e-4 → 5e-6 (40× lower, stable meta-updates)
- LoRA rank: 64 → 16 (4× lower, prevents overfitting on 681 hard examples)
- Tasks/batch: 1 → 4 (4× higher, stable gradient estimates)
- Expected behavior: Smooth loss curve, no divergence after 2500 steps

---

## Step 1: Select GPU Instance on Vast.ai

### Recommended Configuration

**GPU Requirements:**
- **GPU:** H100 SXM (80GB)
- **VRAM:** 80GB minimum (training uses ~40GB)
- **Storage:** 100GB minimum (model + dataset + checkpoints)
- **Network:** Fast download speed for model/data transfer

### Finding the Right Instance

1. Go to [Vast.ai Console](https://vast.ai/console/create/)
2. **Search Filters:**
   ```
   GPU Model: H100 SXM
   GPU RAM: 80 GB
   GPU Count: 1
   Disk Space: 100 GB+
   Reliability Score: 0.95+
   ```

3. **Sort by:**
   - Price (ascending) - Look for ~$2.00-2.50/hour
   - Reliability score (descending)

4. **Verify Compute Capability:**
   - H100 = SM 9.0 (Hopper architecture) ✓ Flash Attention 2 compatible
   - Check instance details show "Compute Capability: 9.0"

### Instance Template Selection

**Recommended: vastai/pytorch (Vast.ai optimized)**
```
Docker Image: vastai/pytorch
```

**Why this image:**
- Pre-installed PyTorch with CUDA
- Optimized for Vast.ai infrastructure
- Faster startup (cached on their servers)
- Tested on their GPU fleet

**Launch Settings:**
- On-Demand instance (not interruptible)
- SSH enabled
- Jupyter (optional, for monitoring)

---

## Step 2: Initial Setup on Vast.ai Instance

### Connect via SSH

```bash
# Get SSH command from Vast.ai console (example):
ssh -p 12345 root@ssh.vast.ai -L 8080:localhost:8080
```

### System Update & Dependencies

```bash
# Update system
apt-get update && apt-get install -y git wget curl vim htop

# Check GPU
nvidia-smi

# Verify H100 and CUDA
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
# Expected: H100, 9.0, 81920 MiB
```

### Python Environment Setup

```bash
# Install Python 3.10+ if needed
apt-get install -y python3.10 python3.10-venv python3-pip

# Verify Python
python3 --version  # Should be 3.10 or higher
```

---

## Step 3: Clone Repository and Setup

### Clone Project

```bash
# Clone repo
cd /workspace  # or your preferred directory
git clone https://github.com/dkeviv/Cogumi-LLM.git
cd Cogumi-LLM

# Verify cleaned data file exists
ls -lh data/phase1/answers/training_data_clean.jsonl
# Should show ~20 MB, 53597 lines
```

### Create Virtual Environment

```bash
# Create venv
python3 -m venv venv

# Activate
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Install Dependencies

Use the automated setup script for complete installation:

```bash
# Run automated setup (includes all dependencies)
bash scripts/gpu_setup.sh

# This will install:
# - PyTorch 2.6.0 + CUDA 12.4
# - Flash Attention 2.7.4 (pre-built wheel - no compilation!)
# - Transformers 4.43.3, PEFT 0.11.1, Accelerate 1.11.0
# - xformers, Unsloth, and all training dependencies
# - All exact pinned versions (no version conflicts)
```

**Installation Time:** ~5 minutes (pre-built wheels, no compilation)

**Verify Flash Attention 2:**
```bash
python3 -c "import flash_attn; print('Flash Attention version:', flash_attn.__version__)"
# Should show: 2.7.4
```

**Verify GPU Setup:**
```bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
# Should show: CUDA available: True, GPU: NVIDIA H100 SXM5 80GB
```

---

## Step 4: Download Base Model

### Llama-3.1-8B-Instruct

```bash
# Option A: Download during training (automatic, recommended)
# The training script will auto-download from HuggingFace

# Option B: Pre-download (optional, saves time during training)
python3 - << 'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "meta-llama/Llama-3.1-8B-Instruct"
print(f"Downloading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
print("Download complete!")
PY
```

**Download Size:** ~16 GB  
**Download Time:** 5-10 minutes (depends on network)

**Note:** You may need HuggingFace token for gated models:
```bash
# Set token (if required)
export HF_TOKEN="your_huggingface_token_here"
# Or use huggingface-cli login
```

---

## Step 5: Pre-Flight Validation (CRITICAL)

### Run Validation Script

```bash
# Activate venv if not already active
source venv/bin/activate

# Run validation
python3 scripts/validate_training_setup.py
```

**Expected Output:**
```
✓ Test 1: Data Loading & Validation
  - Loaded 53597 examples
  - Domain distribution: 8 domains
  - Difficulty: 52916 easy, 681 hard
  
✓ Test 2: Model Loading with Flash Attention
  - GPU: H100 SXM (80GB)
  - Compute Capability: 9.0
  - Flash Attention 2: Available (version 2.7.4)
  - Model loaded: meta-llama/Llama-3.1-8B-Instruct
  
✓ Test 3: Example Formatting
  - Easy example formatted correctly
  - Hard example formatted with natural language CoT
  
✓ Test 4: Forward Pass
  - Forward pass successful
  - Loss computed: X.XXX
  
✓ Test 5: Memory Usage
  - Estimated memory: ~40GB
  - Available VRAM: 80GB
  - Sufficient for training ✓

All validation tests passed!
```

**If validation fails:** Stop and troubleshoot before proceeding to training.

---

## Step 6: Training Execution

### ⚠️ CRITICAL UPDATE (November 17, 2025)

**Hyperparameters have been corrected for MAML at 8B scale.**

Initial training run revealed learning rates were 40-100× too high, causing loss divergence after 2500 steps (12%). Updated to MAML best practices for large language models.

**What Changed:**
- Inner LR: 5e-3 → 3e-5 (100× lower)
- Outer LR: 2e-4 → 5e-6 (40× lower)
- LoRA rank: 64 → 16 (4× lower, prevents overfitting)
- Tasks/batch: 1 → 4 (stable gradients)
- Support/Query: 4 → 6 (better adaptation)

**Use the commands below (v2 corrected hyperparameters):**

---

### Test Mode (Recommended First)

Run on 100 examples first to verify everything works:

```bash
# Test mode: 5 minutes, <$0.50 cost
python3 scripts/phase1_train_maml_lora.py \
  --test_mode \
  --output_dir models/test_phase1_maml_lora
```

**Expected Output:**
```
🚀 Phase 1: MAML + LoRA Training (v2 - Corrected Hyperparameters)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Configuration:
  Model: meta-llama/Llama-3.1-8B-Instruct
  Dataset: data/phase1/answers/training_data_clean.jsonl
  GPU: H100 SXM (80GB)
  Flash Attention: v2.7.4 (pre-built wheel)
  
  Training Examples: 100 (TEST MODE)
  Batch Size: 2
  Gradient Accumulation: 4
  Effective Batch: 8
  
  LoRA Config:
    Rank: 16 (CORRECTED: 4× lower than v1)
    Alpha: 32
    Dropout: 0.05
  
  MAML Config:
    Inner steps: 3
    Tasks per batch: 4
    Support size: 6
    Query size: 6
    
  Learning Rates:
    Outer (meta): 5e-6 (CORRECTED: 40× lower than v1)
    Inner (task): 3e-5 (CORRECTED: 100× lower than v1)
    
Training Progress:
[████████████████████████████████] 100% | 12/12 steps | Loss: 1.234
```

**If test mode succeeds → Proceed to full training**

### Full Training Run (v2 - CORRECTED)

```bash
# Full training: 3-4 hours, ~$8-10 cost
# Use corrected hyperparameters (v2)
python3 scripts/phase1_train_maml_lora.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --data_file data/phase1/answers/training_data_clean.jsonl \
  --output_dir models/phase1_maml_lora_v2 \
  --num_epochs 3 \
  --inner_steps 3 \
  --tasks_per_batch 4 \
  --support_size 6 \
  --query_size 6 \
  --inner_learning_rate 3e-5 \
  --learning_rate 5e-6 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --patience 2 \
  --max_seq_length 2048 \
  2>&1 | tee training_log_v2.txt
```

**Or use the convenience script:**
```bash
./scripts/run_maml_training_v2.sh
```

**Alternative: Force Flash Attention Version**

Flash Attention 2 is automatically detected and used:
```bash
# Standard execution (auto-detects FA2)
python3 scripts/phase1_train_maml_lora.py ...

# Verify FA2 is being used (check training log output)
# Should show: "Flash Attention: v2.7.4 (pre-built wheel)"
```

### Training Parameters Explained

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `--batch_size` | 2 | Per-device batch size (fits in 80GB VRAM) |
| `--gradient_accumulation_steps` | 4 | Effective batch = 2 × 4 = 8 |
| `--learning_rate` | 2e-4 | Outer loop (meta-optimization) |
| `--inner_learning_rate` | 5e-3 | Inner loop (task adaptation) |
| `--num_epochs` | 3 | Full passes over dataset |
| `--lora_rank` | 64 | LoRA low-rank dimension |
| `--lora_alpha` | 128 | LoRA scaling factor (2x rank) |
| `--max_seq_length` | 2048 | Maximum token length (covers 99.99% of data) |

---

## Step 7: Monitoring Training

### Real-Time Monitoring

**Option A: Watch Training Log**
```bash
# In separate terminal/tmux pane
tail -f training_log.txt
```

**Option B: TensorBoard**
```bash
# In separate terminal
tensorboard --logdir models/phase1_maml_lora --port 6006

# Access via SSH tunnel:
# ssh -p <port> root@ssh.vast.ai -L 6006:localhost:6006
# Then open: http://localhost:6006
```

**Option C: GPU Monitoring**
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Or detailed stats
nvtop  # Install: apt-get install nvtop
```

### Key Metrics to Watch

**Training Progress:**
- Loss should decrease steadily (start ~2.5-3.0, end ~0.8-1.2)
- Throughput: ~8-9 samples/second with FA2
- GPU Memory: ~40GB used (should stay under 80GB)
- GPU Utilization: 95-100%

**Red Flags:**
- 🚨 Loss = NaN → Stop training, check learning rate
- 🚨 GPU OOM → Reduce batch_size or max_seq_length
- 🚨 Loss stuck/increasing → Check data quality or learning rate
- 🚨 Very slow progress → Verify Flash Attention 2 is active

---

## Step 8: Checkpoints & Saving

### Automatic Checkpointing

The training script automatically saves checkpoints:

```
models/phase1_maml_lora/
├── checkpoint-1000/       # Every 1000 steps
├── checkpoint-2000/
├── checkpoint-3000/
├── ...
├── final_model/           # Final model (end of training)
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   ├── tokenizer_config.json
│   └── ...
├── training_args.json
├── training_metrics.json
└── tensorboard_logs/
```

### Manual Checkpoint (If Training Interrupted)

Training can be resumed from last checkpoint:
```bash
python3 scripts/phase1_train_maml_lora.py \
  --resume_from_checkpoint models/phase1_maml_lora/checkpoint-XXXX \
  ... (other args)
```

### Download Trained Model

**Option A: During Training (Incremental)**
```bash
# From your local machine
rsync -avz -e "ssh -p <port>" \
  root@ssh.vast.ai:/workspace/Cogumi-LLM/models/phase1_maml_lora/ \
  ./models/phase1_maml_lora/
```

**Option B: After Training (Full Download)**
```bash
# Compress first (on Vast.ai instance)
cd models
tar -czf phase1_maml_lora.tar.gz phase1_maml_lora/

# Download (from local machine)
scp -P <port> root@ssh.vast.ai:/workspace/Cogumi-LLM/models/phase1_maml_lora.tar.gz .
```

**Model Size:** ~14GB compressed, ~16GB uncompressed

---

## Step 9: Post-Training Validation

### Quick Inference Test

```bash
python3 - << 'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(
    base_model,
    "models/phase1_maml_lora/final_model"
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Test prompt
prompt = "What is the derivative of x^2?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Prompt:", prompt)
print("Response:", response)
PY
```

### Expected Output
```
Prompt: What is the derivative of x^2?
Response: The derivative of x^2 is 2x.
```

---

## Step 10: Cost Optimization Tips

### Minimize Costs

**Before Training:**
1. ✅ Run validation script (~2 minutes, ~$0.10)
2. ✅ Run test mode first (~5 minutes, ~$0.50)
3. ✅ Pre-download model during cheap instance time

**During Training:**
1. ✅ Use on-demand (not interruptible) instances for reliability
2. ✅ Monitor first 30 minutes closely for issues
3. ✅ Use tmux/screen to prevent SSH disconnection loss
4. ✅ Set up auto-checkpoint every 1000 steps

**After Training:**
1. ✅ Download model immediately
2. ✅ Destroy instance promptly (don't leave idle)
3. ✅ Verify files before destroying

### Cost Breakdown (H100 @ $2.20/hour)

| Stage | Time | Cost |
|-------|------|------|
| Setup & Validation | 10 min | $0.37 |
| Test Mode | 5 min | $0.18 |
| Full Training (FA2) | 6-7 hours | $13.20-15.40 |
| Post-Validation | 10 min | $0.37 |
| **Total** | **~7 hours** | **~$13-15** |

---

## Troubleshooting

### Issue: Flash Attention 2 Installation

**Symptoms:**
```
Could not find flash_attn module
ImportError: No module named 'flash_attn'
```

**Solutions:**
1. Verify installation:
   ```bash
   pip show flash-attn
   # Should show: Version: 2.7.4
   ```

2. If not installed, run gpu_setup.sh again:
   ```bash
   bash scripts/gpu_setup.sh
   ```

3. Manual installation (if needed):
   ```bash
   pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.7.4+cu124torch2.6-cp310-cp310-linux_x86_64.whl
   ```

4. Verify GPU compatibility:
   ```bash
   nvidia-smi --query-gpu=compute_cap --format=csv
   # Should be 9.0 for H100
   ```

### Issue: Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size:
   ```bash
   --batch_size 1 --gradient_accumulation_steps 8
   ```

2. Reduce sequence length:
   ```bash
   --max_seq_length 1024
   ```

3. Enable gradient checkpointing (add to training script args if available)

4. Verify VRAM:
   ```bash
   nvidia-smi
   # Should have 80GB total
   ```

### Issue: Training Loss = NaN

**Symptoms:**
```
Loss: nan
```

**Solutions:**
1. Lower learning rate:
   ```bash
   --learning_rate 1e-4 --inner_learning_rate 2.5e-3
   ```

2. Check data quality:
   ```bash
   python3 scripts/validate_training_setup.py
   ```

3. Enable mixed precision safeguards (script has BF16 by default)

### Issue: Slow Training Speed

**Symptoms:**
- <5 samples/second
- Training estimate >10 hours

**Solutions:**
1. Verify Flash Attention 2 is active:
   - Check training log for "Flash Attention: v2.7.4" message
   
2. Check GPU utilization:
   ```bash
   nvidia-smi
   # Should show 95-100% GPU usage
   ```

3. Verify no CPU bottleneck:
   ```bash
   htop
   # Should not show 100% CPU
   ```

4. Check I/O bottleneck:
   ```bash
   iostat -x 1
   # Should not show 100% disk usage
   ```

### Issue: SSH Connection Lost

**Prevention:**
```bash
# Use tmux or screen
tmux new -s training
# Or
screen -S training

# Run training inside tmux/screen
python3 scripts/phase1_train_maml_lora.py ...

# Detach: Ctrl+B, D (tmux) or Ctrl+A, D (screen)
# Reattach later: tmux attach -t training
```

### Issue: HuggingFace Model Download Fails

**Symptoms:**
```
HTTPError: 401 Unauthorized
```

**Solutions:**
1. Login to HuggingFace:
   ```bash
   pip install huggingface-hub
   huggingface-cli login
   # Enter your token
   ```

2. Or set token:
   ```bash
   export HF_TOKEN="your_token_here"
   ```

---

## Quick Reference Commands

### Essential Commands

```bash
# Activate environment
source venv/bin/activate

# Check GPU
nvidia-smi

# Validate setup
python3 scripts/validate_training_setup.py

# Test mode training
python3 scripts/phase1_train_maml_lora.py --test_mode

# Full training
python3 scripts/phase1_train_maml_lora.py

# Monitor GPU
watch -n 1 nvidia-smi

# Monitor training log
tail -f training_log.txt

# Download model
scp -P <port> root@ssh.vast.ai:/workspace/Cogumi-LLM/models/phase1_maml_lora.tar.gz .
```

---

## Expected Training Output

### Successful Training Log (Abbreviated)

```
```
🚀 Phase 1: MAML + LoRA Training
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GPU Detection:
  ✓ Device: H100 SXM (80GB VRAM)
  ✓ Compute Capability: 9.0 (Hopper)
  ✓ Flash Attention 2: Enabled (v2.7.4)

Dataset Validation:
  ✓ Loaded 53,597 examples (cleaned format)
  ✓ Easy: 52,916 (98.73%)
  ✓ Hard: 681 (1.27%)
```
  ✓ Domains: 8 balanced
  ✓ No missing fields
  ✓ No duplicates

Model Configuration:
  Model: meta-llama/Llama-3.1-8B-Instruct
  Parameters: 8.3B
  Trainable (LoRA): 134M (1.6%)
  Precision: BF16
  
Training Progress:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Epoch 1/3
Step    500/6,000 | Loss: 1.234 | LR: 2e-4 | 9.2 samples/s | ETA: 4h 23m
Step  1,000/6,000 | Loss: 1.098 | LR: 2e-4 | 9.4 samples/s | ETA: 4h 15m
Step  1,500/6,000 | Loss: 0.987 | LR: 2e-4 | 9.3 samples/s | ETA: 4h 18m
Step  2,000/6,000 | Loss: 0.923 | LR: 1.9e-4 | 9.5 samples/s | ETA: 4h 10m
...

Epoch 2/3
Step  2,500/6,000 | Loss: 0.876 | LR: 1.8e-4 | 9.4 samples/s | ETA: 3h 42m
...

Epoch 3/3
Step  5,500/6,000 | Loss: 0.812 | LR: 5e-5 | 9.6 samples/s | ETA: 0h 52m
Step  6,000/6,000 | Loss: 0.798 | LR: 1e-5 | 9.5 samples/s | ETA: 0h 00m

Training Complete! 🎉
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Time: 5h 32m
Final Loss: 0.798
Avg Throughput: 9.4 samples/s
Model Saved: models/phase1_maml_lora/final_model
```

---

## Next Steps After Training

1. **Download Model** (see Step 8)
2. **Run Benchmarks** (Phase 1B evaluation)
   ```bash
   python3 scripts/run_phase1b_benchmark.sh
   ```
3. **Proceed to Phase 2** (Speed Infrastructure)
   - Draft model training (500M params)
   - Speculative decoding
   - MoD router
   - KV cache quantization

---

## Support & Resources

**Project Documentation:**
- `.github/Revised_complete_pipeline.md` - Full pipeline architecture
- `docs/IMPLEMENTATION_CHECKLIST.md` - Task tracking
- `docs/technical_specification.md` - Technical details
- `docs/H100_UNSLOTH_MIGRATION.md` - H100 optimization notes

**Vast.ai Resources:**
- [Vast.ai Documentation](https://vast.ai/docs/)
- [Vast.ai Discord](https://discord.gg/vast)

**Model Resources:**
- [Llama 3.1 Model Card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [Flash Attention Paper](https://arxiv.org/abs/2307.08691)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

---

## Checklist for Training Run

Before starting training, verify:

- [ ] H100 GPU instance reserved ($2.00-2.50/hour)
- [ ] SSH connection working
- [ ] CUDA 12.0+ and nvidia-smi working
- [ ] Repository cloned and data file exists (21.5 MB)
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (flash-attn >= 2.6.0)
- [ ] Base model downloaded or accessible
- [ ] Validation script passes all tests
- [ ] Test mode training successful
- [ ] tmux/screen session created (prevents SSH loss)
- [ ] Sufficient disk space (100GB+)
- [ ] TensorBoard/monitoring set up (optional)

Ready to train? Run:
```bash
python3 scripts/phase1_train_maml_lora.py
```

---

**Estimated Total Cost: $13-15 for full Phase 1 training on H100 with Flash Attention 3**

**Good luck! 🚀**
