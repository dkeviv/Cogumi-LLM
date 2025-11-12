# Phase 1C Training - Vast.ai Quickstart Guide

**Goal:** Train Phase 1C model with 10K CoT examples to achieve 90-95% pass rate  
**Current Baseline:** 78.62% pass rate  
**Expected Improvement:** +12-16 points  
**Duration:** 5-7 hours on H100 80GB  
**Cost:** ~$15-20  

---

## 📋 Prerequisites

### Required Files (Prepare Locally)
1. **Training Dataset:** `phase1c_10k_with_cot_deduped.jsonl` (9,488 examples, ~18MB) ⭐ **RECOMMENDED**
   - Deduplicated version with zero duplicates
   - 100% training-ready, 5-7% faster training
   - Original: `phase1c_10k_with_cot.jsonl` (9,998 examples) still available but not recommended
2. **Base Model:** Phase 1C current checkpoint (from previous training)
3. **Training Script:** `train_phase1c_cot.py` (provided below)

### Required Credentials
- **HuggingFace Token:** For model upload (read + write access)
- **Vast.ai Account:** With sufficient credits (~$20)

---

## 🚀 Step-by-Step Setup

### Step 1: Rent Vast.ai Instance

**Recommended GPU:** H100 80GB (fastest, most efficient)  
**Alternative:** A100 80GB (slightly longer training time)

#### Search Filters:
```
GPU: H100 80GB (or A100 80GB)
VRAM: 80GB minimum
Disk: 50GB minimum
DLPerf: >30
Reliability: >99%
On-demand: Yes (for flexibility)
```

#### Template Selection:
```
Image: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
OR
Image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel
```

#### Pricing Guide:
- H100 80GB: $1.50-2.50/hour → 5-7 hours = $7.50-17.50
- A100 80GB: $1.20-2.00/hour → 6-8 hours = $7.20-16.00

---

### Step 2: Connect to Instance

Once instance is running:

```bash
# SSH connection (from Vast.ai dashboard)
ssh -p <PORT> root@<IP_ADDRESS> -L 8080:localhost:8080

# Or use Vast.ai's web-based SSH terminal
```

---

### Step 3: Setup Environment

```bash
# Navigate to workspace
cd /workspace

# Create directory structure
mkdir -p data models scripts logs

# Install dependencies (copy-paste this entire block)
pip uninstall -y torch torchvision torchaudio xformers transformers \
    tokenizers bitsandbytes peft accelerate trl datasets unsloth \
    unsloth-zoo huggingface-hub tensorboard wandb

pip install psutil==7.1.2 packaging rich tensorboard

pip install --force-reinstall torch==2.6.0+cu124 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.7.4+cu124torch2.6-cp310-cp310-linux_x86_64.whl

pip install bitsandbytes==0.48.1
pip install transformers==4.43.3 tokenizers==0.19.1
pip install peft==0.11.1 accelerate==1.11.0 trl==0.9.6 datasets==4.3.0
pip install xformers==0.0.28.post2 --no-deps
pip install "unsloth[cu124-torch260] @ git+https://github.com/unslothai/unsloth.git@July-2024" --no-deps
pip install tqdm jsonlines ninja huggingface-hub safetensors

# Verify installation
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python3 -c "import flash_attn; print('FlashAttention: OK')"
python3 -c "import unsloth; print('Unsloth: OK')"
```

**Expected Output:**
```
PyTorch: 2.6.0+cu124
CUDA: True
FlashAttention: OK
Unsloth: OK
```

---

### Step 4: Upload Data & Model

#### Option A: Upload from Local Machine (Recommended)

**From your local machine terminal:**

```bash
# Navigate to your project directory
cd /Users/vivekdurairaj/Projects/Cogumi-LLM

# Upload training dataset (deduplicated version - RECOMMENDED)
scp -P <PORT> Phase1C_Targeted_Distillation/data/phase1c_10k_with_cot_deduped.jsonl \
    root@<IP_ADDRESS>:/workspace/data/

# Upload training script
scp -P <PORT> Phase1C_Targeted_Distillation/scripts/train_phase1c_cot.py \
    root@<IP_ADDRESS>:/workspace/scripts/

# Upload validation script (recommended - catches data issues early)
scp -P <PORT> Phase1C_Targeted_Distillation/scripts/validate_training_data.py \
    root@<IP_ADDRESS>:/workspace/scripts/

# Upload base model (if you have local checkpoint)
# If model is on HuggingFace, skip this and use HF hub in script
scp -P <PORT> -r models/phase1c_base/* \
    root@<IP_ADDRESS>:/workspace/models/phase1c_base/
```

#### Option B: Download from HuggingFace/URLs

**On Vast.ai instance:**

```bash
cd /workspace/data

# Download training dataset (if uploaded to cloud/HF)
# Use deduplicated version for better training quality
wget https://your-url/phase1c_10k_with_cot_deduped.jsonl

# OR use HuggingFace CLI
huggingface-cli download <your-org>/phase1c-data --repo-type dataset --local-dir .

# Download base model (if on HuggingFace)
cd /workspace/models
huggingface-cli download <your-org>/phase1c-base --local-dir phase1c_base
```

#### Option C: Use HuggingFace Hub Directly (Simplest)

If your model is on HuggingFace, the training script will download automatically.

---

### Step 5: Set HuggingFace Token

```bash
# Login to HuggingFace (for model upload)
huggingface-cli login

# Or set as environment variable
export HUGGINGFACE_TOKEN="hf_YOUR_TOKEN_HERE"
```

---

### Step 6: Verify Data

```bash
cd /workspace

### Step 6: Validate Training Data (Recommended)

**Run comprehensive validation before training to catch issues early:**

```bash
cd /workspace

# Run comprehensive validation (on deduplicated data)
python3 scripts/validate_training_data.py \
    --data_path /workspace/data/phase1c_10k_with_cot_deduped.jsonl
```

**What This Checks:**
- ✅ All required fields present (`instruction`, `cot_response`, `generation_success`)
- ✅ No empty/null values in critical fields
- ✅ Instruction lengths appropriate (10-5000 chars)
- ✅ CoT structure complete (`<thinking>`, `DRAFT`, `CRITIQUE`, `REVISED`)
- ✅ EOS tokens present (`<|end_of_text|>`)
- ✅ Categories valid (code, math, reasoning, creative)
- ✅ No duplicates (deduplicated data has 0 duplicates!)
- ✅ Training-readiness assessment

**Expected Output:**
```
🔍 COMPREHENSIVE DATA VALIDATION
================================

✅ Loaded 9,488 examples (deduplicated)
✅ All examples have required fields
✅ No empty/null values
✅ All instructions have appropriate length
✅ 100% EOS tokens present
✅ 99.3%+ complete CoT structure
✅ All categories valid
✅ ZERO duplicates (100% unique!)

🎯 FINAL VALIDATION VERDICT
✅ VALIDATION PASSED - NO ISSUES FOUND!
   • Training-ready: 9,488 examples (100%)
   • Zero failed generations
   • Zero duplicates
🎉 PERFECT - READY FOR TRAINING!
```

**If Validation Fails:** Fix issues before training to avoid wasting GPU hours.

---

### Step 7: Start Training

```bash
cd /workspace

# Run training (with deduplicated data - RECOMMENDED)
python3 scripts/train_phase1c_cot.py \
    --data_path /workspace/data/phase1c_10k_with_cot_deduped.jsonl \
    --model_path /workspace/models/phase1c_base \
    --output_dir /workspace/models/phase1c_cot_trained \
    --learning_rate 3e-6 \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --max_seq_length 2048 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --save_steps 100 \
    --logging_steps 10 \
    --warmup_steps 50 \
    --use_flash_attention \
    --hf_token $HUGGINGFACE_TOKEN

# Monitor training in real-time
tail -f /workspace/logs/training.log
```

#### Alternative: Use tmux for Background Training

```bash
# Start tmux session
tmux new -s training

# Run training (inside tmux)
python3 scripts/train_phase1c_cot.py [... parameters ...]

# Detach: Press Ctrl+B, then D

# Reattach later
tmux attach -t training

# Check if still running
tmux ls
```

---

### Step 8: Monitor Training

Training will display:

```
🚀 Phase 1C - CoT Training
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Configuration:
   • Base model: /workspace/models/phase1c_base
   • Training examples: 9,998
   • Learning rate: 3e-6
   • Batch size: 4 (effective: 32 with grad accum)
   • Max sequence length: 2048
   • LoRA rank: 64
   • Training epochs: 3
   • Total steps: ~937

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Training Progress:
  Epoch 1/3  ━━━━━━━━━━━━━━━━━━━━━━━  45%  Step 140/312
  Loss: 0.523  LR: 2.8e-6  Time: 2.3s/step

🔍 Key Metrics:
   • Average loss: 0.545
   • Best loss: 0.489
   • Tokens/sec: 2,340
   • ETA: 3.2 hours
```

---

### Step 9: Training Checkpoints

Training automatically saves checkpoints:

```
/workspace/models/phase1c_cot_trained/
├── checkpoint-100/
├── checkpoint-200/
├── checkpoint-300/
├── ...
└── final/
    ├── adapter_model.bin
    ├── adapter_config.json
    ├── tokenizer/
    └── training_args.json
```

**Best checkpoint selected based on lowest validation loss.**

---

### Step 10: Merge & Upload Model

After training completes:

```bash
cd /workspace

# Merge LoRA adapter with base model
python3 << 'EOF'
from unsloth import FastLanguageModel
import torch

print("🔄 Merging LoRA adapter with base model...")

# Load base + adapter
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/workspace/models/phase1c_cot_trained/final",
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True,
)

# Merge and save
print("💾 Saving merged model...")
model.save_pretrained_merged(
    "/workspace/models/phase1c_cot_merged",
    tokenizer,
    save_method="merged_16bit"
)

print("✅ Model merged successfully!")
EOF

# Upload to HuggingFace
huggingface-cli upload <your-org>/phase1c-cot-model /workspace/models/phase1c_cot_merged \
    --repo-type model
```

---

### Step 11: Quick Test

```bash
# Test the trained model
python3 << 'EOF'
from unsloth import FastLanguageModel
import torch

print("🧪 Testing trained model...\n")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/workspace/models/phase1c_cot_merged",
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

# Test prompt
test_instruction = "Explain the concept of recursion in programming with an example."

prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{test_instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=1024,
    temperature=0.7,
    do_sample=True,
    use_cache=True
)

response = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(response)

# Check for CoT structure
has_thinking = "<thinking>" in response
has_answer = "<answer>" in response
has_eos = response.endswith("<|end_of_text|>")

print(f"\n📊 Response Quality:")
print(f"   • Has <thinking>: {has_thinking}")
print(f"   • Has <answer>: {has_answer}")
print(f"   • Has EOS token: {has_eos}")
print(f"   • Length: {len(response)} chars")

if has_thinking and has_eos:
    print(f"\n✅ Model is generating CoT responses correctly!")
else:
    print(f"\n⚠️  Model may need more training or prompt adjustment")
EOF
```

---

### Step 12: Download Trained Model (Optional)

**Download to your local machine:**

```bash
# From local terminal
scp -P <PORT> -r root@<IP_ADDRESS>:/workspace/models/phase1c_cot_merged \
    /Users/vivekdurairaj/Projects/Cogumi-LLM/models/
```

---

### Step 13: Cleanup

```bash
# Stop Vast.ai instance to avoid charges
# From Vast.ai dashboard: Click "Destroy" on your instance

# Or via CLI
vast destroy instance <INSTANCE_ID>
```

---

## 📊 Expected Results

### Training Metrics
- **Initial Loss:** ~1.2-1.5
- **Final Loss:** ~0.4-0.6
- **Training Time:** 5-7 hours on H100
- **Peak VRAM:** ~65-70GB
- **Checkpoints:** Every 100 steps (~10 checkpoints total)

### Performance Improvement
- **Before:** 78.62% pass rate
- **After:** 90-95% pass rate (target)
- **Improvement:** +12-16 percentage points

### Output Model
- **Size:** ~10GB merged model (FP16)
- **Format:** HuggingFace Transformers compatible
- **LoRA Adapter:** ~260MB (if keeping separate)

---

## 🔧 Troubleshooting

### Issue: CUDA Out of Memory

**Solution:**
```bash
# Reduce batch size in training script
--batch_size 2  # Instead of 4
--gradient_accumulation_steps 16  # Instead of 8
```

### Issue: Training Too Slow

**Solution:**
```bash
# Ensure FlashAttention is enabled
python3 -c "import flash_attn; print('OK')"

# Check GPU utilization
nvidia-smi

# If <80%, increase batch size or reduce gradient accumulation steps
```

### Issue: Model Not Generating CoT

**Solution:**
- Check if training data has correct format
- Verify EOS token in training examples
- May need more epochs (try 5 instead of 3)
- Check learning rate (try 5e-6 if 3e-6 too conservative)

### Issue: Connection Lost During Training

**Solution:**
```bash
# Always use tmux for long training runs
tmux new -s training
python3 scripts/train_phase1c_cot.py [...]

# Reattach after reconnecting
tmux attach -t training
```

### Issue: HuggingFace Upload Fails

**Solution:**
```bash
# Check token permissions (need write access)
huggingface-cli whoami

# Re-login
huggingface-cli login

# Increase timeout for large files
export HF_HUB_TIMEOUT=600

# Try upload again
huggingface-cli upload [...]
```

---

## 💡 Tips & Best Practices

1. **Always use tmux** - Prevents losing progress if connection drops
2. **Monitor GPU usage** - Should be 80-95% during training
3. **Save checkpoints frequently** - Every 100 steps is good balance
4. **Test model before uploading** - Verify CoT generation works
5. **Document your run** - Save training logs and metrics
6. **Use early stopping** - If validation loss plateaus, stop early
7. **Keep base model backup** - Don't overwrite until new model validated

---

## 📝 Training Script Usage

Full command with all options (using deduplicated data):

```bash
python3 scripts/train_phase1c_cot.py \
    --data_path /workspace/data/phase1c_10k_with_cot_deduped.jsonl \
    --model_path /workspace/models/phase1c_base \
    --output_dir /workspace/models/phase1c_cot_trained \
    --learning_rate 3e-6 \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --max_seq_length 2048 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --save_steps 100 \
    --logging_steps 10 \
    --warmup_steps 50 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --use_flash_attention \
    --hf_token $HUGGINGFACE_TOKEN \
    --resume_from_checkpoint /workspace/models/phase1c_cot_trained/checkpoint-200  # Optional
```

**Required Parameters:**
- `--data_path`: Path to training JSONL file
- `--model_path`: Path to base model or HuggingFace model ID
- `--output_dir`: Where to save checkpoints and final model

**Optional Parameters:**
- `--learning_rate`: Default 3e-6 (conservative for fine-tuning)
- `--num_epochs`: Default 3
- `--batch_size`: Default 4 (per device)
- `--gradient_accumulation_steps`: Default 8 (effective batch = 32)
- `--max_seq_length`: Default 2048
- `--lora_rank`: Default 64
- `--lora_alpha`: Default 128
- `--save_steps`: Default 100
- `--logging_steps`: Default 10
- `--warmup_steps`: Default 50
- `--use_flash_attention`: Enable FlashAttention 2
- `--hf_token`: HuggingFace token for model upload
- `--resume_from_checkpoint`: Resume from specific checkpoint

---

## 📚 Additional Resources

- **Vast.ai Docs:** https://vast.ai/docs
- **Unsloth Docs:** https://github.com/unslothai/unsloth
- **Training Logs:** `/workspace/logs/training.log`
- **TensorBoard:** `tensorboard --logdir /workspace/models/phase1c_cot_trained/logs`

---

**Last Updated:** November 11, 2025  
**Tested On:** H100 80GB with PyTorch 2.6.0 + CUDA 12.4  
**Status:** ✅ Production Ready

---

*Happy Training! 🚀*
