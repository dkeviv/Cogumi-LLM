# H100 Training Quick Reference

## Installation (ONE Command!)

```bash
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
```

**That's it!** Unsloth installs:
- PyTorch 2.4.0+cu121
- Transformers (latest stable)
- PEFT, BitsAndBytes, TRL
- Flash Attention
- All other dependencies

---

## Workflow

### 1. Install Unsloth (Cell 9)
```python
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
```
⏱️ 5-10 minutes

### 2. Restart Kernel
- Kernel → Restart Kernel
- Wait 15 seconds

### 3. Verify (Cell 12)
```python
from unsloth import FastLanguageModel
```
✅ Should show all packages working

### 4. Authenticate (Cell 15)
```python
from huggingface_hub import login
login()
```

### 5. Upload Dataset
- Use JupyterLab UI
- Upload `public_500k_filtered.jsonl` to `/data/Cogumi-LLM/data/phase1/`

### 6. Create Training Script (Cell 19)
```python
%%writefile /data/Cogumi-LLM/train_qlora_h100.py
# Auto-generates Unsloth-optimized training script
```

### 7. Start Training (Cell 22)
```python
# Launches training in tmux background session
# Runs for 8-9 hours
# Saves checkpoints every 1000 steps
```

---

## Key Benefits

✅ **No Dependency Conflicts** - Unsloth handles everything  
✅ **2x Faster Training** - 40-45 it/s instead of 20-25 it/s  
✅ **50% Cost Savings** - 8 hours instead of 16 hours  
✅ **Just Works™** - No configuration needed  

---

## Monitoring Training

### Option 1: Tmux (Recommended)
```bash
# In JupyterLab terminal
tmux attach -t training
# Detach: Ctrl+B, then D
```

### Option 2: GPU Usage (Cell 25)
```python
# Runs nvidia-smi every 5 seconds
# Shows GPU memory and utilization
```

### Option 3: Logs
```bash
# Training logs location
/data/Cogumi-LLM/data/checkpoints/llama-3.1-8b-phase1a-h100/
```

---

## Expected Results

| Metric | Value |
|--------|-------|
| Training Time | 8-9 hours |
| Speed | 40-45 iterations/second |
| Checkpoints | Every 1000 steps (~22 total) |
| VRAM Usage | ~50GB (out of 80GB) |
| Cost | ~$9-10 ($1.20/hr × 8hr) |
| Final Model | ~8GB (4-bit LoRA adapters) |

---

## Troubleshooting

### "ImportError: No module named 'unsloth'"
→ Run installation cell again  
→ Check error messages

### "FastLanguageModel not found"
→ Restart kernel after installation  
→ Wait 15 seconds before running verification

### Training slower than expected
→ Check nvidia-smi for GPU usage  
→ Verify H100 GPU (not A100/V100)  
→ Check no other processes using GPU

### Out of memory error
→ Reduce batch size from 8 to 4  
→ Increase gradient_accumulation_steps from 4 to 8

---

## Files Generated

| Location | Description | Size |
|----------|-------------|------|
| `/data/Cogumi-LLM/train_qlora_h100.py` | Training script | ~4KB |
| `/data/Cogumi-LLM/data/phase1/public_500k_filtered.jsonl` | Dataset | 870MB |
| `/data/Cogumi-LLM/data/checkpoints/llama-3.1-8b-phase1a-h100/` | Checkpoints | ~8GB each |
| `/data/Cogumi-LLM/.hf_token` | HF token (secure) | <1KB |

---

## After Training Completes

### 1. Merge LoRA Adapters
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/data/Cogumi-LLM/data/checkpoints/llama-3.1-8b-phase1a-h100",
    max_seq_length=2048,
)

# Save merged model
model.save_pretrained_merged(
    "/data/Cogumi-LLM/models/llama-3.1-8b-phase1a-merged",
    tokenizer,
)
```

### 2. Test Model
```python
inputs = tokenizer("Write a Python function to calculate fibonacci", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=512)
print(tokenizer.decode(outputs[0]))
```

### 3. Download Model
```bash
# Compress for download
tar -czf llama-3.1-8b-phase1a-merged.tar.gz -C /data/Cogumi-LLM/models llama-3.1-8b-phase1a-merged

# Download via JupyterLab UI or scp
```

---

## Support

- **Unsloth Docs**: https://docs.unsloth.ai/
- **Unsloth GitHub**: https://github.com/unslothai/unsloth
- **Vast.ai Support**: https://vast.ai/support
- **Migration Guide**: `docs/H100_UNSLOTH_MIGRATION.md`
