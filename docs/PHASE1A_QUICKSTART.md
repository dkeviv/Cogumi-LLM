# Phase 1A Training - Quick Start Guide

## Overview
Train LLAMA-3.2-8B on 640,637 curated examples using QLoRA on Google Colab Pro+ A100 GPU.

**Duration**: 36-48 hours  
**Cost**: Included in Colab Pro+ subscription ($50/month)  
**GPU**: A100 40GB required  

---

## Setup Steps

### 1. Open Colab Notebook

1. Upload `notebooks/Phase1A_Training_Colab.ipynb` to Google Drive
2. Open with Google Colab
3. Or directly open: [Open in Colab](https://colab.research.google.com/)

### 2. Select A100 GPU

1. Click **Runtime** → **Change runtime type**
2. Select **Hardware accelerator**: GPU
3. Select **GPU type**: A100
4. Click **Save**

### 3. Get HuggingFace Token

1. Visit: https://huggingface.co/settings/tokens
2. Click **New token** → Create read token
3. Accept LLAMA license: https://huggingface.co/meta-llama/Llama-3.2-8B
4. Copy token for use in notebook

### 4. Run Training

1. Click **Runtime** → **Run all**
2. Paste HF_TOKEN when prompted (cell 4)
3. Monitor training in TensorBoard (cell 7)
4. Training will run for 36-48 hours

### 5. Handle Session Timeouts

Colab Pro+ sessions disconnect after 24 hours. To resume:

1. Reconnect to runtime
2. Rerun setup cells (1-5)
3. Run resume cell (cell 8)
4. Training continues from last checkpoint

---

## Files Created

### Local Files (Before Upload)
- `configs/base_training.yaml` - Axolotl configuration
- `notebooks/Phase1A_Training_Colab.ipynb` - Training notebook
- `data/phase1/public_500k_filtered.jsonl` - Dataset (640K examples)

### Colab Files (During Training)
- `data/checkpoints/llama-3.2-8b-phase1a/` - Checkpoints (every 1000 steps)
- `data/checkpoints/llama-3.2-8b-phase1a/training.log` - Training logs
- `data/checkpoints/llama-3.2-8b-phase1a/trainer_state.json` - Training state

### Output Files (After Training)
- `models/llama-3.2-8b-phase1a-merged/` - Final merged model (16.6GB)
- `llama-3.2-8b-phase1a-merged.tar.gz` - Compressed for download

---

## Training Monitoring

### TensorBoard (Real-time)
Open in Colab:
```python
%load_ext tensorboard
%tensorboard --logdir data/checkpoints/llama-3.2-8b-phase1a
```

### Check Progress (Manual)
```python
# See cell 9 in notebook
import json
with open('data/checkpoints/llama-3.2-8b-phase1a/trainer_state.json') as f:
    state = json.load(f)
    print(f"Step: {state['global_step']}/60000")
    print(f"Progress: {state['global_step']/60000*100:.1f}%")
```

### Expected Loss Curve
- **Epoch 1** (0-20K steps): Loss 2.8 → 1.6
- **Epoch 2** (20K-40K steps): Loss 1.6 → 1.3
- **Epoch 3** (40K-60K steps): Loss 1.3 → 1.2

---

## Troubleshooting

### Problem: "No A100 GPU available"
**Solution**: Colab Pro+ gives priority access but not guaranteed. Try:
- Wait 30 minutes and reconnect
- Try different time of day (less busy hours)
- Contact Colab support if persistent

### Problem: "Out of Memory (OOM)"
**Solution**: Reduce batch size in config:
```yaml
micro_batch_size: 2  # Changed from 4
gradient_accumulation_steps: 16  # Changed from 8
```

### Problem: "Session disconnected"
**Solution**: Normal for 24+ hour training. Resume from checkpoint:
1. Reconnect to runtime
2. Run setup cells (1-5)
3. Run resume cell (8)

### Problem: "Loss not decreasing"
**Solution**: Check TensorBoard:
- If stuck at high loss (>2.5) after 5K steps → Restart with lower LR
- If validation loss > training loss → Overfitting, use early stopping

### Problem: "Training very slow"
**Solution**: Check GPU utilization:
```python
!nvidia-smi
```
- Should show 85-95% GPU utilization
- If low, check `dataloader_num_workers` in config

---

## Expected Timeline

| Phase | Steps | Duration | Status Check |
|-------|-------|----------|--------------|
| Setup | - | 10-15 min | Dependencies installed |
| Epoch 1 | 0-20K | 12-14 hours | Loss 2.8→1.6 |
| Epoch 2 | 20K-40K | 12-14 hours | Loss 1.6→1.3 |
| Epoch 3 | 40K-60K | 12-14 hours | Loss 1.3→1.2 |
| Merge | - | 10-15 min | 16.6GB model created |
| **Total** | **60K** | **36-48 hrs** | **Ready for Phase 2** |

---

## After Training Completes

### 1. Merge LoRA Adapters
Run cell 10 in notebook to merge LoRA weights into base model.

### 2. Test Model
Run cell 11 to verify model works correctly.

### 3. Download Model
**Option A**: Download compressed tar.gz (cell 12)
**Option B**: Upload to HuggingFace Hub (cell 12, alternative)

### 4. Verify Benchmarks
Expected performance:
- MMLU: 78-82%
- HumanEval: 58-62%
- GSM8K: 86-88%
- BBH: 72-76%

### 5. Proceed to Phase 2
Compression pipeline (95% size reduction: 16.6GB → 520MB)

---

## Cost Breakdown

| Item | Cost | Notes |
|------|------|-------|
| Colab Pro+ | $50/month | Includes A100 access |
| HuggingFace | Free | Public model hosting |
| **Total** | **$50** | One-time month subscription |

Compare to cloud GPU rental:
- RunPod A100: $1.12/hr × 45 hrs = $505
- **Savings**: $455 with Colab Pro+

---

## Next Steps

1. ✅ Complete Phase 1A training (this guide)
2. ⏳ Evaluate on benchmarks
3. ⏳ Phase 2: Compression (520MB base model)
4. ⏳ Phase 3: Domain modifiers (code, reasoning, automation)
5. ⏳ Phase 4: Router system
6. ⏳ Phase 5: Deployment

---

## Support

**Issues**: Create issue on GitHub: https://github.com/dkeviv/Cogumi-LLM/issues  
**Documentation**: See `docs/COLAB_PRO_PLUS_GUIDE.md` for detailed Colab setup  
**Technical Spec**: See `docs/technical_specification.md` for methodology  

---

**Last Updated**: October 19, 2025  
**Version**: 1.0  
**Status**: Ready for execution
