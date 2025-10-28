# PHASE 1A RETRAINING PLAN - CRITICAL ARCHITECTURE FIX

## 🚨 CRITICAL ISSUE DISCOVERED

**Problem:** Original Phase 1A training used 4-bit quantized base model, causing catastrophic merge corruption.

**Evidence:**
- adapter_config.json: `"base_model_name_or_path": "unsloth/meta-llama-3.1-8b-instruct-unsloth-bnb-4bit"`
- Phase 1A merged validation: MATH 4% wins, 28% ties (expected 6%, 70%)
- Phase 1A merged validation: CODE 12% wins, 0% ties (expected 48%, 20%)
- Degradation: -42% ties, -36% code wins

**Root Cause:** Training on quantized base violates standard fine-tuning workflow:
```
❌ WRONG:  Train on 4-bit → Merge (rounding errors) → Corrupted model
✅ CORRECT: Train full precision → Merge cleanly → Optionally quantize
```

---

## 📋 RETRAINING CHECKLIST

### Step 1: Setup Vast.ai Instance (10 mins)

**GPU Requirements:**
- [ ] A100 80GB OR H100 (need 50-60GB for full precision)
- [ ] At least 200GB disk space
- [ ] Ubuntu 20.04+ with CUDA 12.1+

**Setup Commands:**
```bash
# Clone repo
cd /workspace
git clone https://github.com/yourusername/Cogumi-LLM.git
cd Cogumi-LLM

# Create venv
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements-h100-training.txt

# Verify GPU
nvidia-smi  # Should show A100 80GB or H100
python -c "import torch; print(torch.cuda.is_available())"
```

---

### Step 2: Verify Training Script (5 mins)

**Check corrected script:**
```bash
# Verify full precision configuration
grep "torch_dtype=torch.bfloat16" train_phase1a_fullprecision.py
grep "meta-llama/Meta-Llama-3.1-8B-Instruct" train_phase1a_fullprecision.py

# Should NOT contain:
grep "load_in_4bit" train_phase1a_fullprecision.py  # Should find nothing
grep "unsloth" train_phase1a_fullprecision.py       # Should find nothing
```

**Expected output:**
- `torch_dtype=torch.bfloat16` ✅
- `meta-llama/Meta-Llama-3.1-8B-Instruct` ✅
- NO `load_in_4bit=True` ✅
- NO `unsloth` references ✅

---

### Step 3: Download Dataset (10 mins)

```bash
# Verify dataset exists
ls -lh data/phase1/public_500k_filtered.jsonl

# Should show: ~500MB file, 600,000 examples

# If missing, download from backup
# (Add backup location here)
```

---

### Step 4: Start Training (12-16 hours)

**Launch training:**
```bash
# Start in tmux (survives disconnects)
tmux new -s phase1a_training

# Activate venv
source venv/bin/activate

# Start training
python train_phase1a_fullprecision.py 2>&1 | tee logs/phase1a_fullprecision_training.log

# Detach: Ctrl+B then D
# Reattach later: tmux attach -t phase1a_training
```

**Monitor progress:**
```bash
# In separate tmux pane (Ctrl+B then ")
watch -n 60 nvidia-smi

# Check training log
tail -f logs/phase1a_fullprecision_training.log

# Check tensorboard (optional)
tensorboard --logdir data/checkpoints/phase1a_fullprecision/runs --port 6006
```

**Expected timeline:**
- Steps: ~28,000 (600K examples, batch=8, accumulation=4)
- Time: 12-16 hours on A100 80GB
- Cost: ~$30-40 on Vast.ai

**Success indicators:**
- Training loss decreasing steadily
- No OOM errors
- Checkpoints saving regularly
- GPU utilization 90-100%

---

### Step 5: Merge Adapter (30 mins)

**After training completes:**
```bash
# Verify adapter saved
ls -lh data/checkpoints/phase1a_fullprecision/

# Should contain:
# - adapter_model.safetensors (~1GB)
# - adapter_config.json
# - training_args.bin
# - trainer_state.json

# Run merge script
python scripts/merge_adapter_fullprecision.py 2>&1 | tee logs/phase1a_merge_fullprecision.log

# Verify merged model
ls -lh checkpoints/phase1a_merged_fullprecision/

# Should contain:
# - model-*.safetensors (~16GB total)
# - config.json
# - tokenizer files
```

**CRITICAL VERIFICATION:**
Check adapter_config.json in NEW checkpoint:
```bash
cat data/checkpoints/phase1a_fullprecision/adapter_config.json
```

Should show:
```json
{
  "base_model_name_or_path": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  ...
}
```

**NOT** `unsloth/...bnb-4bit` ❌

---

### Step 6: Validate Merged Model (2 hours)

**Run comprehensive benchmarks:**
```bash
# MATH validation (50 examples)
python scripts/automated_gpt4_benchmark.py \
  --model_path checkpoints/phase1a_merged_fullprecision \
  --dataset_name gsm8k \
  --num_samples 50 \
  --output_dir results/phase1a_fullprecision_validation

# CODE validation (50 examples - 25 HumanEval + 25 MBPP)
python scripts/automated_gpt4_benchmark.py \
  --model_path checkpoints/phase1a_merged_fullprecision \
  --dataset_name humaneval \
  --num_samples 25 \
  --output_dir results/phase1a_fullprecision_validation

python scripts/automated_gpt4_benchmark.py \
  --model_path checkpoints/phase1a_merged_fullprecision \
  --dataset_name mbpp \
  --num_samples 25 \
  --output_dir results/phase1a_fullprecision_validation
```

**Expected results (SUCCESS CRITERIA):**
```
MATH (GSM8K):
✅ Wins: 2-4 (4-8%)
✅ Ties: 30-40 (60-80%)  ← KEY METRIC
✅ Losses: 8-18 (16-36%)

CODE (HumanEval + MBPP):
✅ Wins: 20-28 (40-56%)  ← KEY METRIC
✅ Ties: 8-14 (16-28%)
✅ Losses: 10-22 (20-44%)
```

**If results match:** ✅ Phase 1A is CORRECT, proceed to Phase 1B

**If results differ significantly:** ❌ STOP, investigate further

---

### Step 7: Phase 1B Preparation (if validation passes)

**Extract failures from corrected model:**
```bash
python scripts/extract_all_failures_prepare_training.py \
  --model_path checkpoints/phase1a_merged_fullprecision \
  --output_dir data/phase1b_failures \
  --num_samples_per_dataset 1000
```

**Expected output:**
- 200-400 real failures extracted
- NO tie examples (model already correct)
- Data saved to `data/phase1b_failures/combined_losses.jsonl`

**Then train Phase 1B:**
```bash
python scripts/train_phase1b_on_adapter.sh \
  --base_model checkpoints/phase1a_merged_fullprecision \
  --training_data data/phase1b_failures/combined_losses.jsonl
```

---

## 🎯 SUCCESS CRITERIA

### Phase 1A Training:
- [ ] Training completes without OOM errors
- [ ] Final adapter saved to `data/checkpoints/phase1a_fullprecision/`
- [ ] adapter_config.json shows `meta-llama/Meta-Llama-3.1-8B-Instruct` (NOT unsloth)
- [ ] Training loss converges smoothly

### Phase 1A Merge:
- [ ] Merge completes without errors
- [ ] Merged model saved to `checkpoints/phase1a_merged_fullprecision/`
- [ ] Model size ~16GB (bfloat16)
- [ ] NO 4-bit rounding warnings in logs

### Phase 1A Validation:
- [ ] MATH: 60-80% ties (vs 28% in corrupted model)
- [ ] CODE: 40-56% wins (vs 12% in corrupted model)
- [ ] No catastrophic degradation patterns
- [ ] Results within 10% of expected baseline

---

## 🚨 RED FLAGS - STOP IF YOU SEE THESE

**During Training:**
- ❌ OOM errors → Reduce batch size, increase gradient accumulation
- ❌ adapter_config.json shows `unsloth` → Wrong base model loaded
- ❌ Training loss not decreasing → Check learning rate, data quality
- ❌ Loss spikes/NaNs → Check gradient clipping, mixed precision settings

**During Merge:**
- ❌ 4-bit rounding warnings → Loaded quantized base instead of full precision
- ❌ Merge errors → Check adapter and base model compatibility
- ❌ Merged model size <10GB → Model quantized during merge (wrong!)

**During Validation:**
- ❌ MATH ties <50% → Model still corrupted, investigate merge
- ❌ CODE wins <30% → Model degraded, check training quality
- ❌ Results match old corrupted model → Used wrong checkpoint

---

## 📊 COST & TIME ESTIMATES

| Task | Time | Cost (Vast.ai) |
|------|------|----------------|
| Setup | 30 mins | $0 |
| Training | 12-16 hours | $30-40 |
| Merge | 30 mins | $1 |
| Validation | 2 hours | $5 |
| **TOTAL** | **~18 hours** | **~$36-46** |

**Note:** Cost assumes A100 80GB at ~$2.50/hr on Vast.ai

---

## 📁 FILE STRUCTURE AFTER COMPLETION

```
Cogumi-LLM/
├── data/
│   └── checkpoints/
│       ├── phase1a_fullprecision/          # NEW - Correct adapter
│       │   ├── adapter_model.safetensors
│       │   └── adapter_config.json         # Shows meta-llama base
│       └── final/                          # OLD - DO NOT USE
│           └── adapter_config.json         # Shows unsloth 4-bit base
├── checkpoints/
│   ├── phase1a_merged_fullprecision/       # NEW - Correct merged (~16GB)
│   └── phase1a_merged/                     # OLD - Corrupted (~5.5GB)
├── results/
│   └── phase1a_fullprecision_validation/   # NEW - Should show 60-80% ties
└── logs/
    ├── phase1a_fullprecision_training.log
    └── phase1a_merge_fullprecision.log
```

---

## 🔄 ROLLBACK PLAN (If Something Goes Wrong)

**If training fails:**
1. Check logs: `logs/phase1a_fullprecision_training.log`
2. Verify GPU memory: `nvidia-smi`
3. Reduce batch size in config if OOM
4. Restart training (will resume from last checkpoint)

**If merge fails:**
1. Verify adapter exists: `ls data/checkpoints/phase1a_fullprecision/`
2. Check adapter_config.json for correct base model
3. Manually load adapter + base to test compatibility

**If validation fails:**
1. Compare with corrupted model results (28% ties)
2. If similar → merge issue, check for quantization
3. If different but still poor → training issue, check data quality
4. Re-run validation with more samples (200+) to verify

---

## 📚 REFERENCES

- Issue Log: `docs/ISSUES_LOG.md` (2025-01-25 entry)
- Technical Spec: `docs/technical_specification.md` (Phase 1A section)
- Original Training: `train_qlora_optimized.py` (DO NOT USE - has 4-bit bug)
- Corrected Training: `train_phase1a_fullprecision.py` (USE THIS)
- Corrected Merge: `scripts/merge_adapter_fullprecision.py` (USE THIS)

---

## ✅ COMPLETION CHECKLIST

- [ ] Vast.ai instance configured (A100 80GB or H100)
- [ ] Training completed successfully (~16 hours)
- [ ] Adapter saved and verified (meta-llama base, NOT unsloth)
- [ ] Merge completed without 4-bit warnings
- [ ] Validation shows 60-80% MATH ties (vs 28% corrupted)
- [ ] Validation shows 40-56% CODE wins (vs 12% corrupted)
- [ ] Phase 1B preparation script ready
- [ ] Documentation updated with results
- [ ] Old corrupted checkpoints marked as DO_NOT_USE

**When all checkboxes are complete:** ✅ Phase 1A is CORRECTLY trained and ready for Phase 1B!
