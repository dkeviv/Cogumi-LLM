# Vast.ai Self-Critique Execution Guide

**Date:** November 1, 2025  
**Task:** Run self-critique generation (step9 + step10) on 7,331 failures  
**Cost:** ~$0.75 (2-3 hours on A10)  
**Alternative to:** 30-40 hours on local Mac CPU

---

## 📋 Prerequisites

### Files to Prepare for Upload

1. **Model (15GB):**
   ```
   Phase1A_2_0/models/phase1a_merged_10gb/
   ├── config.json
   ├── generation_config.json
   ├── model-00001-of-00004.safetensors
   ├── model-00002-of-00004.safetensors
   ├── model-00003-of-00004.safetensors
   ├── model-00004-of-00004.safetensors
   ├── model.safetensors.index.json
   ├── special_tokens_map.json
   ├── tokenizer.json
   └── tokenizer_config.json
   ```

2. **Input Data (~8MB):**
   ```
   Phase 1B_2_0/data/haiku_replay/phase1c_failures_haiku_replay.jsonl
   ```

3. **Scripts:**
   ```
   Phase 1B_2_0/step9_self_critique_rewrite.py
   Phase 1B_2_0/step10_evaluate_self_critique_local.py
   scripts/run_self_critique_on_vast.sh
   requirements.txt
   ```

**Total Upload Size:** ~15.1GB (mostly model)

---

## 🚀 Step-by-Step Instructions

### Step 1: Find and Rent GPU Instance

1. **Go to:** https://vast.ai/console/create/
2. **Search Settings:**
   - GPU: "RTX A10" or "A10"
   - VRAM: ≥40GB
   - Storage: ≥25GB
   - Bandwidth: ≥100 Mbps (faster upload)
   - CUDA: 12.1+ recommended

3. **Filter by Price:**
   - Sort by "$/hr" (cheapest first)
   - Look for: $0.20-0.30/hour range
   - Total cost: ~$0.60-0.90 for 3 hours

4. **Rent Instance:**
   - Click "RENT" on your chosen instance
   - Select "PyTorch" template (comes with Python + CUDA)
   - Storage: 25GB minimum
   - Click "RENT INSTANCE"

### Step 2: Connect to Instance

1. **Wait for Instance to Start:**
   - Status will change to "Running" (2-5 minutes)
   - Get SSH connection details from console

2. **Connect via SSH:**
   ```bash
   # Example (your actual command will be different)
   ssh -p 12345 root@ssh.vast.ai
   ```

3. **Verify GPU:**
   ```bash
   nvidia-smi
   # Should show: RTX A10, 40GB VRAM
   ```

### Step 3: Upload Files to Vast.ai

**Option A: Using SCP (Recommended for speed)**

From your local Mac terminal:

```bash
# Navigate to project root
cd /Users/vivekdurairaj/Projects/Cogumi-LLM

# Upload model (15GB - will take time)
scp -P 12345 -r Phase1A_2_0/models/phase1a_merged_10gb root@ssh.vast.ai:/workspace/

# Upload failure data
scp -P 12345 "Phase 1B_2_0/data/haiku_replay/phase1c_failures_haiku_replay.jsonl" root@ssh.vast.ai:/workspace/

# Upload scripts
scp -P 12345 "Phase 1B_2_0/step9_self_critique_rewrite.py" root@ssh.vast.ai:/workspace/
scp -P 12345 "Phase 1B_2_0/step10_evaluate_self_critique_local.py" root@ssh.vast.ai:/workspace/
scp -P 12345 scripts/run_self_critique_on_vast.sh root@ssh.vast.ai:/workspace/
scp -P 12345 requirements.txt root@ssh.vast.ai:/workspace/
```

**Upload Progress:**
- Model: 10-30 minutes (depends on bandwidth)
- Data + Scripts: 1-2 minutes

**Option B: Using Vast.ai File Manager**
1. Go to your instance in Vast.ai console
2. Click "Files" tab
3. Upload via web interface (slower, but easier)

### Step 4: Setup Environment on Vast.ai

SSH into your instance, then run:

```bash
# Navigate to workspace
cd /workspace

# Create directory structure
mkdir -p "Phase 1B_2_0/data/haiku_replay"
mkdir -p "Phase 1B_2_0/data/self_critique"
mkdir -p "Phase1A_2_0/models"
mkdir -p scripts

# Move uploaded files to correct locations
mv phase1a_merged_10gb Phase1A_2_0/models/
mv phase1c_failures_haiku_replay.jsonl "Phase 1B_2_0/data/haiku_replay/"
mv step9_self_critique_rewrite.py "Phase 1B_2_0/"
mv step10_evaluate_self_critique_local.py "Phase 1B_2_0/"
mv run_self_critique_on_vast.sh scripts/

# Verify files are in place
ls -lh Phase1A_2_0/models/phase1a_merged_10gb/
ls -lh "Phase 1B_2_0/data/haiku_replay/"
```

### Step 5: Install Dependencies

```bash
# Update pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# Install bitsandbytes for 4-bit quantization (GPU acceleration)
pip install bitsandbytes==0.42.0

# Verify installations
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import bitsandbytes; print('BitsAndBytes: OK')"
```

**Expected Install Time:** 5-10 minutes

### Step 6: Run Self-Critique Pipeline

**Start in a tmux session** (so it continues if SSH disconnects):

```bash
# Start tmux
tmux new -s self_critique

# Run the full pipeline (step9 + step10)
bash scripts/run_self_critique_on_vast.sh \
  /workspace/Phase1A_2_0/models/phase1a_merged_10gb \
  "/workspace/Phase 1B_2_0/data/haiku_replay/phase1c_failures_haiku_replay.jsonl" \
  "/workspace/Phase 1B_2_0/data/self_critique/rewrite_full.jsonl" \
  7331
```

**What This Does:**
1. Loads Phase 1A model with 4-bit quantization
2. Processes all 7,331 failures
3. Generates critique + corrected answer for each
4. Automatically runs step10 evaluation
5. Outputs training-ready datasets

**Expected Runtime:** 2-3 hours on A10

**To Detach from tmux:** Press `Ctrl+B`, then `D`  
**To Reattach:** `tmux attach -t self_critique`

### Step 7: Monitor Progress

**Check progress periodically:**

```bash
# Reattach to tmux
tmux attach -t self_critique

# Or check output file size
wc -l "/workspace/Phase 1B_2_0/data/self_critique/rewrite_full.jsonl"

# View latest item
tail -1 "/workspace/Phase 1B_2_0/data/self_critique/rewrite_full.jsonl" | python3 -m json.tool
```

**Progress Indicators:**
- Logs every 100 items
- File grows as items are processed
- 7,331 lines = complete

### Step 8: Download Results

**Once complete, download outputs back to Mac:**

From your local Mac terminal:

```bash
cd /Users/vivekdurairaj/Projects/Cogumi-LLM

# Download main rewrite file (7,331 items with critiques)
scp -P 12345 root@ssh.vast.ai:"/workspace/Phase 1B_2_0/data/self_critique/rewrite_full.jsonl" \
  "Phase 1B_2_0/data/self_critique/"

# Download evaluation results
scp -P 12345 -r root@ssh.vast.ai:"/workspace/Phase 1B_2_0/data/self_critique/eval" \
  "Phase 1B_2_0/data/self_critique/"
```

**Files to Download:**
- `rewrite_full.jsonl` (~50MB) - All critiques + corrected answers
- `eval/summary.json` - Pass rate, metrics
- `eval/improved_forward.jsonl` - Training-ready data (pass=True only)
- `eval/improved_reverse.jsonl` - Bidirectional pairs

### Step 9: Verify Results Locally

```bash
# Check item count
wc -l "Phase 1B_2_0/data/self_critique/rewrite_full.jsonl"
# Expected: 7331

# View summary
cat "Phase 1B_2_0/data/self_critique/eval/summary.json"

# Check improved dataset size
wc -l "Phase 1B_2_0/data/self_critique/eval/improved_forward.jsonl"
```

### Step 10: Destroy Instance

**IMPORTANT:** Don't forget to destroy the instance to stop billing!

1. Go to Vast.ai console
2. Find your instance
3. Click "DESTROY"
4. Confirm destruction

**Total Cost:** ~$0.60-0.90 (depending on runtime and GPU pricing)

---

## 🔧 Troubleshooting

### Issue: Upload Too Slow
**Solution:** Compress model before upload:
```bash
# On Mac
tar -czf phase1a_model.tar.gz Phase1A_2_0/models/phase1a_merged_10gb/

# Upload compressed
scp -P 12345 phase1a_model.tar.gz root@ssh.vast.ai:/workspace/

# Extract on Vast.ai
tar -xzf phase1a_model.tar.gz
```

### Issue: Out of Memory on GPU
**Solution:** Already handled - script uses 4-bit quantization (`--load_in_4bit`)

### Issue: SSH Disconnects
**Solution:** Use tmux (instructions in Step 6)

### Issue: Script Fails
**Check:**
```bash
# Verify model loaded correctly
python3 -c "from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained('/workspace/Phase1A_2_0/models/phase1a_merged_10gb'); print('Model OK')"

# Check input file
head -1 "/workspace/Phase 1B_2_0/data/haiku_replay/phase1c_failures_haiku_replay.jsonl"
```

---

## 📊 Expected Outputs

### rewrite_full.jsonl Format:
```json
{
  "id": 0,
  "category": "code",
  "instruction": "...",
  "reference_answer": "...",
  "previous_output": "...",
  "critique": "The previous answer incorrectly...",
  "final_answer": "Corrected: ..."
}
```

### eval/summary.json Format:
```json
{
  "count": 7331,
  "passes": 6500,
  "fails": 831,
  "pass_rate": 88.7,
  "threshold": 0.74,
  "model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

---

## 🎯 Next Steps After Completion

1. **Run step11 (Haiku replay uplift):**
   ```bash
   python3 "Phase 1B_2_0/step11_rejudge_uplift_sample.py" \
     --input_jsonl "Phase 1B_2_0/data/self_critique/rewrite_full.jsonl" \
     --output_dir "Phase 1B_2_0/data/self_critique/uplift_full"
   ```

2. **Run step12 (Semantic uplift):**
   ```bash
   python3 "Phase 1B_2_0/step12_semantic_uplift.py" \
     --input_jsonl "Phase 1B_2_0/data/self_critique/rewrite_full.jsonl" \
     --output_dir "Phase 1B_2_0/data/self_critique/semantic_uplift_full"
   ```

3. **Prepare Phase 1C training dataset** (combine with 600K original data)

---

## 💡 Tips for Success

1. **Monitor costs:** Set billing alerts on Vast.ai
2. **Use tmux:** Prevents loss if SSH disconnects
3. **Test first:** Can run with `--limit 100` to test before full run
4. **Backup outputs:** Download results immediately after completion
5. **Destroy instance:** Don't forget to stop billing!

---

## 📞 Support

- **Vast.ai Docs:** https://vast.ai/docs/
- **Script Issues:** Check `Phase 1B_2_0/step9_self_critique_rewrite.py` docstring
- **Resume:** Script supports `--resume` flag (auto-enabled in runner)

**Good luck! The full run should complete in 2-3 hours on A10.**
