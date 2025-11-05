# Quick Start: Benchmark on New Instance

## ✅ Your Model Location Confirmed

**Path:** `/workspace/data/Cogumi-LLM/checkpoints/final`

---

## 🚀 Quick Setup Commands

SSH into your new instance and run these commands:

```bash
# 1. Verify your model is there
ls -lh /workspace/data/Cogumi-LLM/checkpoints/final/

# Expected output:
# adapter_config.json
# adapter_model.safetensors  (~260MB)
# tokenizer files

# 2. Verify benchmark script is there
ls -lh /workspace/data/Cogumi-LLM/scripts/automated_gpt4_benchmark.py

# 3. Install dependencies (5-10 mins)
pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install -q openai datasets pandas rich transformers accelerate bitsandbytes

# 4. Verify installation
python3 -c "import torch; print(f'✅ PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'✅ CUDA: {torch.cuda.is_available()}')"
python3 -c "import unsloth; print('✅ Unsloth OK')"
python3 -c "import openai; print('✅ OpenAI SDK OK')"

# 5. Start Jupyter
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser
```

---

## 📓 Upload and Run Notebook

### **Step 1: Access Jupyter**
- Look for URL in terminal output: `http://127.0.0.1:8888/lab?token=...`
- Replace `127.0.0.1` with your Vast.ai instance IP
- Open in browser

### **Step 2: Upload Notebook**
- In Jupyter, navigate to `/workspace/data/Cogumi-LLM/notebooks/`
- Click upload button (↑ icon)
- Select `H100_Phase1B_Benchmark.ipynb` from your Mac
- Click "Upload"

### **Step 3: Run Benchmark**
1. Open `H100_Phase1B_Benchmark.ipynb`
2. Run Cell 1: Should show `✅ Model found at: /workspace/data/Cogumi-LLM/checkpoints/final`
3. Run Cell 2: Install dependencies (if not already done)
4. Run Cell 3: Enter OpenAI API key (get from https://platform.openai.com/api-keys)
5. Run remaining cells: Benchmark will take ~30-60 mins

---

## 🎯 Your Directory Structure

```
/workspace/data/Cogumi-LLM/
├── checkpoints/
│   ├── final/                          ← Your trained model here! ✅
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   └── tokenizer files
│   └── checkpoint-240240/              ← Last checkpoint
├── scripts/
│   └── automated_gpt4_benchmark.py     ← Benchmark script ✅
├── notebooks/
│   └── (upload H100_Phase1B_Benchmark.ipynb here)
└── benchmark_results/                  ← Results will be saved here
    └── (created automatically)
```

---

## ✅ Verification Checklist

Before running benchmark, verify:

- [ ] Model exists: `ls /workspace/data/Cogumi-LLM/checkpoints/final/`
- [ ] Script exists: `ls /workspace/data/Cogumi-LLM/scripts/automated_gpt4_benchmark.py`
- [ ] Dependencies installed: `python3 -c "import unsloth, openai"`
- [ ] Jupyter running: Can access in browser
- [ ] Notebook uploaded: See it in Jupyter file browser
- [ ] OpenAI API key ready: From https://platform.openai.com/api-keys

---

## 🔧 If Issues Occur

### **Model not found:**
```bash
# Find model location
find /workspace -name "adapter_model.safetensors" 2>/dev/null
```

### **Script not found:**
```bash
# Find benchmark script
find /workspace -name "automated_gpt4_benchmark.py" 2>/dev/null
```

### **Import errors:**
```bash
# Reinstall packages
pip install --upgrade transformers accelerate bitsandbytes
```

### **GPU not detected:**
```bash
# Check GPU
nvidia-smi

# Check PyTorch CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## 💰 Expected Costs

- **GPU time:** ~1-2 hours @ $0.15-0.20/hr = **$0.30-0.40**
- **OpenAI API:** 300 GPT-4 comparisons = **$5-10**
- **Total:** **~$6-11**

---

## 📊 Expected Results

After ~1 hour, you should see:

```
📊 PHASE 1B BENCHMARK RESULTS
═══════════════════════════════════════
🎯 Overall Performance: 75-82% of GPT-4
   Rating: Good/Excellent
   Total comparisons: 300

📋 Results by Category:
math       : 78.5% | W:35 L:10 T:5  | 🟢 Good
code       : 82.1% | W:38 L:8  T:4  | ✅ Excellent
reasoning  : 76.3% | W:34 L:12 T:4  | 🟢 Good
...
```

---

## 🎊 After Benchmark Completes

1. **Download results:**
   - Right-click `/workspace/data/Cogumi-LLM/benchmark_results/` in Jupyter
   - Select "Download as Archive"

2. **Download notebook output:**
   - File → Download → Download as .ipynb

3. **Destroy instance** (stop billing!)
   - Go to https://cloud.vast.ai/instances/
   - Click "Destroy" on your instance

4. **Keep persistent volume** (for Phase 2!)
   - Don't delete the volume
   - You'll need it for compression phase

---

## 🚀 You're Ready!

Your model is at: `/workspace/data/Cogumi-LLM/checkpoints/final` ✅

Just install dependencies → upload notebook → run benchmark! 🎯

---
