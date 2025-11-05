# Phase 1B Benchmark - Terminal Mode

## 🚀 Run Benchmark from Terminal (Recommended)

**Why use terminal mode:**
- ✅ Survives browser disconnections
- ✅ Runs in background (no need to keep browser open)
- ✅ Can monitor with `tail -f`
- ✅ More reliable for 6-hour runs

---

## 📋 Quick Start on Vast.ai

### Step 1: Upload Files

Upload these files to Vast.ai (if not already there):
1. `scripts/automated_gpt4_benchmark.py` (fixed version)
2. `scripts/run_phase1b_benchmark_standalone.py` (new standalone script)

**Location:** `/workspace/data/Cogumi-LLM/scripts/`

### Step 2: Set API Key

```bash
# In Vast.ai SSH or Jupyter terminal:
export OPENAI_API_KEY="sk-your-api-key-here"
```

### Step 3: Run Benchmark in Background

```bash
cd /workspace/data/Cogumi-LLM

# Run with nohup (continues even if SSH disconnects)
nohup python3 scripts/run_phase1b_benchmark_standalone.py \
    --samples 50 \
    > benchmark_log.txt 2>&1 &

# Note the process ID (e.g., [1] 12345)
```

### Step 4: Monitor Progress

```bash
# Watch live progress
tail -f benchmark_log.txt

# Exit tail with Ctrl+C (benchmark keeps running)

# Check if still running
ps aux | grep run_phase1b_benchmark

# View last 100 lines
tail -100 benchmark_log.txt
```

---

## 🎯 Advanced Options

### Custom Settings

```bash
# Custom model path and output directory
python3 scripts/run_phase1b_benchmark_standalone.py \
    --model_path /workspace/data/Cogumi-LLM/checkpoints/final \
    --output_dir /workspace/data/Cogumi-LLM/benchmark_results \
    --samples 50 \
    --api_key "sk-your-key"
```

### Quick Test (10 samples)

```bash
# Test with 10 samples per category (60 total, ~1 hour)
nohup python3 scripts/run_phase1b_benchmark_standalone.py \
    --samples 10 \
    > benchmark_test_log.txt 2>&1 &
```

### Full Benchmark (50 samples)

```bash
# Full benchmark (300 samples, ~6 hours)
nohup python3 scripts/run_phase1b_benchmark_standalone.py \
    --samples 50 \
    > benchmark_full_log.txt 2>&1 &
```

---

## 📊 Check Results

### View Log File

```bash
# See full log
cat benchmark_log.txt

# See last 50 lines
tail -50 benchmark_log.txt

# Search for errors
grep "ERROR" benchmark_log.txt
grep "Failed" benchmark_log.txt
```

### View Results

```bash
# List result files
ls -lh /workspace/data/Cogumi-LLM/checkpoints/benchmark_results/

# View latest report
cat /workspace/data/Cogumi-LLM/checkpoints/benchmark_results/benchmark_report_*.json | tail -1
```

---

## 🛑 Stop Benchmark

```bash
# Find process ID
ps aux | grep run_phase1b_benchmark

# Kill process (use PID from ps output)
kill <PID>

# Force kill if needed
kill -9 <PID>
```

---

## 💾 Download Results

### From Jupyter File Browser

Navigate to `/workspace/data/Cogumi-LLM/checkpoints/benchmark_results/`
- Right-click folder → Download as Archive

### Via SSH/SCP

```bash
# From your Mac terminal:
scp -r root@<vast-ip>:/workspace/data/Cogumi-LLM/checkpoints/benchmark_results ./
```

---

## ⏱️ Expected Timeline

**Full Benchmark (50 samples × 6 categories = 300 comparisons):**
- Duration: ~6 hours
- Cost: $10 OpenAI API + $1.20 GPU time
- Output: Comprehensive performance report

**Quick Test (10 samples × 6 categories = 60 comparisons):**
- Duration: ~1 hour  
- Cost: $2 OpenAI API + $0.20 GPU time
- Output: Rough performance estimate

---

## 🔍 Troubleshooting

### Benchmark Not Starting

```bash
# Check if script exists
ls -lh scripts/run_phase1b_benchmark_standalone.py

# Check if model exists
ls -lh /workspace/data/Cogumi-LLM/checkpoints/final/

# Test script directly (foreground)
python3 scripts/run_phase1b_benchmark_standalone.py --samples 1
```

### Out of Memory

```bash
# Check GPU memory
nvidia-smi

# Restart if needed (clears GPU memory)
# Then run benchmark again
```

### API Key Issues

```bash
# Verify API key is set
echo $OPENAI_API_KEY

# Test OpenAI API
python3 -c "from openai import OpenAI; client = OpenAI(); print(client.models.list())"
```

---

## ✅ Success Indicators

**Benchmark is running if you see:**
- Progress bars updating (e.g., "Testing math: 10%|█ | 5/50")
- New lines appearing in `tail -f benchmark_log.txt`
- Process shows up in `ps aux | grep run_phase1b`

**Benchmark completed if you see:**
- "✅ BENCHMARK COMPLETE!" in log
- Result files in `benchmark_results/` directory
- Process no longer in `ps aux` output

---

## 📁 Output Files

After completion, you'll find:

```
checkpoints/benchmark_results/
├── benchmark_report_20251026_123456.json  # Full results
├── failure_examples.json                   # Examples where GPT-4 won
├── category_math.json                      # Per-category details
├── category_code.json
├── category_reasoning.json
└── ...
```

---

## 🎯 Next Steps

1. **Stop the current notebook cell** on Vast.ai (if running)
2. **Upload standalone script** to `/workspace/data/Cogumi-LLM/scripts/`
3. **Run in terminal** with `nohup` command above
4. **Close browser** and let it run overnight
5. **Check results** in the morning!

No need to keep browser open! 🚀
