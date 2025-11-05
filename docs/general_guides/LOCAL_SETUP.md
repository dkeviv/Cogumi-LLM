# Local Development Setup (Mac)

## ✅ Environment Created: October 27, 2025

### Python Version
- **Python 3.14.0** (latest stable)
- **Environment:** `cogumi` (660MB - no PyTorch, training happens on Vast.ai)

### What's Installed Locally

**API Clients** (for dataset creation scripts):
- groq 0.33.0 (Llama-405B access)
- openai 2.6.1 (GPT-4o, GPT-4-mini, GPT-5)
- together 1.5.29 (Qwen3-Coder-480B)
- anthropic 0.71.0 (Claude 3.5)

**Development Tools**:
- jupyter 1.1.1 + jupyterlab 4.4.10
- matplotlib 3.10.7 + seaborn 0.13.2
- black 25.9.0 + ruff 0.14.2

**Data Processing**:
- datasets 4.3.0
- pandas 2.3.3
- numpy 2.3.4

### What's NOT Installed (On Purpose)
- ❌ PyTorch (74GB - not needed locally)
- ❌ Transformers (12GB - not needed locally)
- ❌ CUDA/GPU libraries (Mac doesn't support them)
- ❌ Training dependencies (training happens on Vast.ai H100)

### Why This Setup?

**Mac is for:**
- ✅ Code editing (VS Code with Python 3.14)
- ✅ Running data collection scripts (API calls)
- ✅ Viewing/editing notebooks locally
- ✅ Git operations, documentation

**Vast.ai H100 is for:**
- ⚡ Training (Phase 1A: QLoRA on 600K dataset)
- ⚡ Compression (Phase 2: Neural Magic pruning)
- ⚡ Running benchmarks (Phase 1B diagnostics)
- ⚡ Everything GPU-related

### Activation

```bash
cd /Users/vivekdurairaj/Projects/Cogumi-LLM
source cogumi/bin/activate
```

### VS Code Setup

1. **Select Python Interpreter:**
   - Press `Cmd+Shift+P`
   - Type: "Python: Select Interpreter"
   - Choose: `./cogumi/bin/python` (Python 3.14.0)

2. **Notebook Kernel:**
   - Open any `.ipynb` file
   - Click kernel selector (top right)
   - Choose: **"Cogumi (Python 3.14)"**

### Requirements Files

- `requirements-local.txt` - Local development (THIS Mac)
- `requirements.txt` - Full training (Vast.ai H100)
- `requirements-h100-training.txt` - H100-specific (Unsloth + optimizations)

### Package Updates

```bash
# Update local packages
./cogumi/bin/pip install --upgrade -r requirements-local.txt

# Check outdated
./cogumi/bin/pip list --outdated
```

### Size Comparison

| Environment | Size | Purpose |
|------------|------|---------|
| **Local (Mac)** | 660MB | Code editing, API calls |
| **Vast.ai Full** | ~40GB | Training, compression, inference |
| **Savings** | 39.3GB | You don't need GPU libs locally! |

---

**Updated:** October 27, 2025
**Python:** 3.14.0 (upgraded from 3.9.6)
**Rationale:** No point installing PyTorch locally - all training happens on Vast.ai
