# Axolotl Migration Plan (Phase 1C+)

## Overview
Switch from traditional PyTorch training to Axolotl framework starting with Phase 1C (GPT-5 distillation) through all modifier phases (Phase 3-5).

## Why Axolotl After Phase 1A?

### Phase 1A (Current - Traditional PyTorch) ✅
- **Status:** 80% complete (epoch 2.4/3.0)
- **Rationale:** Already invested 26+ hours, full control with torch.compile optimization
- **Output:** 10GB merged base model (meta-llama/Meta-Llama-3.1-8B-Instruct + 640K LoRA)
- **Keep:** No reason to switch mid-training

### Phase 1C+ (Switch to Axolotl) ✅
- **Target:** QLoRA fine-tuning on smaller datasets (40K-80K examples)
- **Rationale:** 
  - Axolotl designed for efficient LoRA/QLoRA training
  - YAML configs easier to template and automate
  - Document's 93% automation assumes Axolotl
  - Proven pipeline for modifier generation

## Migration Steps

### Step 1: Install Axolotl (After Phase 1A Completes)
```bash
# Create Axolotl environment (separate from current venv)
conda create -n axolotl-env python=3.11
conda activate axolotl-env

# Install Axolotl with Flash Attention 2
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install packaging ninja
MAX_JOBS=4 pip install flash-attn --no-build-isolation

# Install Axolotl
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
pip install -e '.[flash-attn,deepspeed]'
```

**Time:** 30 minutes
**Cost:** $0

### Step 2: Create Base Axolotl Config Template (Phase 1C)
```yaml
# configs/axolotl/phase1c_distillation.yaml
base_model: models/phase1a_merged_10gb  # Our Phase 1A output
model_type: LlamaForCausalLM
tokenizer_type: LlamaTokenizer

load_in_8bit: false
load_in_4bit: true  # QLoRA 4-bit
strict: false

datasets:
  - path: data/phase1c/bidirectional_80k.jsonl
    type: alpaca  # instruction-response format
    
dataset_prepared_path: data/phase1c/prepared
val_set_size: 0.05
output_dir: models/phase1c_adapter

adapter: qlora
lora_r: 64
lora_alpha: 128
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj
  - gate_proj
  - down_proj
  - up_proj

sequence_len: 2048
sample_packing: true  # Efficient batch packing
pad_to_sequence_len: true

# Training hyperparameters
micro_batch_size: 4
gradient_accumulation_steps: 2
num_epochs: 3
learning_rate: 3e-6  # Lower for distillation
lr_scheduler: cosine
warmup_steps: 100

# Optimization
bf16: true
fp16: false
tf32: true
flash_attention: true
gradient_checkpointing: true

# Logging & Checkpoints
logging_steps: 10
save_steps: 1000
save_total_limit: 2
eval_steps: 500

# Special for distillation
special_tokens:
  eos_token: "<|eot_id|>"
  pad_token: "<|end_of_text|>"
```

**Time:** 15 minutes (template creation)
**Cost:** $0

### Step 3: Test Axolotl on Phase 1C Data (Validation Run)
```bash
# Prepare small test dataset (1K examples)
head -n 1000 data/phase1c/bidirectional_80k.jsonl > data/phase1c/test_1k.jsonl

# Update config for test run
# Change: num_epochs: 1, datasets.path: test_1k.jsonl

# Run Axolotl training test
cd axolotl
accelerate launch -m axolotl.cli.train \
  ../configs/axolotl/phase1c_distillation.yaml \
  --debug

# Should complete in ~30 minutes
# Validates: Data format, LoRA setup, Flash Attention, checkpointing
```

**Validation Criteria:**
- ✅ Training starts without errors
- ✅ GPU utilization >90%
- ✅ Loss decreases (should drop from ~2.0 to ~1.2 in 1 epoch)
- ✅ Checkpoints saved correctly
- ✅ Speed comparable to traditional training (~0.5-0.6s/step)

**Time:** 1 hour (30 min training + 30 min validation)
**Cost:** $2.50

### Step 4: Full Phase 1C Training with Axolotl
```bash
# Update config for full 80K dataset
# Change: datasets.path: bidirectional_80k.jsonl, num_epochs: 3

# Launch full training
accelerate launch -m axolotl.cli.train \
  configs/axolotl/phase1c_distillation.yaml

# Expected: 30K steps * 0.5s/step = 4.2 hours
# Cost: 5 hours * $2.50/hr = $12.50
```

**Output:**
- `models/phase1c_adapter/adapter_model.safetensors` (~260MB QLoRA adapter)
- `models/phase1c_adapter/adapter_config.json`

**Merge adapter to base:**
```bash
python axolotl/scripts/merge_lora.py \
  --base_model models/phase1a_merged_10gb \
  --lora_model models/phase1c_adapter \
  --output_dir models/phase1c_enhanced_10gb
```

**Time:** 5 hours training + 30 min merging
**Cost:** $12.50 (confirmed from earlier estimate)

### Step 5: Template Configs for Modifiers (Phase 3-5)

Axolotl makes it trivial to reuse configs - just change:
- `base_model`: Use compressed 500MB base after Phase 2
- `datasets.path`: Domain-specific data (code, reasoning, automation)
- `lora_r`: Adjust rank per domain (96-128)
- `output_dir`: Per-modifier output path

**Example for Code Modifier:**
```yaml
# configs/axolotl/code_modifier.yaml
base_model: models/phase2_compressed_500mb
datasets:
  - path: data/phase3/code_modifier_12.5k.jsonl
lora_r: 128  # Higher rank for code complexity
# ... rest same as phase1c template
```

**Automation with Claude 4.5:**
Prompt: "Generate Axolotl config for {domain} modifier using base model {path}, dataset {path}, LoRA rank {rank}"

**Time per modifier:** 5 minutes config generation + 4 hours training
**Cost per modifier:** $0 generation + $10-12 training

---

## Comparison: Traditional vs Axolotl

| Aspect | Traditional PyTorch (Phase 1A) | Axolotl (Phase 1C+) |
|--------|-------------------------------|---------------------|
| **Setup Time** | 2 hours (custom script) | 30 min (YAML config) |
| **Code Complexity** | 500+ lines Python | 50 lines YAML |
| **Optimization** | torch.compile manual | Flash Attn + DeepSpeed built-in |
| **Automation** | Difficult (code generation) | Easy (config templating) |
| **Debugging** | Custom logging | Rich Axolotl logging |
| **Reusability** | Low (script per phase) | High (template configs) |
| **Speed** | 0.49s/step (excellent) | 0.45-0.55s/step (comparable) |
| **Best For** | Full training (large datasets) | LoRA/QLoRA fine-tuning |

---

## Risk Mitigation

### Risk 1: Axolotl Compatibility Issues
**Mitigation:**
- Test on 1K examples first (Step 3)
- Validate speed matches traditional training
- Keep traditional scripts as backup

### Risk 2: Data Format Conversion
**Mitigation:**
- Axolotl supports standard formats (alpaca, sharegpt, chat_template)
- Our instruction-response format maps directly to alpaca
- Test with pretokenize_dataset.py output

### Risk 3: Slower Training
**Mitigation:**
- Axolotl's Flash Attention 2 + sample packing often FASTER
- If slower, adjust micro_batch_size and gradient_accumulation_steps
- Worst case: Revert to traditional for that phase

### Risk 4: Automation Script Generation
**Mitigation:**
- YAML configs much easier for Claude 4.5 to generate than Python
- Templates provided above for all phases
- Human validation before each training run

---

## Success Criteria

### Phase 1C (Axolotl Validation)
- ✅ Training completes in 4-5 hours (comparable to traditional estimate)
- ✅ Loss converges to 0.8-1.0 (same as Phase 1A)
- ✅ Merged model achieves 88-100% GPT-4 (Phase 1C target)
- ✅ Adapter size ~260MB (QLoRA 4-bit)
- ✅ Cost: $12.50 (confirmed budget)

### Phase 3-5 (Modifier Generation)
- ✅ Config generation <5 minutes per modifier (Claude 4.5)
- ✅ Training time 4-6 hours per modifier (acceptable)
- ✅ Compressed modifiers: 40-50MB each (after pruning)
- ✅ Quality: Code >115% GPT-4, Reasoning >100%, Automation >105%

---

## Timeline Impact

| Phase | Original (Traditional) | With Axolotl | Savings |
|-------|----------------------|--------------|---------|
| Phase 1C Setup | 2 hours (script) | 30 min (config) | 1.5 hours |
| Phase 1C Training | 5 hours | 4-5 hours | 0 hours |
| Modifier Setup (x3) | 6 hours (scripts) | 15 min (configs) | 5.75 hours |
| Modifier Training (x3) | 12-15 hours | 12-15 hours | 0 hours |
| **Total Saved** | - | - | **7.25 hours** |

---

## Recommendation: ✅ APPROVED

**Switch to Axolotl starting Phase 1C** for:
1. Simplified config management
2. Better automation with Claude 4.5
3. Proven pipeline for LoRA/QLoRA
4. Time savings (7+ hours over Phases 1C-5)
5. Matches document's 93% automation strategy

**Action Items:**
1. ⏳ Wait for Phase 1A to complete (~7 hours)
2. ✅ Install Axolotl in separate conda environment (30 min)
3. ✅ Test on 1K Phase 1C examples (1 hour, $2.50)
4. ✅ Run full Phase 1C training if test passes (5 hours, $12.50)
5. ✅ Template configs for Phase 3-5 modifiers (15 min)
6. ✅ Template configs for Phase 7 meta-learning (10 min)

**No changes needed to Phase 1A - let it complete as-is!**

---

## Phase 7: Meta-Learning with Axolotl (MVP - Week 15-16)

### 7.1 Meta-Learning (MAML) Training

**Why Axolotl for Meta-Learning:**
- MAML requires gradient-based few-shot adaptation
- Axolotl's LoRA infrastructure perfect for meta-learning adapters
- YAML configs enable easy task distribution specification
- Flash Attention 2 + gradient checkpointing handle meta-batch complexity

**Axolotl Config for MAML:**

```yaml
# configs/axolotl/phase7_meta_learning.yaml
base_model: models/phase2_compressed_520mb  # After compression
model_type: LlamaForCausalLM
tokenizer_type: LlamaTokenizer

load_in_8bit: false
load_in_4bit: true  # QLoRA for meta-adapter
strict: false

# Meta-learning dataset (task distributions)
datasets:
  - path: data/phase7/meta_tasks.jsonl  # Task-structured data
    type: completion  # Custom format for meta-tasks
    
dataset_prepared_path: data/phase7/prepared
val_set_size: 0.10  # 10% validation tasks
output_dir: models/phase7_meta_adapter

# LoRA for meta-learning adapter
adapter: qlora
lora_r: 48  # Moderate rank for adaptation
lora_alpha: 96
lora_dropout: 0.05
lora_target_modules:
  - q_proj  # Query projection (critical for adaptation)
  - v_proj  # Value projection (critical for adaptation)
  - k_proj
  - o_proj

sequence_len: 2048
sample_packing: true
pad_to_sequence_len: true

# MAML-specific hyperparameters
micro_batch_size: 4  # 4 tasks per batch
gradient_accumulation_steps: 4  # Effective 16 tasks per meta-update
num_epochs: 2  # 2 passes over meta-task distribution
learning_rate: 5e-6  # Outer loop learning rate (meta-updates)
lr_scheduler: cosine
warmup_steps: 500

# Meta-learning specific
# (Note: MAML inner loop requires custom training script)
# Axolotl handles outer loop, custom script handles inner adaptation

# Optimization
bf16: true
fp16: false
tf32: true
flash_attention: true
gradient_checkpointing: true

# Logging & Checkpoints
logging_steps: 50
save_steps: 1000
save_total_limit: 3
eval_steps: 500

special_tokens:
  eos_token: "<|eot_id|>"
  pad_token: "<|end_of_text|>"
```

**Meta-Task Dataset Format:**

```jsonl
{
  "task_id": "code_style_adaptation_001",
  "support_set": [
    {"instruction": "Write a function to add two numbers", "output": "def add(a, b):\n    return a + b"},
    {"instruction": "Write a function to subtract", "output": "def subtract(a, b):\n    return a - b"}
  ],
  "query_set": [
    {"instruction": "Write a function to multiply", "output": "def multiply(a, b):\n    return a * b"}
  ]
}
```

**Custom MAML Training Script (wraps Axolotl):**

```python
# scripts/train_maml_with_axolotl.py
"""
MAML training using Axolotl for outer loop optimization.
Custom inner loop for few-shot adaptation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
import json
from tqdm import tqdm

def inner_loop_adaptation(model, support_set, inner_lr=1e-4, inner_steps=3):
    """
    Fast adaptation on support set (MAML inner loop).
    Returns adapted model state.
    """
    # Clone model parameters
    adapted_params = {name: param.clone() for name, param in model.named_parameters() if param.requires_grad}
    
    # K gradient steps on support set
    for step in range(inner_steps):
        loss = compute_loss(model, support_set)
        grads = torch.autograd.grad(loss, adapted_params.values(), create_graph=True)
        
        # Update adapted parameters
        adapted_params = {
            name: param - inner_lr * grad 
            for (name, param), grad in zip(adapted_params.items(), grads)
        }
        
        # Load adapted parameters into model
        load_params(model, adapted_params)
    
    return model

def meta_training_step(model, tasks, inner_lr=1e-4, inner_steps=3):
    """
    One meta-training step (MAML outer loop).
    """
    meta_loss = 0.0
    
    for task in tasks:
        # Inner loop: adapt on support set
        adapted_model = inner_loop_adaptation(
            model.clone(), 
            task['support_set'], 
            inner_lr, 
            inner_steps
        )
        
        # Outer loop: evaluate on query set
        query_loss = compute_loss(adapted_model, task['query_set'])
        meta_loss += query_loss
    
    # Average across tasks
    meta_loss /= len(tasks)
    
    return meta_loss

# Load base model and attach LoRA adapter
model = AutoModelForCausalLM.from_pretrained(
    "models/phase2_compressed_520mb",
    load_in_4bit=True
)

lora_config = LoraConfig(
    r=48,
    lora_alpha=96,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)

model = get_peft_model(model, lora_config)

# Meta-training loop
for epoch in range(2):
    for batch_idx, task_batch in enumerate(meta_task_loader):
        # MAML meta-update
        meta_loss = meta_training_step(
            model, 
            task_batch, 
            inner_lr=1e-4, 
            inner_steps=3
        )
        
        # Backprop through meta-loss
        meta_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Meta-Loss: {meta_loss:.4f}")
```

**Training Execution:**

```bash
# Generate meta-task dataset
python scripts/generate_meta_tasks.py \
  --output data/phase7/meta_tasks.jsonl \
  --num_tasks 10000 \
  --tasks_per_domain 1250

# Train MAML meta-learning adapter
python scripts/train_maml_with_axolotl.py \
  --config configs/axolotl/phase7_meta_learning.yaml \
  --inner_lr 1e-4 \
  --inner_steps 3 \
  --meta_batch_size 16

# Expected: 15,000 meta-iterations * 0.8s/iter = 3.3 hours per epoch
# 2 epochs = 6.6 hours
# Cost: 7 hours * $2.50/hr = $17.50

# Merge meta-adapter to base
python axolotl/scripts/merge_lora.py \
  --base_model models/phase2_compressed_520mb \
  --lora_model models/phase7_meta_adapter \
  --output_dir models/phase7_meta_enhanced_520mb

# Compress meta-adapter
python scripts/compress_modifier.py \
  --input models/phase7_meta_adapter \
  --output models/phase7_meta_adapter_12mb \
  --target_size 12MB
```

**Validation:**

```bash
# Test few-shot adaptation on held-out tasks
python scripts/validate_meta_learning.py \
  --model models/phase7_meta_enhanced_520mb \
  --test_tasks data/phase7/test_tasks.jsonl \
  --shots 1 3 5

# Expected results:
# 1-shot: +10-12% over base in-context learning
# 3-shot: +12-15% over base in-context learning
# 5-shot: +13-18% over base in-context learning
```

**Time:** 1 week (7 days)
- Meta-task generation: 1 day
- MAML training: 7 hours
- Validation: 2 days
- Compression + integration: 1 day

**Cost:** $85
- Meta-task generation (GPT-4-mini): $20
- MAML training (H100 SXM): $17.50
- Validation (H100 SXM): $12.50
- Few-shot template development: $0

**Output:** 12MB compressed meta-learning adapter

---

## Post-MVP Enhancement Phases

### Phase 10: Multi-Mode Architecture (Week 18)

**Implementation:** Code-only, no Axolotl training needed

```python
# src/phase4_router/multi_mode.py
"""
Multi-mode architecture: Fast vs Accurate mode selection.
"""

class MultiModeRouter:
    def __init__(self, base_model, modifiers, router, threshold=0.80):
        self.base = base_model
        self.modifiers = modifiers
        self.router = router
        self.threshold = threshold
        self.current_mode = "fast"
        self.loaded_modifier = None
    
    def route(self, query, mode="auto"):
        """
        Route query to Fast or Accurate mode.
        """
        if mode == "fast":
            return self.fast_mode(query)
        elif mode == "accurate":
            return self.accurate_mode(query)
        else:  # auto
            confidence = self.router.predict_confidence(query)
            if confidence >= self.threshold:
                return self.fast_mode(query)
            else:
                return self.accurate_mode(query)
    
    def fast_mode(self, query):
        """Base model only (520MB + 13MB router + 12MB meta)."""
        return self.base.generate(query)
    
    def accurate_mode(self, query):
        """Base + domain modifier + enhancements."""
        domain = self.router.predict_domain(query)
        
        # Load modifier if not already loaded
        if self.loaded_modifier != domain:
            self.load_modifier(domain)
        
        # Generate with modifier
        response = self.base.generate_with_modifier(
            query, 
            self.modifiers[domain]
        )
        
        return response
```

**Time:** 1 week (implementation + testing)
**Cost:** $0
**Output:** Runtime mode selection

---

### Phase 11: Self-Consistency Voting (Week 19)

**Implementation:** Inference-time logic, no training needed (uses existing modifiers)

```python
# src/phase4_router/self_consistency.py
"""
Self-consistency multi-path voting for Accurate mode.
"""

def self_consistency_generate(model, query, num_paths=5, temperature=0.8):
    """
    Generate multiple reasoning paths and vote on answer.
    Only activated in Accurate mode for hard queries.
    """
    paths = []
    
    # Generate N diverse paths
    for i in range(num_paths):
        response = model.generate(
            query,
            temperature=temperature,
            top_p=0.9,
            frequency_penalty=1.2
        )
        paths.append(response)
    
    # Extract answers from each path
    answers = [extract_answer(path) for path in paths]
    
    # Vote on most common answer
    voted_answer = majority_vote(answers)
    
    # Compute confidence (agreement ratio)
    confidence = answers.count(voted_answer) / len(answers)
    
    return {
        "answer": voted_answer,
        "confidence": confidence,
        "paths": paths  # Return all paths for transparency
    }
```

**Time:** 1 week (implementation + validation)
**Cost:** $55 (training dataset generation for validation)
**Output:** Runtime voting logic

---

### Phase 12: Self-Critique Classifier (Week 20)

**Axolotl Config for Critique Classifier:**

```yaml
# configs/axolotl/phase12_critique_classifier.yaml
base_model: bert-base-uncased  # Small BERT for critique
model_type: BertForSequenceClassification
tokenizer_type: BertTokenizer

num_labels: 11  # 0-10 quality score

datasets:
  - path: data/phase12/critique_dataset.jsonl  # 8K (query, response, score) examples
    type: classification
    
dataset_prepared_path: data/phase12/prepared
val_set_size: 0.15
output_dir: models/phase12_critique_classifier

# Fine-tuning hyperparameters
micro_batch_size: 16
gradient_accumulation_steps: 2
num_epochs: 3
learning_rate: 2e-5
lr_scheduler: linear
warmup_steps: 200

# Optimization
bf16: true
fp16: false

# Logging & Checkpoints
logging_steps: 20
save_steps: 500
save_total_limit: 2
eval_steps: 250
```

**Training:**

```bash
# Generate critique dataset
python scripts/generate_critique_dataset.py \
  --model models/phase7_meta_enhanced_520mb \
  --num_examples 8000 \
  --scorer gpt-4-mini

# Train critique classifier
accelerate launch -m axolotl.cli.train \
  configs/axolotl/phase12_critique_classifier.yaml

# Distill to smaller model (110MB → 10MB)
python scripts/distill_critique_classifier.py \
  --teacher models/phase12_critique_classifier \
  --student lstm \
  --output models/phase12_critique_10mb
```

**Time:** 10 days
- Dataset generation: 3 days, $40
- BERT training: 2 days, $5
- Distillation: 2 days
- Integration: 3 days

**Cost:** $45
**Output:** 10MB critique classifier

---

### Phase 13: Adaptive Threshold Learning (Week 21)

**Implementation:** Requires real user data, trains lightweight adapter

```python
# scripts/train_adaptive_router.py
"""
Train adaptive learning layer on user interaction data.
"""

from sklearn.linear_model import LogisticRegression
import numpy as np

# Collect 10K+ user interactions
interactions = load_user_interactions(min_count=10000)

# Extract features
X = []
Y = []
for interaction in interactions:
    features = extract_features(
        interaction['query'],
        interaction['confidence'],
        interaction['domain']
    )
    satisfaction = interaction['user_satisfied']  # 0 or 1
    
    X.append(features)
    Y.append(satisfaction)

# Train adaptive layer
adaptive_layer = LogisticRegression()
adaptive_layer.fit(np.array(X), np.array(Y))

# Save adaptive layer (2MB)
save_model(adaptive_layer, "models/phase13_adaptive_2mb")
```

**Time:** 1 week (after 10K+ interactions collected)
**Cost:** $30
**Output:** 2MB adaptive learning layer

---

## Summary: Axolotl Usage Across All Phases

| Phase | Component | Axolotl? | Why / Why Not |
|-------|-----------|----------|---------------|
| **1A** | Base Training | ❌ NO | Traditional PyTorch, already 80% complete |
| **1C** | GPT-5 Distillation | ✅ YES | QLoRA sweet spot, YAML configs |
| **2** | Compression | ❌ NO | Neural Magic, AWQ, llama.cpp (not training) |
| **3-5** | MVP Modifiers | ✅ YES | QLoRA adapters, perfect for Axolotl |
| **6** | Router | ❌ NO | Lightweight classifier, scikit-learn |
| **7** | Meta-Learning | ✅ YES (hybrid) | MAML outer loop via Axolotl, custom inner loop |
| **8** | Deployment | ❌ NO | HuggingFace upload, Gradio (no training) |
| **10** | Multi-Mode | ❌ NO | Code implementation (no training) |
| **11** | Self-Consistency | ❌ NO | Inference-time voting (no training) |
| **12** | Self-Critique | ✅ YES | BERT fine-tuning via Axolotl |
| **13** | Adaptive Learning | ❌ NO | Logistic regression, lightweight |
| **14** | 5 More Modifiers | ✅ YES | Same as Phases 3-5, QLoRA |

**Axolotl Verdict:** ✅ **Excellent choice for all LLM fine-tuning phases (1C, 3-5, 7, 12, 14)**
- Saves 7+ hours manual scripting
- YAML configs = easy templating
- Flash Attention 2, DeepSpeed, sample packing built-in
- Industry-proven for LoRA/QLoRA
- Handles meta-learning outer loop elegantly
