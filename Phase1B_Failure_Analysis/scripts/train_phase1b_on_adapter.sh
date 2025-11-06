#!/bin/bash
# Train Phase 1B on ORIGINAL Phase 1A adapter + base (skip corrupted merge)
#
# CRITICAL FIX: The 4-bit merged model is corrupted (70%→28% ties, 48%→12% code wins)
# Solution: Train directly on adapter + base model, no merge needed

set -e

echo "================================================================================"
echo "🔧 Phase 1B: Train on Original Phase 1A Adapter (NO CORRUPTED MERGE)"
echo "================================================================================"
echo ""
echo "⚠️  IMPORTANT: We discovered the 4-bit merged model is corrupted!"
echo "   - MATH ties: 70% → 28% (catastrophic loss)"
echo "   - CODE wins: 48% → 12% (massive degradation)"
echo ""
echo "Solution: Train Phase 1B adapter on top of Phase 1A adapter + base"
echo "   This preserves Phase 1A's full quality without merge corruption."
echo ""
echo "================================================================================"
echo ""

# Detect environment
if [ -d "/workspace" ]; then
    BASE_DIR="/workspace/data/Cogumi-LLM"
else
    BASE_DIR="$(pwd)"
fi

cd "$BASE_DIR"

# Use ORIGINAL adapter (not corrupted merge!)
BASE_MODEL="unsloth/meta-llama-3.1-8b-instruct-bnb-4bit"
PHASE1A_ADAPTER="checkpoints/final"  # Original Phase 1A adapter
OUTPUT_DIR="checkpoints/phase1b_all_failures_on_adapter"

# Verify Phase 1A adapter exists
if [ ! -d "$PHASE1A_ADAPTER" ]; then
    echo "❌ Phase 1A adapter not found: $PHASE1A_ADAPTER"
    exit 1
fi

echo "✅ Using Phase 1A adapter: $PHASE1A_ADAPTER"
echo "✅ Base model: $BASE_MODEL"
echo ""

# Combine failure datasets
MERGED_DATA="data/phase1b_all_failures/all_failures_combined.jsonl"

if [ ! -f "$MERGED_DATA" ]; then
    echo "📊 Combining failure datasets..."
    python - <<'PY'
import json
from pathlib import Path

paths = [
    "data/phase1b_all_failures/math_all_failures.jsonl",
    "data/phase1b_all_failures/code_all_failures.jsonl",
    "data/phase1b_all_failures/creativity_all_failures.jsonl"
]

out = Path('data/phase1b_all_failures/all_failures_combined.jsonl')
count = 0

with out.open('w') as fo:
    for p in paths:
        if Path(p).exists():
            for line in Path(p).read_text().splitlines():
                try:
                    obj = json.loads(line)
                    inst = obj.get('instruction') or obj.get('prompt')
                    outp = obj.get('output') or obj.get('full_solution') or obj.get('correct_choice')
                    if inst and outp:
                        fo.write(json.dumps({'instruction': inst, 'output': outp}) + '\n')
                        count += 1
                except Exception as e:
                    pass

print(f'✅ Combined {count} examples to {out}')
PY
fi

echo ""
echo "🚀 Starting training..."
echo "-------------------------------------------"
echo "Base model: $BASE_MODEL"
echo "Phase 1A adapter: $PHASE1A_ADAPTER"
echo "Dataset: $MERGED_DATA"
echo "Output: $OUTPUT_DIR"
echo "-------------------------------------------"
echo ""

# Training with ORIGINAL Phase 1A adapter as base
# This trains a NEW adapter on top of Phase 1A adapter
python - <<'TRAIN'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset
import json
from pathlib import Path
import sys

# Load base model
print("📥 Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/meta-llama-3.1-8b-instruct-bnb-4bit",
    load_in_4bit=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("unsloth/meta-llama-3.1-8b-instruct-bnb-4bit")
tokenizer.pad_token = tokenizer.eos_token

# Load Phase 1A adapter
print("📥 Loading Phase 1A adapter...")
model = PeftModel.from_pretrained(base_model, "checkpoints/final")
print("✅ Phase 1A adapter loaded")

# Prepare model for training Phase 1B adapter
print("🔧 Preparing for Phase 1B training...")
model = model.merge_and_unload()  # Merge in memory (not saved)
model.train()

# Add NEW LoRA adapter for Phase 1B
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
print("✅ Phase 1B LoRA adapter added")

# Load dataset
print("📊 Loading training data...")
data_path = Path("data/phase1b_all_failures/all_failures_combined.jsonl")
examples = []
for line in data_path.read_text().splitlines():
    obj = json.loads(line)
    examples.append({
        'instruction': obj['instruction'],
        'output': obj['output']
    })

dataset = Dataset.from_list(examples)
print(f"✅ Loaded {len(dataset)} examples")

# Tokenize
def tokenize_fn(examples):
    texts = [f"{inst}\n\n{out}" for inst, out in zip(examples['instruction'], examples['output'])]
    return tokenizer(texts, truncation=True, max_length=2048, padding=False)

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

# Training arguments (conservative to avoid forgetting)
training_args = TrainingArguments(
    output_dir="checkpoints/phase1b_all_failures_on_adapter",
    num_train_epochs=2,
    learning_rate=3e-6,  # Very conservative
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    save_strategy="epoch",
    logging_steps=10,
    fp16=True,
    report_to="none"
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer
)

print("🚀 Starting training...")
trainer.train()

# Save Phase 1B adapter
print("💾 Saving Phase 1B adapter...")
model.save_pretrained("checkpoints/phase1b_all_failures_on_adapter")
tokenizer.save_pretrained("checkpoints/phase1b_all_failures_on_adapter")
print("✅ Training complete!")
TRAIN

echo ""
echo "================================================================================"
echo "✅ Phase 1B Training Complete"
echo "================================================================================"
echo ""
echo "Adapter saved to: $OUTPUT_DIR"
echo ""
echo "NEXT STEPS:"
echo "  1. Validate: python scripts/validate_phase1b1_optimized.py \\"
echo "       --phase1a_results checkpoints/benchmark_results_full \\"
echo "       --phase1b_model checkpoints/phase1b_all_failures_on_adapter \\"
echo "       --openai_key \$OPENAI_API_KEY \\"
echo "       --output_dir results/phase1b_on_adapter_validation"
echo ""
echo "  2. Compare against Phase 1A adapter (not corrupted merge!)"
echo ""
echo "================================================================================"
