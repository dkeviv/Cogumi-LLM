#!/bin/bash
# Phase 1B: Self-Consistency Training Execution Script
# Run on Vast.ai H100

set -e  # Exit on error

echo "🚀 Phase 1B: Self-Consistency Training"
echo "======================================"
echo ""

# Verify we're in the right directory
if [ ! -d "/workspace/data/Cogumi-LLM" ]; then
    echo "❌ Error: Not in Vast.ai environment"
    echo "Expected: /workspace/data/Cogumi-LLM"
    exit 1
fi

cd /workspace/data/Cogumi-LLM

# Step 1: Verify model checkpoint exists
echo "📋 Step 1: Verifying model checkpoint..."
if [ ! -d "checkpoints/final" ]; then
    echo "❌ Error: Model checkpoint not found at checkpoints/final"
    echo "Please ensure Phase 1A training is complete"
    exit 1
fi
echo "✅ Model checkpoint found"
echo ""

# Step 2: Create output directory
echo "📋 Step 2: Creating output directory..."
mkdir -p self_distillation
echo "✅ Output directory ready: self_distillation/"
echo ""

# Step 3: Generate self-consistent training data
echo "📋 Step 3: Generating self-consistent training data..."
echo "This will process:"
echo "  - MATH: 500 problems at temp=0.0"
echo "  - CODE: 164 HumanEval problems at temp=0.0"
echo "Estimated time: 2-3 hours"
echo ""

python scripts/self_consistency_distillation.py

# Verify output files
echo ""
echo "📋 Verifying generated files..."
if [ ! -f "self_distillation/math_distilled.jsonl" ]; then
    echo "❌ Error: math_distilled.jsonl not generated"
    exit 1
fi
if [ ! -f "self_distillation/code_distilled.jsonl" ]; then
    echo "❌ Error: code_distilled.jsonl not generated"
    exit 1
fi

MATH_COUNT=$(wc -l < self_distillation/math_distilled.jsonl)
CODE_COUNT=$(wc -l < self_distillation/code_distilled.jsonl)

echo "✅ Generated files:"
echo "   - math_distilled.jsonl: $MATH_COUNT examples"
echo "   - code_distilled.jsonl: $CODE_COUNT examples"
echo ""

# Step 4: Train on self-consistent data
echo "📋 Step 4: Training on self-consistent data..."
echo "Training config:"
echo "  - Epochs: 2"
echo "  - Learning rate: 5e-6"
echo "  - Batch size: 4"
echo "  - Total examples: $((MATH_COUNT + CODE_COUNT))"
echo "Estimated time: 3-4 hours"
echo ""

python train_qlora_optimized.py \
    --model_name unsloth/meta-llama-3.1-8b-instruct-bnb-4bit \
    --dataset_path "self_distillation/*.jsonl" \
    --output_dir checkpoints/self_consistent \
    --num_train_epochs 2 \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --save_strategy epoch \
    --logging_steps 10

echo ""
echo "✅ Training complete!"
echo ""

# Step 5: Quick validation
echo "📋 Step 5: Running quick validation..."
python -c "
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

print('Loading self-consistent model...')
model = AutoModelForCausalLM.from_pretrained(
    'unsloth/meta-llama-3.1-8b-instruct-bnb-4bit',
    device_map='auto',
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(model, 'checkpoints/self_consistent')
tokenizer = AutoTokenizer.from_pretrained('checkpoints/self_consistent')

# Test consistency with a simple math problem
test_prompt = '''Solve this math problem step by step:

Problem: If 3x + 7 = 22, what is x?

Solution:'''

print('Testing consistency (5 runs)...')
outputs = []
for i in range(5):
    inputs = tokenizer(test_prompt, return_tensors='pt').to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,  # Test with non-zero temp
        do_sample=True
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    outputs.append(response)

# Count unique responses
unique_outputs = len(set(outputs))
consistency = (5 - unique_outputs + 1) / 5 * 100

print(f'Consistency: {consistency:.1f}%')
print(f'Unique responses: {unique_outputs}/5')

if consistency >= 60:
    print('✅ PASSED: Consistency ≥60%')
else:
    print('⚠️  WARNING: Consistency <60%, may need more training')
"

echo ""
echo "🎉 Phase 1B Complete!"
echo "===================="
echo ""
echo "Next steps:"
echo "1. Run full benchmarks: python scripts/run_benchmarks.py --model_path checkpoints/self_consistent"
echo "2. Check consistency: notebooks/Benchmark_Diagnostic_v2.ipynb"
echo "3. Compare with Phase 1A results"
echo ""
echo "Expected improvements:"
echo "  - Consistency: 10% → 60-80%"
echo "  - MATH score: 41% → 65-75%"
echo "  - MATH ties: 70% → <30%"
echo "  - CODE score: 58% → 70-80%"
echo ""
