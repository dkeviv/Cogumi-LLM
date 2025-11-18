#!/usr/bin/env python3
"""Quick diagnostic to check generation speed."""

import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
print("Loading model...")
model_path = "models/phase1_maml_lora_v2/merged"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()
print("Model loaded!\n")

# Load a few examples
print("Loading test examples...")
with open("data/phase1/answers/training_data_clean.jsonl", 'r') as f:
    examples = [json.loads(line) for line in f][:10]  # Just 10 examples

print(f"Testing with {len(examples)} examples\n")

# Test different batch sizes and max_new_tokens
test_configs = [
    (1, 512, "Single, 512 tokens"),
    (10, 512, "Batch 10, 512 tokens"),
    (10, 256, "Batch 10, 256 tokens"),
    (10, 128, "Batch 10, 128 tokens"),
]

for batch_size, max_new_tokens, desc in test_configs:
    prompts = [ex['input'] for ex in examples[:batch_size]]
    
    # Tokenize
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    print(f"Testing: {desc}")
    print(f"  Input shape: {inputs['input_ids'].shape}")
    
    # Generate
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    elapsed = time.time() - start
    
    # Check output lengths
    actual_lengths = []
    for i, output in enumerate(outputs):
        input_len = inputs['input_ids'][i].ne(tokenizer.pad_token_id).sum()
        gen_len = len(output) - input_len
        actual_lengths.append(gen_len.item())
    
    avg_length = sum(actual_lengths) / len(actual_lengths)
    
    print(f"  Time: {elapsed:.2f}s ({elapsed/batch_size:.2f}s per example)")
    print(f"  Avg generated tokens: {avg_length:.0f}")
    print(f"  Throughput: {batch_size/elapsed:.1f} examples/sec")
    print()

print("Diagnosis complete!")
