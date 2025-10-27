#!/usr/bin/env python3
"""
Test: Full Precision Base vs 4-bit Quantized Base
Compare math performance to see if quantization is causing errors
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import time

# Test problems that failed in benchmark
test_problems = [
    {
        "name": "House Flipping",
        "prompt": "Solve this math problem step by step:\n\nJosh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
        "expected_answer": "$70,000"
    },
    {
        "name": "James Sprints",
        "prompt": "Solve this math problem step by step:\n\nJames decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
        "expected_answer": "540"
    },
    {
        "name": "Simple Addition",
        "prompt": "Solve this math problem step by step:\n\nJanet has 3 apples. She buys 7 more apples. How many apples does she have?",
        "expected_answer": "10"
    }
]

def load_model_quantized(adapter_path):
    """Load with 4-bit quantized base (current approach)"""
    print("\n" + "="*80)
    print("LOADING MODEL: 4-bit Quantized Base")
    print("="*80)
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    
    peft_config = PeftConfig.from_pretrained(adapter_path)
    
    # Load 4-bit quantized base
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,  # unsloth 4-bit version
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True
    )
    
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    model.eval()
    
    print(f"✓ Model loaded (4-bit base)")
    print(f"  Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    return tokenizer, model

def load_model_fullprecision(adapter_path):
    """Load with full precision base (better approach)"""
    print("\n" + "="*80)
    print("LOADING MODEL: Full Precision Base")
    print("="*80)
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    
    # Load FULL PRECISION base model (not the 4-bit unsloth version)
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",  # Full precision official version
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    model.eval()
    
    print(f"✓ Model loaded (full precision base)")
    print(f"  Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    return tokenizer, model

def generate_response(tokenizer, model, prompt, max_tokens=300):
    """Generate response with proper Llama-3.1 formatting"""
    formatted_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response_with_tokens = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    if "<|start_header_id|>assistant<|end_header_id|>" in response_with_tokens:
        assistant_part = response_with_tokens.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        if "<|eot_id|>" in assistant_part:
            assistant_part = assistant_part.split("<|eot_id|>")[0]
        response = assistant_part.strip()
    else:
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
    
    return response

def test_model(tokenizer, model, model_name):
    """Test model on problem set"""
    print(f"\n" + "="*80)
    print(f"TESTING: {model_name}")
    print("="*80)
    
    for i, problem in enumerate(test_problems, 1):
        print(f"\n" + "-"*80)
        print(f"Problem {i}: {problem['name']}")
        print("-"*80)
        print(f"Prompt: {problem['prompt'][:100]}...")
        
        response = generate_response(tokenizer, model, problem['prompt'])
        
        print(f"\nResponse:")
        print(response)
        
        expected = problem['expected_answer']
        found = expected in response
        
        print(f"\nExpected answer: {expected}")
        print(f"Found in response: {'✓ YES' if found else '✗ NO'}")

def main():
    adapter_path = "/workspace/data/Cogumi-LLM/checkpoints/final"
    
    print("="*80)
    print("FULL PRECISION vs 4-BIT QUANTIZED COMPARISON")
    print("="*80)
    print("\nThis test compares mathematical reasoning quality between:")
    print("  1. Current: 4-bit quantized base + adapter")
    print("  2. Better: Full precision base + adapter")
    print("\nIf full precision scores better, quantization is hurting math!")
    
    # Test 4-bit quantized version (current)
    print("\n\n" + "="*80)
    print("TEST 1: 4-BIT QUANTIZED BASE (Current Approach)")
    print("="*80)
    tokenizer_q, model_q = load_model_quantized(adapter_path)
    test_model(tokenizer_q, model_q, "4-bit Quantized Base")
    
    # Clear memory
    del model_q
    torch.cuda.empty_cache()
    
    # Test full precision version (recommended)
    print("\n\n" + "="*80)
    print("TEST 2: FULL PRECISION BASE (Better Approach)")
    print("="*80)
    tokenizer_fp, model_fp = load_model_fullprecision(adapter_path)
    test_model(tokenizer_fp, model_fp, "Full Precision Base")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print("\nIf full precision gives better answers:")
    print("  → Quantization is hurting math reasoning")
    print("  → Re-run benchmark with full precision base")
    print("  → Expected improvement: 47% → 60-70% on math")
    print("\nIf both give same wrong answers:")
    print("  → Problem is training data quality")
    print("  → Need Phase 1C (GPT-5 distillation)")

if __name__ == "__main__":
    main()
