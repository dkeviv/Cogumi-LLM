#!/usr/bin/env python3
"""
Deep Diagnostic Script for Model Failure Analysis

Purpose: Understand WHY the model has:
- MATH: 70% ties (capability present but inconsistent)
- CODE: 58% score (competitive but needs improvement)

This script tests consistency by running the same prompt multiple times
and measuring how much the answers vary.

Run on Vast.ai with GPU access.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import re
from datasets import load_dataset
from collections import Counter
from tqdm import tqdm

# Configuration
BASE_MODEL = "unsloth/meta-llama-3.1-8b-instruct-bnb-4bit"
ADAPTER_PATH = "/workspace/data/Cogumi-LLM/checkpoints/final"
OUTPUT_FILE = "/workspace/diagnostic_results.json"

print("🔬 Deep Diagnostic Analysis Starting...")
print("=" * 80)

print("\n📦 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model = model.merge_and_unload()  # type: ignore[assignment]
model.eval()

print("✅ Model loaded successfully")

def extract_answer(response: str, problem_type: str = "math") -> str:
    """Extract answer from response based on problem type."""
    if problem_type == "math":
        # Try multiple patterns
        patterns = [
            r'####\s*([^\n]+)',
            r'\\boxed\{([^}]+)\}',
            r'(?:answer|Answer)[:\s]+([^\n\.]+)',
            r'\$([0-9,]+(?:\.[0-9]+)?)\$',
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: find last number
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response)
        if numbers:
            return numbers[-1]
        return response[-50:]
    
    elif problem_type == "code":
        # Extract code block
        match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response[:200]  # First 200 chars
    
    return response.strip()

def test_consistency(prompt: str, num_runs: int = 10, temp: float = 0.7, 
                    problem_type: str = "math") -> dict:
    """Test response consistency for a single prompt."""
    responses = []
    answers = []
    
    for i in range(num_runs):
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=temp,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if response.startswith(formatted):
            response = response[len(formatted):].strip()
        
        responses.append(response)
        answer = extract_answer(response, problem_type)
        answers.append(answer)
    
    unique = len(set(answers))
    most_common = Counter(answers).most_common(1)[0]
    
    return {
        'responses': responses,
        'answers': answers,
        'unique_count': unique,
        'consistency_rate': most_common[1] / num_runs,
        'most_common_answer': most_common[0],
        'answer_distribution': dict(Counter(answers))
    }

# Test 1: Math problem consistency analysis
print("\n" + "=" * 80)
print("📊 Testing MATH consistency (10 problems × 10 runs = 100 generations)")
print("=" * 80)

gsm8k = load_dataset("gsm8k", "main", split="test")
math_samples = list(gsm8k.select(range(10)))

math_results = []
for i, example in enumerate(tqdm(math_samples, desc="Testing MATH")):
    prompt = example['question']
    correct_answer = example['answer'].split('####')[-1].strip()
    
    result = test_consistency(prompt, num_runs=10, temp=0.7, problem_type="math")
    result['prompt'] = prompt
    result['correct_answer'] = correct_answer
    
    # Don't include full responses in JSON (too large)
    result['sample_responses'] = result['responses'][:2]
    del result['responses']
    
    math_results.append(result)
    
    print(f"\n  Problem {i+1}: Consistency {result['consistency_rate']*100:.0f}%, "
          f"Unique answers: {result['unique_count']}/10")

# Test 2: Code problem consistency analysis
print("\n" + "=" * 80)
print("💻 Testing CODE consistency (10 problems × 10 runs = 100 generations)")
print("=" * 80)

humaneval = load_dataset("openai_humaneval", split="test")
code_samples = list(humaneval.select(range(10)))

code_results = []
for i, example in enumerate(tqdm(code_samples, desc="Testing CODE")):
    prompt = f"Complete this Python function:\n\n{example['prompt']}"
    
    result = test_consistency(prompt, num_runs=10, temp=0.7, problem_type="code")
    result['prompt'] = prompt[:100]  # Truncate for readability
    result['canonical_solution'] = example['canonical_solution']
    
    # Don't include full responses
    result['sample_responses'] = result['responses'][:2]
    del result['responses']
    
    code_results.append(result)
    
    print(f"\n  Problem {i+1}: Consistency {result['consistency_rate']*100:.0f}%, "
          f"Unique solutions: {result['unique_count']}/10")

# Save results
results = {
    'math_consistency': math_results,
    'code_consistency': code_results,
    'test_config': {
        'num_runs_per_problem': 10,
        'temperature': 0.7,
        'do_sample': True,
        'base_model': BASE_MODEL,
        'adapter_path': ADAPTER_PATH
    }
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Diagnostic complete! Results saved to: {OUTPUT_FILE}")

# Print summary
print("\n" + "=" * 80)
print("🎯 DIAGNOSTIC SUMMARY")
print("=" * 80)

avg_math_consistency = sum(r['consistency_rate'] for r in math_results) / len(math_results)
avg_math_unique = sum(r['unique_count'] for r in math_results) / len(math_results)
avg_code_consistency = sum(r['consistency_rate'] for r in code_results) / len(code_results)
avg_code_unique = sum(r['unique_count'] for r in code_results) / len(code_results)

print(f"\n📊 MATH Results:")
print(f"   Average consistency: {avg_math_consistency*100:.1f}%")
print(f"   Average unique answers: {avg_math_unique:.1f}/10")

if avg_math_consistency < 0.6:
    print(f"   ⚠️ MATH consistency is LOW - this EXPLAINS 70% ties!")
    print(f"   → Model generates different answers each time")
    print(f"   → GPT-4 sees \"comparable but different\" → judges as TIE")
elif avg_math_consistency < 0.8:
    print(f"   ⚠️ MATH consistency is MODERATE")
else:
    print(f"   ✅ MATH consistency is GOOD")

print(f"\n💻 CODE Results:")
print(f"   Average consistency: {avg_code_consistency*100:.1f}%")
print(f"   Average unique solutions: {avg_code_unique:.1f}/10")

if avg_code_consistency < 0.6:
    print(f"   ⚠️ CODE consistency is LOW - impacts performance!")
elif avg_code_consistency < 0.8:
    print(f"   ⚠️ CODE consistency is MODERATE")
else:
    print(f"   ✅ CODE consistency is GOOD")

print(f"\n💡 Recommendations:")
if avg_math_consistency < 0.7 or avg_code_consistency < 0.7:
    print(f"   1. Use greedy decoding (do_sample=False) to eliminate randomness")
    print(f"   2. Run self-consistency training to 'bake in' deterministic behavior")
    print(f"   3. Expected improvement: MATH 41% → 70-80%, CODE 58% → 75-80%")
else:
    print(f"   • Consistency is acceptable - ties may be due to formatting")
    print(f"   • Consider response format standardization")

print("\n" + "=" * 80)
print(f"📥 Download {OUTPUT_FILE} and analyze in Benchmark_Diagnostic_Analysis.ipynb")
print("=" * 80)
