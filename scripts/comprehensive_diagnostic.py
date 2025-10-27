#!/usr/bin/env python3
"""
Comprehensive Diagnostic for Phase 1B Benchmark Issues
Tests model loading, generation, formatting, and comparison with GPT-4
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import json
from pathlib import Path
import sys

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def test_model_loading(model_path):
    """Test 1: Verify model loads correctly."""
    print_section("TEST 1: MODEL LOADING")
    
    try:
        # Check files exist
        adapter_config = Path(model_path) / "adapter_config.json"
        adapter_model = Path(model_path) / "adapter_model.safetensors"
        
        print(f"📁 Model path: {model_path}")
        print(f"✓ adapter_config.json exists: {adapter_config.exists()}")
        print(f"✓ adapter_model.safetensors exists: {adapter_model.exists()}")
        
        if adapter_config.exists():
            with open(adapter_config) as f:
                config = json.load(f)
            print(f"\n📋 Adapter Config:")
            print(f"   Base model: {config.get('base_model_name_or_path')}")
            print(f"   PEFT type: {config.get('peft_type')}")
            print(f"   LoRA rank: {config.get('r')}")
            print(f"   LoRA alpha: {config.get('lora_alpha')}")
            print(f"   Target modules: {config.get('target_modules')}")
        
        print("\n📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"✓ Tokenizer loaded: {tokenizer.__class__.__name__}")
        print(f"   Vocab size: {len(tokenizer)}")
        print(f"   Pad token: {tokenizer.pad_token}")
        print(f"   EOS token: {tokenizer.eos_token}")
        
        if tokenizer.pad_token is None:
            print("   ⚠️  Setting pad_token = eos_token")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
        
        print("\n📥 Loading base model + adapter...")
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model_name = peft_config.base_model_name_or_path
        if not base_model_name:
            raise ValueError("No base model name found in adapter config")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True
        )
        print(f"✓ Base model loaded: {base_model.__class__.__name__}")
        print(f"   Device: {base_model.device}")
        print(f"   Dtype: {base_model.dtype}")
        
        peft_model = PeftModel.from_pretrained(base_model, model_path)
        print(f"✓ Adapter applied")
        
        model = peft_model.merge_and_unload()
        print(f"✓ Adapter merged")
        
        model.eval()
        print(f"✓ Model in eval mode")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n📊 Model Stats:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        return tokenizer, model, True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None, False

def test_chat_formatting(tokenizer):
    """Test 2: Verify chat formatting."""
    print_section("TEST 2: CHAT FORMATTING")
    
    test_prompt = "What is 2 + 2?"
    
    # Test different formats
    formats = {
        "Current (Llama-3.1-Instruct)": f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{test_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

""",
        "Simple": test_prompt,
        "ChatML": f"<|im_start|>user\n{test_prompt}<|im_end|>\n<|im_start|>assistant\n"
    }
    
    for name, formatted in formats.items():
        print(f"\n📝 Format: {name}")
        print(f"   Text: {repr(formatted[:100])}")
        tokens = tokenizer.encode(formatted)
        print(f"   Tokens: {len(tokens)}")
        print(f"   Token IDs (first 10): {tokens[:10]}")
    
    return True

def test_generation(tokenizer, model, test_cases):
    """Test 3: Generate responses for test cases."""
    print_section("TEST 3: MODEL GENERATION")
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'─'*80}")
        print(f"Test Case {i}: {test_case['category']}")
        print(f"{'─'*80}")
        
        prompt = test_case['prompt']
        print(f"\n📝 Prompt: {prompt}")
        
        # Format with Llama-3.1-Instruct template
        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        print(f"\n🔢 Input tokens: {inputs['input_ids'].shape[1]}")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        print(f"🔢 Output tokens: {outputs.shape[1]}")
        print(f"🔢 Generated tokens: {outputs.shape[1] - inputs['input_ids'].shape[1]}")
        
        # Decode with special tokens first
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"\n📤 FULL OUTPUT (with special tokens):")
        print(full_response[:500] + ("..." if len(full_response) > 500 else ""))
        
        # Extract assistant response using special token marker
        if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
            assistant_part = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
            # Remove the end-of-text token if present
            if "<|eot_id|>" in assistant_part:
                assistant_part = assistant_part.split("<|eot_id|>")[0]
            extracted = assistant_part.strip()
        else:
            # Fallback: decode without special tokens and remove prompt
            response_clean = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if response_clean.startswith(prompt):
                extracted = response_clean[len(prompt):].strip()
            else:
                extracted = response_clean.strip()
        
        print(f"\n📤 EXTRACTED RESPONSE:")
        print(extracted)
        
        results.append({
            'category': test_case['category'],
            'prompt': prompt,
            'response': extracted,
            'expected': test_case.get('expected'),
            'input_tokens': inputs['input_ids'].shape[1],
            'output_tokens': outputs.shape[1],
            'generated_tokens': outputs.shape[1] - inputs['input_ids'].shape[1]
        })
    
    return results

def test_response_quality(results):
    """Test 4: Analyze response quality."""
    print_section("TEST 4: RESPONSE QUALITY ANALYSIS")
    
    for i, result in enumerate(results, 1):
        print(f"\n{'─'*80}")
        print(f"Analysis {i}: {result['category']}")
        print(f"{'─'*80}")
        
        response = result['response']
        
        # Basic quality checks
        print(f"\n✓ Response length: {len(response)} chars, {len(response.split())} words")
        print(f"✓ Has numbers: {'Yes' if any(c.isdigit() for c in response) else 'No'}")
        print(f"✓ Has punctuation: {'Yes' if any(c in '.,!?;:' for c in response) else 'No'}")
        print(f"✓ Starts with capital: {'Yes' if response and response[0].isupper() else 'No'}")
        
        # Check for common issues
        issues = []
        if len(response) < 10:
            issues.append("Response too short")
        if response.count('\n\n') > 5:
            issues.append("Too many blank lines")
        if any(marker in response.lower() for marker in ['<|', '|>', 'assistant', 'user', 'system']):
            issues.append("Contains chat markers")
        
        if issues:
            print(f"\n⚠️  Issues detected:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print(f"\n✓ No obvious issues detected")
        
        # Compare with expected (if provided)
        if result.get('expected'):
            print(f"\n📋 Expected answer: {result['expected']}")
            if result['expected'].lower() in response.lower():
                print(f"✓ Expected answer found in response")
            else:
                print(f"⚠️  Expected answer NOT found in response")

def test_benchmark_script_logic():
    """Test 5: Verify benchmark script logic."""
    print_section("TEST 5: BENCHMARK SCRIPT LOGIC")
    
    script_path = Path(__file__).parent / "automated_gpt4_benchmark.py"
    
    if not script_path.exists():
        print(f"❌ Benchmark script not found: {script_path}")
        return False
    
    print(f"✓ Benchmark script exists: {script_path}")
    
    # Check for key components
    with open(script_path) as f:
        content = f.read()
    
    checks = {
        "PEFT import": "from peft import",
        "Adapter detection": "adapter_config.json",
        "Base model loading": "base_model_name_or_path",
        "Adapter merging": "merge_and_unload()",
        "Eval mode": "model.eval()",
        "Llama-3.1 format": "<|start_header_id|>",
        "Pad token config": "pad_token =",
        "Response extraction": "split(",
    }
    
    for check_name, check_pattern in checks.items():
        found = check_pattern in content
        status = "✓" if found else "❌"
        print(f"{status} {check_name}: {'Found' if found else 'MISSING'}")
    
    return True

def main():
    """Run all diagnostics."""
    print_section("COMPREHENSIVE PHASE 1B DIAGNOSTIC")
    print("Testing model loading, generation, formatting, and benchmark logic")
    
    # Configuration
    model_path = "/workspace/data/Cogumi-LLM/checkpoints/final"
    
    # Test cases covering different categories
    test_cases = [
        {
            'category': 'Math (Simple)',
            'prompt': 'Solve this math problem step by step:\n\nJanet has 3 apples. She buys 7 more apples. How many apples does she have?',
            'expected': '10'
        },
        {
            'category': 'Math (Complex)',
            'prompt': 'Solve this math problem step by step:\n\nA store sells pens for $2 each and notebooks for $5 each. If you buy 3 pens and 2 notebooks, how much do you spend in total?',
            'expected': '16'
        },
        {
            'category': 'Code',
            'prompt': 'Complete this Python function:\n\ndef add_two_numbers(a, b):\n    """Return the sum of a and b."""',
            'expected': 'return a + b'
        },
        {
            'category': 'Reasoning',
            'prompt': 'Question: If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?\n\nChoices:\nA. Yes\nB. No\nC. Cannot be determined\n\nAnswer:',
            'expected': 'C'
        },
        {
            'category': 'Knowledge',
            'prompt': 'What is the capital of France?',
            'expected': 'Paris'
        }
    ]
    
    # Run tests
    tokenizer, model, test1_passed = test_model_loading(model_path)
    
    if not test1_passed:
        print("\n❌ CRITICAL: Model loading failed. Cannot continue.")
        sys.exit(1)
    
    test_chat_formatting(tokenizer)
    
    results = test_generation(tokenizer, model, test_cases)
    
    test_response_quality(results)
    
    test_benchmark_script_logic()
    
    # Final summary
    print_section("DIAGNOSTIC SUMMARY")
    
    print("\n📊 Test Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['category']}")
        print(f"   Prompt: {result['prompt'][:60]}...")
        print(f"   Response: {result['response'][:100]}...")
        if result.get('expected'):
            found = result['expected'].lower() in result['response'].lower()
            print(f"   Expected '{result['expected']}': {'✓ FOUND' if found else '❌ NOT FOUND'}")
    
    print("\n" + "="*80)
    print("✅ DIAGNOSTIC COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Review the generated responses above")
    print("2. Check if responses match expected answers")
    print("3. If responses look good, issue may be in GPT-4 judging bias")
    print("4. If responses look bad, investigate training data quality")
    print("\nSave this output for analysis!")

if __name__ == "__main__":
    main()
