#!/usr/bin/env python3
"""
Phase 1B - Step 3: Proper LLM-Based Evaluation

Uses FREE Llama-405B via HuggingFace Inference API for semantic evaluation.

Key improvements:
1. Detailed evaluation criteria (correctness, completeness, accuracy, relevance)
2. Clear PASS/FAIL guidelines 
3. Batch processing with progress tracking
4. Automatic retry on failures
5. Cost tracking (FREE with HF)

Expected results:
- 72-76% pass rate (Phase 1A baseline: 75-82% GPT-4)
- ~5,000-6,000 failures out of 20K
- Clear failure patterns for Phase 1B Step 4 clustering

Usage:
    export HF_TOKEN="your_token_here"
    python "Phase 1B_2_0/step3_llm_evaluation.py" \\
        --test_dataset "./Phase 1B_2_0/data/test_dataset_20k.jsonl" \\
        --model_outputs "./Phase 1B_2_0/data/model_outputs_20k.jsonl" \\
        --output_path "./Phase 1B_2_0/data/llm_evaluation_results.jsonl" \\
        --batch_size 50 \\
        --start_index 0 \\
        --max_examples 20000
"""

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import time
import re


def create_evaluation_prompt(instruction, reference, model_output, category):
    """
    Create detailed evaluation prompt with clear criteria.
    
    This is the improved prompt that focuses on semantic correctness,
    not superficial word matching.
    """
    # Truncate very long inputs for context window
    if len(model_output) > 2000:
        model_output = model_output[:2000] + "\n... [truncated]"
    if len(reference) > 2000:
        reference = reference[:2000] + "\n... [truncated]"
    
    prompt = f"""Evaluate this model output for correctness.

**CATEGORY:** {category}

**INSTRUCTION:**
{instruction}

**REFERENCE ANSWER:**
{reference}

**MODEL OUTPUT:**
{model_output}

**EVALUATION CRITERIA:**

1. CORRECTNESS - Is the core answer/information correct?
2. COMPLETENESS - Does it address all key points from the instruction?
3. ACCURACY - Are facts, logic, code syntax, calculations correct?
4. RELEVANCE - Is it on-topic and addressing the right question?

**IMPORTANT NOTES:**
- Verbose but correct outputs should PASS (detailed explanations are good)
- Different phrasing is acceptable (semantic equivalence matters)
- Minor style differences are OK
- For code: focus on logic correctness, not exact implementation
- For math: focus on final answer and reasoning validity
- For QA: focus on whether question is answered accurately

**WHAT COUNTS AS FAIL:**
- Wrong answer or incorrect conclusion
- Missing major required information
- Factual errors or hallucinations  
- Off-topic or misunderstood instruction
- Broken code or wrong syntax
- Incorrect calculations
- Nonsensical or repetitive garbage

Return ONLY this JSON (no other text):
{{"status": "PASS" or "FAIL", "reason": "brief explanation (1 sentence)"}}"""
    
    return prompt


def call_llm_api(prompt, model="meta-llama/Llama-3.1-405B-Instruct", max_retries=3):
    """
    Call HuggingFace Inference API with retries.
    """
    from huggingface_hub import InferenceClient
    
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise ValueError("HF_TOKEN environment variable not set. Get one from https://huggingface.co/settings/tokens")
    
    client = InferenceClient(token=token)
    
    for attempt in range(max_retries):
        try:
            response = client.text_generation(
                prompt,
                model=model,
                max_new_tokens=150,
                temperature=0.1,  # Low temperature for consistency
                return_full_text=False
            )
            return response
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"   ⚠️ API error (attempt {attempt+1}): {str(e)[:100]}")
                print(f"   ⏳ Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"   ❌ Failed after {max_retries} attempts: {str(e)[:100]}")
                return None
    
    return None


def parse_llm_response(response_text):
    """Parse LLM response to extract status and reason."""
    try:
        # Try to find JSON in response
        response_text = response_text.strip()
        
        # Remove markdown code blocks if present
        if '```json' in response_text:
            start = response_text.find('```json') + 7
            end = response_text.find('```', start)
            response_text = response_text[start:end].strip()
        elif '```' in response_text:
            start = response_text.find('```') + 3
            end = response_text.find('```', start)
            response_text = response_text[start:end].strip()
        
        # Parse JSON
        result = json.loads(response_text)
        
        if 'status' in result and 'reason' in result:
            status = result['status'].upper()
            if status in ['PASS', 'FAIL']:
                return status, result['reason']
        
        # Fallback: look for PASS or FAIL in text
        if 'PASS' in response_text.upper():
            return 'PASS', 'Parsed from text'
        elif 'FAIL' in response_text.upper():
            return 'FAIL', 'Parsed from text'
        
    except Exception as e:
        print(f"   ⚠️ Parse error: {str(e)[:100]}")
        print(f"   Response: {response_text[:200]}")
    
    return 'UNCERTAIN', f'Failed to parse response: {response_text[:100]}'


def evaluate_with_llm(test_dataset_path, model_outputs_path, output_path, 
                      batch_size=50, start_index=0, max_examples=None):
    """
    Evaluate all examples using LLM API.
    """
    print(f"\n📂 Loading test dataset from: {test_dataset_path}")
    test_data = []
    with open(test_dataset_path, 'r') as f:
        for idx, line in enumerate(f):
            if idx < start_index:
                continue
            if max_examples and len(test_data) >= max_examples:
                break
            
            item = json.loads(line.strip())
            item['id'] = idx
            if 'response' in item and 'reference' not in item:
                item['reference'] = item['response']
            test_data.append(item)
    
    print(f"✅ Loaded {len(test_data):,} test examples (indices {start_index}-{start_index+len(test_data)-1})")
    
    print(f"\n📂 Loading model outputs from: {model_outputs_path}")
    model_data = {}
    with open(model_outputs_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            if start_index <= item['id'] < start_index + len(test_data):
                model_data[item['id']] = item
    
    print(f"✅ Loaded {len(model_data):,} model outputs")
    
    # Evaluate with LLM
    print(f"\n🤖 Evaluating with LLM (Llama-405B via HuggingFace)...")
    print(f"⚠️  This will make {len(test_data):,} API calls (FREE but may take time)")
    
    results = []
    stats = defaultdict(lambda: {'total': 0, 'pass': 0, 'fail': 0, 'uncertain': 0})
    
    output_file = open(output_path, 'w')
    failures_file = open(output_path.replace('.jsonl', '_failures.jsonl'), 'w')
    
    api_calls = 0
    start_time = time.time()
    
    try:
        for test_item in tqdm(test_data, desc="LLM Evaluation"):
            item_id = test_item['id']
            model_item = model_data.get(item_id)
            
            if not model_item:
                result = {
                    'id': item_id,
                    'instruction': test_item['instruction'],
                    'reference': test_item['reference'],
                    'model_output': None,
                    'category': test_item['category'],
                    'status': 'FAIL',
                    'reason': 'Missing model output',
                    'confidence': 1.0,
                    'evaluation_method': 'automatic'
                }
                results.append(result)
                output_file.write(json.dumps(result) + '\n')
                failures_file.write(json.dumps(result) + '\n')
                
                stats[test_item['category']]['total'] += 1
                stats[test_item['category']]['fail'] += 1
                stats['overall']['total'] += 1
                stats['overall']['fail'] += 1
                continue
            
            # Create evaluation prompt
            prompt = create_evaluation_prompt(
                test_item['instruction'],
                test_item['reference'],
                model_item['model_output'],
                test_item['category']
            )
            
            # Call LLM API
            response_text = call_llm_api(prompt)
            api_calls += 1
            
            if response_text:
                status, reason = parse_llm_response(response_text)
            else:
                status, reason = 'UNCERTAIN', 'API call failed'
            
            result = {
                'id': item_id,
                'instruction': test_item['instruction'],
                'reference': test_item['reference'],
                'model_output': model_item['model_output'],
                'category': test_item['category'],
                'status': status,
                'reason': reason,
                'confidence': 0.9 if status != 'UNCERTAIN' else 0.0,
                'evaluation_method': 'llm'
            }
            
            results.append(result)
            output_file.write(json.dumps(result) + '\n')
            output_file.flush()
            
            if status == 'FAIL':
                failures_file.write(json.dumps(result) + '\n')
                failures_file.flush()
            
            # Update stats
            stats[test_item['category']]['total'] += 1
            if status == 'PASS':
                stats[test_item['category']]['pass'] += 1
                stats['overall']['pass'] += 1
            elif status == 'FAIL':
                stats[test_item['category']]['fail'] += 1
                stats['overall']['fail'] += 1
            else:
                stats[test_item['category']]['uncertain'] += 1
                stats['overall']['uncertain'] += 1
            
            stats['overall']['total'] += 1
            
            # Progress update every 100 items
            if api_calls % 100 == 0:
                elapsed = time.time() - start_time
                rate = api_calls / elapsed
                remaining = (len(test_data) - api_calls) / rate if rate > 0 else 0
                current_pass_rate = stats['overall']['pass'] / stats['overall']['total'] * 100
                print(f"\n   Progress: {api_calls}/{len(test_data)} ({api_calls/len(test_data)*100:.1f}%)")
                print(f"   Current pass rate: {current_pass_rate:.1f}%")
                print(f"   Speed: {rate:.1f} items/sec")
                print(f"   Est. remaining: {remaining/60:.1f} minutes")
    
    finally:
        output_file.close()
        failures_file.close()
    
    # Print final results
    elapsed_total = time.time() - start_time
    
    print(f"\n✅ Evaluation complete!")
    print(f"   Total time: {elapsed_total/60:.1f} minutes")
    print(f"   API calls: {api_calls:,}")
    print(f"   Average speed: {api_calls/elapsed_total:.2f} items/sec")
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"{'Category':<15} {'Total':<10} {'Pass':<12} {'Fail':<12} {'Uncertain':<12}")
    print("-" * 70)
    
    for cat in sorted(stats.keys()):
        if cat == 'overall':
            print("-" * 70)
        s = stats[cat]
        pass_pct = (s['pass'] / s['total'] * 100) if s['total'] > 0 else 0
        fail_pct = (s['fail'] / s['total'] * 100) if s['total'] > 0 else 0
        unc_pct = (s['uncertain'] / s['total'] * 100) if s['total'] > 0 else 0
        print(f"{cat:<15} {s['total']:<10} {s['pass']:<5} ({pass_pct:>4.1f}%) {s['fail']:<5} ({fail_pct:>4.1f}%) {s['uncertain']:<5} ({unc_pct:>4.1f}%)")
    
    # Save summary
    summary_path = output_path.replace('.jsonl', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(dict(stats), f, indent=2)
    
    print(f"\n💾 Files saved:")
    print(f"   Results: {output_path}")
    print(f"   Failures: {output_path.replace('.jsonl', '_failures.jsonl')}")
    print(f"   Summary: {summary_path}")
    
    # Compare to expected
    expected_pass_rate = 0.72  # Lower bound of 72-76% expected
    actual_pass_rate = stats['overall']['pass'] / stats['overall']['total']
    
    print(f"\n📈 PERFORMANCE ANALYSIS:")
    print(f"   Expected pass rate: {expected_pass_rate*100:.1f}% (Phase 1A baseline: 75-82% GPT-4)")
    print(f"   Actual pass rate: {actual_pass_rate*100:.1f}%")
    
    if actual_pass_rate >= expected_pass_rate:
        print(f"   ✅ MEETS EXPECTATIONS! Model performing as expected.")
    elif actual_pass_rate >= expected_pass_rate - 0.10:
        print(f"   ⚠️  SLIGHTLY BELOW - Within 10% of expected, acceptable")
    else:
        print(f"   ❌ BELOW EXPECTATIONS - May need to review training data quality")
    
    return results, stats


def main():
    parser = argparse.ArgumentParser(description='LLM-based semantic evaluation')
    parser.add_argument('--test_dataset', type=str, required=True)
    parser.add_argument('--model_outputs', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size for progress tracking')
    parser.add_argument('--start_index', type=int, default=0,
                        help='Start from this index (for resuming)')
    parser.add_argument('--max_examples', type=int, default=None,
                        help='Max examples to evaluate (for testing)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for HF_TOKEN
    if not os.environ.get('HF_TOKEN'):
        print("⚠️  HF_TOKEN not set!")
        print("Get a token from: https://huggingface.co/settings/tokens")
        print("Then set it: export HF_TOKEN='your_token_here'")
        return
    
    # Run evaluation
    results, stats = evaluate_with_llm(
        args.test_dataset,
        args.model_outputs,
        args.output_path,
        batch_size=args.batch_size,
        start_index=args.start_index,
        max_examples=args.max_examples
    )


if __name__ == '__main__':
    main()
