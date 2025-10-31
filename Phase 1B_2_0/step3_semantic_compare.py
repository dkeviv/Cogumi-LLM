#!/usr/bin/env python3
"""
Phase 1B - Step 3: Semantic Comparison with Claude Sonnet 4.5

Uses LLM-based semantic evaluation to compare model outputs with references.
This provides intelligent assessment beyond simple word matching.

Context: Phase 1B Step 3 - Identify Real Failures
- Phase 1A trained a 10GB base model on 600K examples (75-82% GPT-4)
- Phase 1B tests on 20K diverse examples to find weaknesses
- This script compares model outputs semantically to identify genuine failures
- Goal: Find 12-14K failures to cluster into 8-12 weakness categories
- Why: Target Phase 1C GPT-5 distillation on specific failure patterns

Evaluation Criteria (What to Check):
1. Correctness: Does the model output contain the right answer/information?
2. Completeness: Does it address all aspects of the instruction?
3. Accuracy: Are facts, logic, code syntax, math correct?
4. Relevance: Does it stay on topic without hallucination?

Why These Criteria:
- Correctness: Core requirement - wrong answers are clear failures
- Completeness: Partial answers indicate capability gaps
- Accuracy: Factual errors show knowledge deficits
- Relevance: Off-topic responses show instruction-following issues

Note: Verbosity is NOT a failure - detailed explanations are acceptable
Note: Different phrasing is OK - semantic equivalence matters, not exact wording

Usage:
    python "Phase 1B_2_0/step3_semantic_compare.py" \\
        --test_dataset "./Phase 1B_2_0/data/test_dataset_20k.jsonl" \\
        --model_outputs "./Phase 1B_2_0/data/model_outputs_20k.jsonl" \\
        --output_path "./Phase 1B_2_0/data/comparison_results_semantic.jsonl" \\
        --batch_size 10 \\
        --sample_size 1000

Output:
    - comparison_results_semantic.jsonl: All results with PASS/FAIL
    - failures_semantic.jsonl: Only failures for clustering
    - summary_semantic.json: Statistics by category
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import time


def create_evaluation_prompt(instruction, reference, model_output, category):
    """
    Create detailed evaluation prompt for LLM comparison.
    
    This prompt guides the LLM to perform semantic evaluation with specific criteria.
    """
    prompt = f"""You are evaluating a language model's output quality.

**CONTEXT:**
- A base model (Llama-3.1-8B trained on 600K examples) generated an answer
- We need to identify REAL failures where the model is wrong, incomplete, or off-topic
- Verbose but correct answers should PASS (detailed explanations are good!)
- Different phrasing is OK (semantic equivalence matters, not exact wording)

**TASK CATEGORY:** {category}

**INSTRUCTION TO MODEL:**
{instruction}

**REFERENCE ANSWER (What we expect):**
{reference}

**MODEL OUTPUT (What the model generated):**
{model_output}

**EVALUATION CRITERIA - Check these in order:**

1. **CORRECTNESS** (Most Important)
   - Does the model output contain the right answer/information?
   - For code: Is the logic correct? Does it solve the problem?
   - For math: Is the final answer correct? Is the reasoning valid?
   - For QA: Does it answer the question accurately?
   - For reasoning: Is the conclusion sound and properly justified?

2. **COMPLETENESS**
   - Does the model address ALL parts of the instruction?
   - Are key points from the reference covered (even if phrased differently)?
   - Missing 1-2 minor details is OK, missing major points is FAIL

3. **ACCURACY**
   - Are facts correct (no hallucinations)?
   - Is logic sound (no reasoning errors)?
   - Is syntax correct (for code)?
   - Are calculations correct (for math)?

4. **RELEVANCE**
   - Does the output stay on topic?
   - Does it follow the instruction's intent?
   - Is it addressing the right question?

**WHAT COUNTS AS PASS:**
- Core answer is correct (even if verbose or differently phrased)
- Minor differences in explanation style are OK
- Additional helpful context is OK (not a failure)
- Slightly longer explanations are OK (verbosity ≠ failure)

**WHAT COUNTS AS FAIL:**
- Wrong answer or conclusion
- Missing major required information
- Factual errors or hallucinations
- Off-topic or misunderstood instruction
- Incomplete code or incorrect syntax
- Wrong math calculations
- Nonsensical or repetitive garbage

**OUTPUT FORMAT (JSON only, no explanation):**
{{
  "status": "PASS" or "FAIL",
  "reason": "Brief explanation of why (1 sentence)",
  "correctness": "correct/incorrect/partial",
  "completeness": "complete/incomplete",
  "accuracy": "accurate/inaccurate",
  "relevance": "relevant/irrelevant"
}}

**IMPORTANT:** Be fair but rigorous. We want to find REAL failures for targeted training.
Evaluate now:"""
    
    return prompt


def parse_llm_response(response_text):
    """Parse LLM evaluation response."""
    try:
        # Try to extract JSON from response
        response_text = response_text.strip()
        
        # If response contains markdown code blocks, extract JSON
        if '```json' in response_text:
            start = response_text.find('```json') + 7
            end = response_text.find('```', start)
            response_text = response_text[start:end].strip()
        elif '```' in response_text:
            start = response_text.find('```') + 3
            end = response_text.find('```', start)
            response_text = response_text[start:end].strip()
        
        result = json.loads(response_text)
        
        # Validate required fields
        if 'status' not in result or 'reason' not in result:
            return None
        
        # Standardize status
        result['status'] = result['status'].upper()
        if result['status'] not in ['PASS', 'FAIL']:
            return None
        
        return result
    
    except Exception as e:
        print(f"⚠️  Failed to parse LLM response: {e}")
        print(f"Response: {response_text[:200]}")
        return None


def evaluate_with_copilot(instruction, reference, model_output, category):
    """
    Placeholder for actual Copilot/Claude evaluation.
    
    In practice, this would call the LLM API. For now, it returns a template
    that Copilot can fill in when reviewing batches.
    """
    prompt = create_evaluation_prompt(instruction, reference, model_output, category)
    
    # This is where you would call the LLM API
    # For now, return the prompt so Copilot can evaluate
    return {
        "prompt": prompt,
        "status": None,
        "reason": "Needs manual evaluation"
    }


def batch_evaluate(batch_items):
    """
    Evaluate a batch of examples.
    
    This function prepares evaluation prompts for a batch of examples.
    In production, this would make parallel API calls for efficiency.
    """
    results = []
    for item in batch_items:
        eval_result = evaluate_with_copilot(
            item['instruction'],
            item['reference'],
            item['model_output'],
            item['category']
        )
        results.append(eval_result)
    
    return results


def compare_files(test_dataset_path, model_outputs_path, output_path, batch_size=10, sample_size=None):
    """
    Compare model outputs with test dataset using semantic evaluation.
    
    Args:
        test_dataset_path: Path to test dataset (JSONL)
        model_outputs_path: Path to model outputs (JSONL)
        output_path: Path to save comparison results (JSONL)
        batch_size: Number of examples to evaluate in parallel
        sample_size: If set, only evaluate first N examples (for testing)
    """
    
    print(f"\n📂 Loading test dataset from: {test_dataset_path}")
    test_data = []
    with open(test_dataset_path, 'r') as f:
        for idx, line in enumerate(f):
            item = json.loads(line.strip())
            item['id'] = idx
            # Standardize field names
            if 'response' in item and 'reference' not in item:
                item['reference'] = item['response']
            test_data.append(item)
            
            if sample_size and len(test_data) >= sample_size:
                break
    
    print(f"✅ Loaded {len(test_data):,} test examples")
    
    print(f"\n📂 Loading model outputs from: {model_outputs_path}")
    model_data = {}
    with open(model_outputs_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            model_data[item['id']] = item
    
    print(f"✅ Loaded {len(model_data):,} model outputs")
    
    # Prepare comparison pairs
    comparison_pairs = []
    for test_item in test_data:
        item_id = test_item['id']
        model_item = model_data.get(item_id)
        
        if not model_item:
            comparison_pairs.append({
                'id': item_id,
                'instruction': test_item['instruction'],
                'reference': test_item['reference'],
                'model_output': None,
                'category': test_item['category']
            })
        else:
            comparison_pairs.append({
                'id': item_id,
                'instruction': test_item['instruction'],
                'reference': test_item['reference'],
                'model_output': model_item['model_output'],
                'category': test_item['category']
            })
    
    print(f"\n🔍 Evaluating {len(comparison_pairs):,} pairs...")
    print(f"⚠️  NOTE: This script prepares evaluation prompts.")
    print(f"⚠️  For actual evaluation, you need to call an LLM API (Claude/GPT/etc)")
    
    # Create output file for evaluation prompts
    prompts_path = output_path.replace('.jsonl', '_prompts.jsonl')
    
    with open(prompts_path, 'w') as f:
        for pair in tqdm(comparison_pairs, desc="Preparing prompts"):
            prompt = create_evaluation_prompt(
                pair['instruction'],
                pair['reference'],
                pair['model_output'] if pair['model_output'] else "[NO OUTPUT]",
                pair['category']
            )
            
            output_item = {
                'id': pair['id'],
                'instruction': pair['instruction'],
                'reference': pair['reference'],
                'model_output': pair['model_output'],
                'category': pair['category'],
                'evaluation_prompt': prompt
            }
            
            f.write(json.dumps(output_item) + '\n')
    
    print(f"\n✅ Saved {len(comparison_pairs):,} evaluation prompts to:")
    print(f"   {prompts_path}")
    print(f"\n📋 NEXT STEPS:")
    print(f"   1. Review the prompts in: {prompts_path}")
    print(f"   2. Feed these to an LLM API (Claude, GPT, etc) for evaluation")
    print(f"   3. Parse responses and generate final comparison results")
    
    return prompts_path


def main():
    parser = argparse.ArgumentParser(description='Semantic comparison of model outputs')
    parser.add_argument('--test_dataset', type=str, required=True,
                        help='Path to test dataset JSONL')
    parser.add_argument('--model_outputs', type=str, required=True,
                        help='Path to model outputs JSONL')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save comparison results')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for parallel evaluation')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='If set, only evaluate first N examples (for testing)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run comparison
    prompts_path = compare_files(
        args.test_dataset,
        args.model_outputs,
        args.output_path,
        batch_size=args.batch_size,
        sample_size=args.sample_size
    )
    
    print(f"\n✅ Evaluation prompts ready!")


if __name__ == '__main__':
    main()
