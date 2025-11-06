#!/usr/bin/env python3
"""
Phase 1B - Step 1: Create Curated Test Dataset

Samples 20K examples from the training dataset with proper category representation.
Ensures diverse coverage across different task types.

Usage:
    python "Phase1B_2_0/step1_create_test_dataset.py" \
        --dataset_path ./Phase1A_2_0/data/public_500k_filtered.jsonl \
        --output_path ./data/phase1b/test_dataset_20k.jsonl \
        --num_samples 20000

Output:
    - test_dataset_20k.jsonl: Curated test examples with expected responses
    - test_dataset_stats.json: Statistics about category distribution
"""

import argparse
import json
from pathlib import Path
import random
from typing import List, Dict
from collections import defaultdict


def detect_category(instruction: str, response: str) -> str:
    """
    Detect the category of an example based on instruction and response.
    
    Categories:
    - math: Mathematical reasoning, calculations, word problems
    - code: Programming, code generation, debugging
    - reasoning: Logical reasoning, analysis, problem-solving
    - creative: Creative writing, storytelling, brainstorming
    - qa: Question answering, factual information
    - other: Everything else
    """
    instruction_lower = instruction.lower()
    response_lower = response.lower()
    
    # Math indicators
    math_keywords = ['calculate', 'solve', 'equation', 'number', 'math', 'sum', 'divide', 
                     'multiply', 'subtract', 'add', 'percentage', 'fraction', 'problem']
    if any(kw in instruction_lower for kw in math_keywords):
        return 'math'
    
    # Code indicators
    code_keywords = ['code', 'function', 'program', 'debug', 'implement', 'python', 'javascript',
                     'java', 'algorithm', 'class', 'method', 'variable', 'loop']
    code_markers = ['def ', 'function ', 'class ', 'import ', 'return ', '```', 'for ', 'while ']
    if any(kw in instruction_lower for kw in code_keywords) or any(mk in response_lower for mk in code_markers):
        return 'code'
    
    # Creative indicators
    creative_keywords = ['write a story', 'creative', 'imagine', 'poem', 'essay', 'narrative',
                        'describe', 'brainstorm', 'generate ideas']
    if any(kw in instruction_lower for kw in creative_keywords):
        return 'creative'
    
    # Reasoning indicators
    reasoning_keywords = ['analyze', 'explain', 'reason', 'why', 'how', 'compare', 'evaluate',
                         'argument', 'conclusion', 'logic']
    if any(kw in instruction_lower for kw in reasoning_keywords):
        return 'reasoning'
    
    # QA indicators
    qa_keywords = ['what is', 'what are', 'who is', 'where is', 'when did', 'list', 'define']
    if any(kw in instruction_lower for kw in qa_keywords):
        return 'qa'
    
    return 'other'


def stratified_sample(dataset: List[Dict], num_samples: int) -> List[Dict]:
    """
    Perform stratified sampling to ensure proper representation of each category.
    """
    # Categorize all examples
    categorized = defaultdict(list)
    
    print("Categorizing examples...")
    for i, example in enumerate(dataset):
        if i % 50000 == 0:
            print(f"  Processed {i}/{len(dataset)} examples...")
        
        instruction = example.get('instruction', example.get('prompt', ''))
        response = example.get('response', example.get('output', ''))
        category = detect_category(instruction, response)
        
        categorized[category].append(example)
    
    print(f"\n✅ Categorization complete!")
    print(f"Category distribution in full dataset:")
    for cat, examples in sorted(categorized.items()):
        print(f"  {cat}: {len(examples):,} examples ({len(examples)/len(dataset)*100:.1f}%)")
    
    # Calculate samples per category (proportional to dataset)
    samples_per_category = {}
    for cat, examples in categorized.items():
        proportion = len(examples) / len(dataset)
        samples_per_category[cat] = int(num_samples * proportion)
    
    # Adjust for rounding errors
    total_allocated = sum(samples_per_category.values())
    if total_allocated < num_samples:
        # Add remainder to largest category
        largest_cat = max(samples_per_category, key=samples_per_category.get)
        samples_per_category[largest_cat] += (num_samples - total_allocated)
    
    print(f"\nTarget samples per category:")
    for cat, count in sorted(samples_per_category.items()):
        print(f"  {cat}: {count:,} samples ({count/num_samples*100:.1f}%)")
    
    # Sample from each category
    sampled = []
    for cat, count in samples_per_category.items():
        if count > len(categorized[cat]):
            print(f"⚠️  Warning: {cat} has only {len(categorized[cat])} examples, sampling all")
            sampled.extend(categorized[cat])
        else:
            sampled.extend(random.sample(categorized[cat], count))
    
    # Shuffle final sample
    random.shuffle(sampled)
    
    # Add category labels
    for example in sampled:
        instruction = example.get('instruction', example.get('prompt', ''))
        response = example.get('response', example.get('output', ''))
        example['category'] = detect_category(instruction, response)
    
    return sampled


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Create curated test dataset with category representation"
    )
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to full training dataset"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save curated test dataset"
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20000,
        help="Number of samples to create (default: 20000)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Create output directory
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    stats_path = output_path.parent / "test_dataset_stats.json"
    
    print("=" * 80)
    print("📊 PHASE 1B - STEP 1: CREATE CURATED TEST DATASET")
    print("=" * 80)
    print(f"Input: {args.dataset_path}")
    print(f"Output: {args.output_path}")
    print(f"Target samples: {args.num_samples:,}")
    print(f"Random seed: {args.seed}")
    print("=" * 80)
    print()
    
    # Load full dataset
    print("Loading full dataset...")
    dataset = []
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))
    
    print(f"✅ Loaded {len(dataset):,} examples")
    print()
    
    # Perform stratified sampling
    print("Performing stratified sampling...")
    sampled = stratified_sample(dataset, args.num_samples)
    
    print(f"\n✅ Sampled {len(sampled):,} examples")
    print()
    
    # Calculate final statistics
    category_counts = defaultdict(int)
    for example in sampled:
        category_counts[example['category']] += 1
    
    stats = {
        "total_samples": len(sampled),
        "source_dataset": args.dataset_path,
        "random_seed": args.seed,
        "category_distribution": dict(category_counts),
        "category_percentages": {
            cat: count / len(sampled) * 100 
            for cat, count in category_counts.items()
        }
    }
    
    # Save test dataset
    print("Saving test dataset...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in sampled:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"✅ Saved to {output_path}")
    
    # Save statistics
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Statistics saved to {stats_path}")
    print()
    
    # Print summary
    print("=" * 80)
    print("📊 FINAL DATASET SUMMARY")
    print("=" * 80)
    print(f"Total samples: {len(sampled):,}")
    print()
    print("Category distribution:")
    for cat, count in sorted(category_counts.items()):
        pct = count / len(sampled) * 100
        print(f"  {cat:12s}: {count:5,} samples ({pct:5.1f}%)")
    print("=" * 80)
    print()
    print("✅ Step 1 Complete! Ready for Step 2: Generate model outputs")
    print()


if __name__ == "__main__":
    main()
