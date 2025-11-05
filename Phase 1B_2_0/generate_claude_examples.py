#!/usr/bin/env python3
"""
Generate targeted training examples for hard failures using Claude Sonnet 4.5 via Copilot.

This script creates batches of failures for manual Claude generation, with prompts
designed to produce high-quality, step-by-step solutions.

Usage:
    python Phase\ 1B_2_0/generate_claude_examples.py \
        --failures Phase\ 1B_2_0/phase1c_hard_failures.jsonl \
        --output Phase\ 1B_2_0/claude_batches \
        --batch_size 50
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict

# Target distribution for 5,000 examples
TARGET_DISTRIBUTION = {
    "code": 2500,      # 50% - most critical
    "math": 1250,      # 25% - second priority  
    "reasoning": 625,  # 12.5%
    "other": 500,      # 10%
    "qa": 125,         # 2.5%
    "creative": 0      # Skip (only 44 examples)
}


def create_prompt_template(category: str, examples: List[Dict[str, Any]], batch_num: int) -> str:
    """Create a prompt for Claude to generate improved examples."""
    
    prompt = f"""# Task: Generate High-Quality Training Examples

You are an expert AI teacher creating training data for a student model. Generate improved solutions for these {category} problems.

## Instructions:
1. For each problem, provide a DETAILED solution with step-by-step reasoning
2. Show your thought process explicitly (Chain-of-Thought)
3. Ensure correctness - reference answer is the ground truth
4. Make the solution pedagogical and clear
5. Format: Return JSON array with instruction, output fields

## Examples to Improve ({len(examples)} problems):

"""
    
    for i, ex in enumerate(examples, 1):
        prompt += f"""
### Problem {i}
**ID:** {ex['id']}
**Category:** {ex['category']}
**Instruction:** {ex['instruction']}
**Reference Answer:** {ex['reference_answer']}
**Previous Wrong Answer:** {ex['previous_output'][:200]}...
**Similarity Score:** {ex['similarity_score']:.3f}

"""
    
    prompt += """
## Output Format:
Return a JSON array of objects with:
```json
[
  {
    "id": "original_id",
    "instruction": "original instruction",
    "output": "Your detailed solution with step-by-step reasoning",
    "meta": {
      "category": "category",
      "teacher": "claude-sonnet-4.5",
      "difficulty": "hard"
    }
  },
  ...
]
```

## Quality Guidelines:
- **Code problems:** Show commented code with explanations
- **Math problems:** Show each calculation step explicitly  
- **Reasoning problems:** Break down logic into clear steps
- **Other problems:** Provide comprehensive, well-structured answers

Generate the improved solutions now:
"""
    
    return prompt


def split_into_batches(failures: List[Dict[str, Any]], output_dir: Path, batch_size: int = 50):
    """Split failures by category and create batch prompts."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by category
    by_category = defaultdict(list)
    for failure in failures:
        cat = failure.get("category", "other")
        by_category[cat].append(failure)
    
    print("\n📊 Category Breakdown:")
    total_generated = 0
    
    for category, target_count in TARGET_DISTRIBUTION.items():
        if target_count == 0:
            continue
            
        available = by_category.get(category, [])
        actual_count = min(len(available), target_count)
        
        print(f"\n{category.upper()}: {actual_count}/{target_count} examples")
        print(f"  Available: {len(available)}")
        
        if actual_count == 0:
            print(f"  ⚠️  Skipping - no examples available")
            continue
        
        # Take up to target_count examples
        examples = available[:actual_count]
        
        # Split into batches
        num_batches = (len(examples) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(examples))
            batch = examples[start:end]
            
            batch_num = batch_idx + 1
            
            # Save examples for this batch
            batch_file = output_dir / f"{category}_batch_{batch_num:03d}_examples.jsonl"
            with open(batch_file, "w", encoding="utf-8") as f:
                for ex in batch:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            
            # Create prompt file
            prompt_file = output_dir / f"{category}_batch_{batch_num:03d}_prompt.txt"
            prompt = create_prompt_template(category, batch, batch_num)
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(prompt)
            
            print(f"  Batch {batch_num}/{num_batches}: {len(batch)} examples → {prompt_file.name}")
            total_generated += len(batch)
    
    print(f"\n✅ Total examples prepared: {total_generated}")
    print(f"📁 Output directory: {output_dir}")
    print(f"\n🚀 Next steps:")
    print(f"1. Open each *_prompt.txt file")
    print(f"2. Copy prompt to Claude Sonnet 4.5 (via Copilot or Claude.ai)")
    print(f"3. Save Claude's JSON response to corresponding *_output.json")
    print(f"4. Run merge script to combine all outputs")


def main():
    parser = argparse.ArgumentParser(description="Generate Claude prompts for hard failures")
    parser.add_argument("--failures", required=True, help="Path to hard_failures.jsonl")
    parser.add_argument("--output", required=True, help="Output directory for batches")
    parser.add_argument("--batch_size", type=int, default=50, help="Examples per batch")
    
    args = parser.parse_args()
    
    # Load failures
    failures = []
    with open(args.failures, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                failures.append(json.loads(line))
    
    print(f"📥 Loaded {len(failures)} hard failures")
    
    # Create batches
    split_into_batches(failures, Path(args.output), args.batch_size)


if __name__ == "__main__":
    main()
