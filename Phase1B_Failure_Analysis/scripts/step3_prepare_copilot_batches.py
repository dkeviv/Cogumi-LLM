#!/usr/bin/env python3
"""
Phase 1B - Step 3: Prepare Batches for Copilot Chat Manual Comparison

Splits 20K model outputs into small batches for manual review by Copilot Chat (Claude Sonnet 4.5).

Usage:
    python "Phase 1B_2_0/step3_prepare_copilot_batches.py" \
        --model_outputs ./data/phase1b/model_outputs_20k.jsonl \
        --output_dir ./data/phase1b/copilot_batches \
        --batch_size 100
        
Output:
    - Creates batch files: batch_001.json, batch_002.json, etc.
    - Each batch contains 100 examples for Copilot to review
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm


def create_batches(model_outputs_path: str, output_dir: str, batch_size: int = 100):
    """Split model outputs into batches for Copilot Chat review."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Loading model outputs from: {model_outputs_path}")
    
    # Load all items
    items = []
    with open(model_outputs_path, 'r') as f:
        for line in f:
            items.append(json.loads(line.strip()))
    
    print(f"✅ Loaded {len(items):,} items")
    
    # Create batches
    total_batches = (len(items) + batch_size - 1) // batch_size
    print(f"\n📊 Creating {total_batches} batches of {batch_size} items each...")
    
    batch_files = []
    for batch_idx in tqdm(range(0, len(items), batch_size), desc="Creating batches"):
        batch_items = items[batch_idx:batch_idx + batch_size]
        batch_num = (batch_idx // batch_size) + 1
        
        # Prepare batch for Copilot review
        batch_data = {
            "batch_number": batch_num,
            "total_batches": total_batches,
            "items_in_batch": len(batch_items),
            "examples": []
        }
        
        for item in batch_items:
            # Truncate long texts for readability
            inst = item['instruction'][:500]  # First 500 chars
            ref = item['reference'][:800]      # First 800 chars
            out = item['model_output'][:800]   # First 800 chars
            
            batch_data["examples"].append({
                "id": item['id'],
                "instruction": inst,
                "reference": ref,
                "model_output": out,
                "category": item['category']
            })
        
        # Save batch
        batch_file = output_path / f"batch_{batch_num:03d}.json"
        with open(batch_file, 'w') as f:
            json.dump(batch_data, f, indent=2)
        
        batch_files.append(str(batch_file))
    
    print(f"\n✅ Created {len(batch_files)} batch files in: {output_dir}")
    print(f"\n📝 Next Steps:")
    print(f"   1. Share each batch file with Copilot Chat")
    print(f"   2. Ask Copilot to compare model outputs with references")
    print(f"   3. Copilot will return JSON with PASS/FAIL results")
    print(f"   4. Save Copilot's responses as response_001.json, response_002.json, etc.")
    print(f"   5. Run step3_merge_copilot_results.py to compile all results")
    
    # Create instructions file
    instructions_file = output_path / "INSTRUCTIONS.md"
    with open(instructions_file, 'w') as f:
        f.write("""# Copilot Chat Comparison Instructions

## Overview
You'll be comparing 20,000 model outputs with reference answers using Copilot Chat (Claude Sonnet 4.5).

## Workflow

### Step 1: Share Batch with Copilot
Copy a batch file (e.g., `batch_001.json`) and paste into Copilot Chat with this prompt:

```
I have a batch of model outputs to compare with reference answers. For each example:
- Compare the model_output with the reference
- Determine if it's PASS (correct, complete, quality 7+/10) or FAIL (errors, incomplete, quality <7/10)

Return a JSON array with results:
[
  {"id": 0, "status": "PASS"},
  {"id": 1, "status": "FAIL", "reason": "brief explanation"},
  ...
]

Here's the batch:
[paste batch_001.json content here]
```

### Step 2: Save Copilot's Response
Save Copilot's JSON response as `response_001.json` in the `responses/` directory.

### Step 3: Repeat for All Batches
Process all batches (batch_001 to batch_200) and save corresponding responses.

### Step 4: Merge Results
Run: `python Phase1B_2_0/step3_merge_copilot_results.py`

## Tips
- Process batches in order (001, 002, 003, ...)
- Double-check that response numbers match batch numbers
- If Copilot gets confused, break batch into smaller chunks
- Estimated time: ~30 seconds per batch = ~1.5-2 hours total
""")
    
    print(f"\n📖 Instructions saved to: {instructions_file}")
    
    return batch_files


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Prepare batches for Copilot Chat manual comparison"
    )
    
    parser.add_argument(
        "--model_outputs",
        type=str,
        required=True,
        help="Path to model outputs from Step 2"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save batch files"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Number of examples per batch (default: 100)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 PHASE 1B STEP 3: PREPARE BATCHES FOR COPILOT CHAT")
    print("=" * 80)
    print(f"\nBatch size: {args.batch_size} examples per batch")
    print(f"Output directory: {args.output_dir}")
    
    batch_files = create_batches(
        args.model_outputs,
        args.output_dir,
        args.batch_size
    )
    
    print(f"\n✨ Done! Ready for Copilot Chat comparison.")


if __name__ == "__main__":
    main()
