#!/usr/bin/env python3
"""
Phase 1B - Step 3: Prepare Data for Copilot Chat Comparison

This script formats the model outputs into batches that you can paste into
Copilot Chat. Copilot (Claude Sonnet 4.5) will then compare them directly!

Usage:
    python "Phase1B_2_0/step3_prepare_for_copilot.py" \
        --model_outputs ./data/phase1b/model_outputs_20k.jsonl \
        --output_dir ./data/phase1b/batches \
        --batch_size 100

Then: Copy each batch file and paste to Copilot Chat for comparison!
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm


def create_batch_files(items, output_dir, batch_size=100):
    """
    Create batch files that can be pasted into Copilot Chat.
    Each file contains a batch of examples to compare.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_batches = (len(items) + batch_size - 1) // batch_size
    print(f"Creating {num_batches} batch files ({batch_size} examples each)...")
    print()
    
    batch_files = []
    
    for batch_idx in tqdm(range(0, len(items), batch_size), desc="Creating batches"):
        batch_items = items[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        
        # Create batch content
        content = f"""# Batch {batch_num}/{num_batches} - Phase 1B Comparison

Compare these {len(batch_items)} examples. For each, determine PASS or FAIL:
- PASS: Model output is correct, matches reference quality
- FAIL: Model output has errors, missing info, or poor quality

Return ONLY a JSON array:
[
  {{"id": 0, "status": "PASS"}},
  {{"id": 1, "status": "FAIL", "reason": "brief issue"}},
  ...
]

---EXAMPLES---
"""
        
        for idx, item in enumerate(batch_items):
            content += f"\n### Example {idx}\n"
            content += f"**Question:** {item['instruction'][:300]}\n\n"
            content += f"**Reference:** {item['reference'][:400]}\n\n"
            content += f"**Model Output:** {item['model_output'][:400]}\n\n"
            content += "---\n"
        
        content += "\nReturn JSON array of results now."
        
        # Save batch file
        batch_file = output_dir / f"batch_{batch_num:03d}.txt"
        with open(batch_file, 'w') as f:
            f.write(content)
        
        batch_files.append({
            "batch_num": batch_num,
            "file": str(batch_file),
            "start_idx": batch_idx,
            "end_idx": min(batch_idx + batch_size, len(items)),
            "num_items": len(batch_items)
        })
    
    # Save batch index
    index_file = output_dir / "batch_index.json"
    with open(index_file, 'w') as f:
        json.dump({
            "total_items": len(items),
            "batch_size": batch_size,
            "num_batches": num_batches,
            "batches": batch_files
        }, f, indent=2)
    
    print()
    print("✅ Batch files created!")
    print(f"   Output directory: {output_dir}")
    print(f"   Batch index: {index_file}")
    print()
    
    return batch_files


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for Copilot Chat comparison"
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
    print("🚀 PHASE 1B STEP 3: PREPARE FOR COPILOT COMPARISON")
    print("=" * 80)
    print()
    
    # Load model outputs
    print(f"📂 Loading model outputs from: {args.model_outputs}")
    with open(args.model_outputs, 'r') as f:
        items = [json.loads(line) for line in f]
    print(f"✅ Loaded {len(items):,} items")
    print()
    
    # Create batch files
    batch_files = create_batch_files(items, args.output_dir, args.batch_size)
    
    # Instructions
    print("=" * 80)
    print("📋 NEXT STEPS")
    print("=" * 80)
    print()
    print("1. Open each batch file (starting with batch_001.txt)")
    print("2. Copy the entire content")
    print("3. Paste into Copilot Chat and send")
    print("4. Copy Copilot's JSON response")
    print("5. Save to: data/phase1b/responses/response_001.json")
    print("6. Repeat for all batches")
    print()
    print("After all batches are done:")
    print("   python Phase1B_2_0/step3_merge_copilot_results.py \\")
    print("       --responses_dir ./data/phase1b/responses \\")
    print("       --model_outputs ./data/phase1b/model_outputs_20k.jsonl \\")
    print("       --output_path ./data/phase1b/comparison_results.jsonl")
    print()
    print(f"Total batches to process: {len(batch_files)}")
    print(f"Estimated time: {len(batch_files)} * 30 seconds = {len(batch_files) * 30 / 60:.1f} minutes")
    print()


if __name__ == "__main__":
    main()
