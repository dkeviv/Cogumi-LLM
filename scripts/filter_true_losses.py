#!/usr/bin/env python3
"""
Filter Training Data to Include ONLY True Losses

PURPOSE:
    Removes "tie" examples from Phase 1B training data. Ties mean the model
    was already correct - training on them causes catastrophic forgetting.
    
PROBLEM DISCOVERED:
    Phase 1B.1 training included 35 ties (74% of MATH data) where model was
    already correct. This caused model to UNLEARN correct behavior:
    - Ties dropped: 70% → 18% (model changed correct answers!)
    - Losses increased: 24% → 78% (catastrophic regression)
    
SOLUTION:
    Keep ONLY loss examples where model was genuinely wrong.
    - MATH: 12 losses (exclude 35 ties)
    - CODE: Check and filter similarly
    
OUTPUT:
    Filtered files: *_losses_only.jsonl (true failures only)
"""

import json
import sys
from pathlib import Path

def filter_losses_only(input_file: str, output_file: str):
    """Filter JSONL to keep only loss examples (exclude ties)"""
    
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        print(f"❌ Error: {input_file} not found")
        return 0, 0
    
    losses_kept = 0
    ties_removed = 0
    
    print(f"📥 Reading: {input_file}")
    
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            data = json.loads(line.strip())
            
            failure_type = data.get('failure_type', '').lower()
            
            if failure_type == 'loss':
                # Keep only true losses
                f_out.write(line)
                losses_kept += 1
            elif failure_type == 'tie':
                # Skip ties (model was already correct!)
                ties_removed += 1
            else:
                print(f"⚠️  Unknown failure_type: {failure_type}")
    
    print(f"✅ Kept {losses_kept} losses")
    print(f"❌ Removed {ties_removed} ties (model was already correct)")
    print(f"💾 Saved to: {output_file}")
    print()
    
    return losses_kept, ties_removed

def main():
    # Filter MATH failures
    math_input = "data/phase1/training_from_benchmark/math_failures_from_benchmark.jsonl"
    math_output = "data/phase1/training_from_benchmark/math_losses_only.jsonl"
    
    # Filter CODE failures  
    code_input = "data/phase1/training_from_benchmark/code_failures_from_benchmark.jsonl"
    code_output = "data/phase1/training_from_benchmark/code_losses_only.jsonl"
    
    print("=" * 80)
    print("🔧 FILTERING TRAINING DATA - LOSSES ONLY")
    print("=" * 80)
    print()
    print("PROBLEM: Training on 'tie' examples causes catastrophic forgetting")
    print("SOLUTION: Keep ONLY 'loss' examples where model was genuinely wrong")
    print()
    print("=" * 80)
    print()
    
    # Filter MATH
    print("📊 MATH Failures:")
    math_losses, math_ties = filter_losses_only(math_input, math_output)
    
    # Filter CODE
    print("💻 CODE Failures:")
    code_losses, code_ties = filter_losses_only(code_input, code_output)
    
    # Summary
    total_losses = math_losses + code_losses
    total_ties = math_ties + code_ties
    
    print("=" * 80)
    print("📋 SUMMARY")
    print("=" * 80)
    print(f"Total losses kept: {total_losses}")
    print(f"Total ties removed: {total_ties}")
    print()
    print(f"New dataset size: {total_losses} examples (was {total_losses + total_ties})")
    print()
    print("NEXT STEPS:")
    print("-" * 80)
    print("1. Review filtered data:")
    print(f"   wc -l data/phase1/training_from_benchmark/*_losses_only.jsonl")
    print()
    print("2. Retrain Phase 1B.1 with filtered data:")
    print("   python train_phase1b_benchmark.py \\")
    print("     --model_name checkpoints/phase1a_merged \\")
    print("     --dataset_path 'data/phase1/training_from_benchmark/*_losses_only.jsonl' \\")
    print("     --output_dir checkpoints/phase1b_losses_only \\")
    print("     --num_train_epochs 2 \\")
    print("     --learning_rate 5e-6")
    print()
    print("3. Validate results:")
    print("   bash scripts/validate_phase1b1.sh")
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
