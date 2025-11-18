#!/usr/bin/env python3
"""
Quick test script to verify batch generation works correctly.
Tests with 10 examples to ensure batching doesn't break output quality.
"""

import json
import subprocess
import sys
from pathlib import Path

def main():
    print("=" * 80)
    print("Testing Batch Generation (10 examples)")
    print("=" * 80)
    
    # Test parameters
    model_path = "models/phase1_maml_lora_v2/merged"
    input_file = "data/phase1/answers/training_data_clean.jsonl"
    output_file = "data/phase1e/test_batch_generation.jsonl"
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("Please ensure the merged model is available.")
        sys.exit(1)
    
    # Check if input exists
    if not Path(input_file).exists():
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    # Run batch generation with 10 examples
    cmd = [
        "python", "scripts/phase1e_generate_teacher_outputs.py",
        "--model_path", model_path,
        "--input_file", input_file,
        "--output_file", output_file,
        "--max_examples", "10",
        "--batch_size", "4",  # Small batch for testing
        "--max_new_tokens", "512"  # Shorter for quick test
    ]
    
    print(f"\n🚀 Running command:")
    print(" ".join(cmd))
    print()
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("\n❌ Generation failed!")
        sys.exit(1)
    
    # Verify output
    if not Path(output_file).exists():
        print(f"\n❌ Output file not created: {output_file}")
        sys.exit(1)
    
    # Check output format
    print("\n" + "=" * 80)
    print("Verifying output format...")
    print("=" * 80)
    
    with open(output_file, 'r') as f:
        lines = f.readlines()
    
    print(f"✅ Generated {len(lines)} examples")
    
    # Check first example
    if lines:
        example = json.loads(lines[0])
        print("\n📄 First example:")
        print(f"  Input: {example.get('input', '')[:100]}...")
        print(f"  Output: {example.get('output', '')[:100]}...")
        print(f"  Difficulty: {example.get('difficulty')}")
        print(f"  Domain: {example.get('domain')}")
        
        # Verify structure
        required_fields = ['input', 'output', 'difficulty', 'domain', 'metadata']
        missing = [f for f in required_fields if f not in example]
        
        if missing:
            print(f"\n❌ Missing fields: {missing}")
            sys.exit(1)
        else:
            print(f"\n✅ All required fields present")
    
    print("\n" + "=" * 80)
    print("✅ Batch generation test passed!")
    print("=" * 80)
    print(f"\nYou can now run the full generation with:")
    print(f"  python scripts/phase1e_generate_teacher_outputs.py \\")
    print(f"      --model_path {model_path} \\")
    print(f"      --input_file {input_file} \\")
    print(f"      --output_file data/phase1e/teacher_outputs_53k.jsonl \\")
    print(f"      --batch_size 32 \\")
    print(f"      --max_new_tokens 2048")

if __name__ == "__main__":
    main()
