#!/usr/bin/env python3
"""
Split data files into chunks for Claude Desktop (31MB limit).

Splits both test_dataset and model_outputs into matching chunks.
"""

import json
from pathlib import Path

def split_files(test_path, model_path, output_dir, examples_per_chunk=6667):
    """Split both files into matching chunks."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Read test dataset
    print(f"Reading {test_path}...")
    test_data = []
    with open(test_path, 'r') as f:
        for line in f:
            test_data.append(line)
    
    print(f"Reading {model_path}...")
    model_data = []
    with open(model_path, 'r') as f:
        for line in f:
            model_data.append(line)
    
    print(f"Total examples: {len(test_data)} test, {len(model_data)} model")
    
    # Split into chunks
    num_chunks = (len(test_data) + examples_per_chunk - 1) // examples_per_chunk
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * examples_per_chunk
        end_idx = min(start_idx + examples_per_chunk, len(test_data))
        
        # Write test chunk
        test_chunk_path = output_dir / f"test_dataset_part{chunk_idx+1}.jsonl"
        with open(test_chunk_path, 'w') as f:
            for line in test_data[start_idx:end_idx]:
                f.write(line)
        
        # Write model chunk
        model_chunk_path = output_dir / f"model_outputs_part{chunk_idx+1}.jsonl"
        with open(model_chunk_path, 'w') as f:
            for line in model_data[start_idx:end_idx]:
                f.write(line)
        
        # Get sizes
        test_size = test_chunk_path.stat().st_size / (1024*1024)
        model_size = model_chunk_path.stat().st_size / (1024*1024)
        
        print(f"\nChunk {chunk_idx+1}:")
        print(f"  Examples: {start_idx} to {end_idx-1} ({end_idx-start_idx} total)")
        print(f"  Test file: {test_chunk_path.name} ({test_size:.1f} MB)")
        print(f"  Model file: {model_chunk_path.name} ({model_size:.1f} MB)")
        
        if test_size > 31 or model_size > 31:
            print(f"  ⚠️  WARNING: File(s) exceed 31MB limit!")
    
    print(f"\n✅ Created {num_chunks} matched chunk pairs in {output_dir}")
    print(f"\nUpload to Claude Desktop in order:")
    for i in range(num_chunks):
        print(f"  Batch {i+1}: test_dataset_part{i+1}.jsonl + model_outputs_part{i+1}.jsonl")

if __name__ == '__main__':
    split_files(
        test_path='./Phase 1B_2_0/data/test_dataset_20k.jsonl',
        model_path='./Phase 1B_2_0/data/model_outputs_20k.jsonl',
        output_dir='./Phase 1B_2_0/data/chunks/',
        examples_per_chunk=6667  # ~3 chunks total
    )
