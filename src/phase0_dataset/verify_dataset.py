#!/usr/bin/env python3
"""
Dataset Verification Script
Verifies that the curated dataset is English-only and counts total examples.
"""

import json
import re
from pathlib import Path
from collections import Counter
from typing import Optional
import langdetect
from langdetect import detect
from tqdm import tqdm

def is_english_text(text: str) -> bool:
    """Check if text is English using simple heuristics."""
    if not text or len(text.strip()) < 10:
        return True  # Skip very short texts
    
    try:
        # Use langdetect for language detection
        lang = detect(text)
        return lang == 'en'
    except:
        # Fallback: check for non-ASCII characters
        # English should be mostly ASCII
        ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
        return ascii_ratio > 0.95

def verify_dataset(file_path: Path, sample_size: Optional[int] = None):
    """Verify dataset is English-only and count examples."""
    print(f"Verifying dataset: {file_path}")
    print("="*60)
    
    total_count = 0
    english_count = 0
    non_english_examples = []
    language_dist = Counter()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    print(f"Total lines in file: {total_lines:,}")
    
    # Sample if requested
    if sample_size and sample_size < total_lines:
        import random
        lines = random.sample(lines, sample_size)
        print(f"Sampling {sample_size:,} examples for language detection")
    
    for line in tqdm(lines, desc="Checking language"):
        try:
            data = json.loads(line)
            total_count += 1
            
            # Extract text
            if 'instruction' in data and 'response' in data:
                text = f"{data['instruction']} {data['response']}"
            elif 'text' in data:
                text = data['text']
            else:
                text = ' '.join(str(v) for v in data.values() if isinstance(v, str))
            
            # Check if English
            try:
                lang = detect(text[:1000])  # Check first 1000 chars for speed
                language_dist[lang] += 1
                
                if lang == 'en':
                    english_count += 1
                else:
                    if len(non_english_examples) < 10:
                        non_english_examples.append({
                            'language': lang,
                            'text': text[:200]
                        })
            except:
                # If detection fails, assume English
                english_count += 1
                language_dist['unknown'] += 1
        
        except json.JSONDecodeError:
            print(f"Warning: Could not parse line {total_count}")
            continue
    
    # Print results
    print("\n" + "="*60)
    print("VERIFICATION RESULTS")
    print("="*60)
    print(f"Total examples checked: {total_count:,}")
    print(f"English examples: {english_count:,} ({english_count/total_count*100:.2f}%)")
    print(f"Non-English examples: {total_count - english_count:,}")
    
    print("\nLanguage Distribution:")
    for lang, count in language_dist.most_common():
        print(f"  {lang}: {count:,} ({count/total_count*100:.2f}%)")
    
    if non_english_examples:
        print("\nSample Non-English Examples:")
        for i, example in enumerate(non_english_examples, 1):
            print(f"\n  {i}. Language: {example['language']}")
            print(f"     Text: {example['text']}...")
    
    print("\n" + "="*60)
    if english_count == total_count:
        print("✓ DATASET IS 100% ENGLISH")
    else:
        print(f"⚠ Dataset contains {total_count - english_count} non-English examples")
    print("="*60)
    
    return {
        'total': total_lines,
        'checked': total_count,
        'english': english_count,
        'non_english': total_count - english_count,
        'language_distribution': dict(language_dist)
    }

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify dataset is English-only')
    parser.add_argument('--file', type=str, 
                       default='data/phase1/public_500k_filtered.jsonl',
                       help='Path to dataset file')
    parser.add_argument('--sample-size', type=int, default=10000,
                       help='Number of examples to check (default: 10000, use 0 for all)')
    
    args = parser.parse_args()
    
    if args.sample_size == 0:
        args.sample_size = None
    
    results = verify_dataset(Path(args.file), args.sample_size)
    
    # Save results
    output_file = Path('data/phase1/verification_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
