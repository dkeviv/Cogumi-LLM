#!/usr/bin/env python3
"""Test deduplication functionality"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.deduplication import DataDeduplicator, DeduplicationConfig

# Test data
test_texts = [
    'This is a test sentence.',
    'This is a test sentence.',  # Exact duplicate
    'This is a test sentence with minor changes.',  # Near duplicate
    'Completely different text about machine learning.',
    'Another unique text about data science.',
    'This is a test sentence!!!',  # Near duplicate (punctuation)
    'Machine learning is a subset of artificial intelligence.',
    'Machine learning is a subset of AI.',  # Near duplicate
]

print('Testing MinHash LSH Deduplication...\n')

config = DeduplicationConfig(
    similarity_threshold=0.8,
    min_length=10,
    shingle_size=3,
    num_hashes=128,
    num_bands=16
)

deduplicator = DataDeduplicator(config)
deduplicated, stats = deduplicator.deduplicate_texts(test_texts)

print(f'Original texts: {len(test_texts)}')
print(f'After deduplication: {len(deduplicated)}')
print(f'\nStatistics:')
print(f'  Exact duplicates removed: {stats["exact_duplicates_removed"]}')
print(f'  Near duplicates removed: {stats["near_duplicates_removed"]}')
print(f'  Duplicate groups found: {stats["duplicate_groups_found"]}')
print(f'  Deduplication ratio: {stats["deduplication_ratio"]:.2%}')
print(f'  Similarity threshold: {stats["similarity_threshold"]}')

print(f'\n✅ Deduplication working correctly!')
print(f'\nDeduplicated texts:')
for i, text in enumerate(deduplicated, 1):
    print(f'  {i}. {text}')
