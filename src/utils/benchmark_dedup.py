#!/usr/bin/env python3
"""
Benchmark script for parallel deduplication performance testing.
"""

import sys
import time
sys.path.insert(0, '/Users/vivekdurairaj/Projects/Cogumi-LLM/src')

from utils.deduplication_parallel import ParallelDataDeduplicator, DeduplicationConfig

def main():
    # Test with 10K samples
    print("Generating 10K test samples...")
    test_data = [f'Sample text number {i} with some content here' for i in range(10000)]
    
    config = DeduplicationConfig(similarity_threshold=0.8, num_workers=10)
    deduplicator = ParallelDataDeduplicator(config)
    
    print('\n' + '='*60)
    print('PARALLEL DEDUPLICATION BENCHMARK')
    print('='*60)
    print(f'Samples: 10,000')
    print(f'Workers: 10')
    print(f'Threshold: 0.8')
    print('='*60 + '\n')
    
    start = time.time()
    result, stats = deduplicator.deduplicate_texts(test_data)
    elapsed = time.time() - start
    
    throughput = 10000 / elapsed
    projected_637k = (637000 / throughput) / 60
    
    print('\n' + '='*60)
    print('BENCHMARK RESULTS')
    print('='*60)
    print(f'✅ Completed in: {elapsed:.1f}s')
    print(f'📊 Throughput: {throughput:.0f} samples/sec')
    print(f'🎯 Unique samples: {len(result):,} ({len(result)/10000*100:.1f}%)')
    print(f'🔮 Projected time for 637K: {projected_637k:.1f} minutes')
    print('='*60)
    
    if projected_637k > 15:
        print('\n⚠️  WARNING: Performance degradation detected!')
        print(f'   Expected: ~9 minutes')
        print(f'   Projected: {projected_637k:.1f} minutes')
        print('   Bottleneck likely in duplicate group finding.')
    else:
        print('\n✅ Performance looks good!')

if __name__ == '__main__':
    main()
