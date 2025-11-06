"""
Context: Profile deduplication to find bottlenecks at scale.

This script profiles each stage of deduplication to identify performance issues.
"""

import time
import json
import cProfile
import pstats
import io
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.deduplication_parallel import ParallelDataDeduplicator, DeduplicationConfig


def profile_stage(func, *args, **kwargs):
    """Profile a function and return results + timing."""
    profiler = cProfile.Profile()
    
    start_time = time.time()
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()
    elapsed = time.time() - start_time
    
    # Get stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    
    return result, elapsed, s.getvalue()


def main():
    """Profile deduplication on small test set."""
    
    # Load small sample for profiling
    input_path = Path("data/phase1/curated_filtered.jsonl")
    
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        print("Please run dataset curator first to generate test data")
        return
    
    # Load first 50K samples for profiling
    print("Loading 50K samples for profiling...")
    samples = []
    with open(input_path) as f:
        for i, line in enumerate(f):
            if i >= 50000:
                break
            samples.append(json.loads(line)["text"])
    
    print(f"✓ Loaded {len(samples)} samples")
    print("\n" + "="*80)
    print("PROFILING DEDUPLICATION STAGES")
    print("="*80 + "\n")
    
    config = DeduplicationConfig(
        similarity_threshold=0.8,
        num_workers=10
    )
    deduplicator = ParallelDataDeduplicator(config)
    
    # Stage 1: Exact duplicates
    print("\n📊 STAGE 1: Exact Duplicate Removal")
    print("-" * 80)
    texts, exact_time, exact_stats = profile_stage(
        deduplicator._remove_exact_duplicates,
        samples
    )
    print(f"⏱️  Time: {exact_time:.2f}s")
    print(f"📉 Removed: {len(samples) - len(texts)} duplicates")
    print(f"📈 Throughput: {len(samples)/exact_time:.0f} samples/sec")
    print(f"\nTop functions:\n{exact_stats[:1000]}")
    
    # Stage 2: MinHash signatures (parallel)
    print("\n📊 STAGE 2: MinHash Signature Computation (Parallel)")
    print("-" * 80)
    signatures, minhash_time, minhash_stats = profile_stage(
        deduplicator.lsh.compute_signatures_parallel,
        texts
    )
    print(f"⏱️  Time: {minhash_time:.2f}s")
    print(f"📈 Throughput: {len(texts)/minhash_time:.0f} samples/sec")
    print(f"\nTop functions:\n{minhash_stats[:1000]}")
    
    # Stage 3: Build LSH index
    print("\n📊 STAGE 3: Build LSH Index")
    print("-" * 80)
    _, index_time, index_stats = profile_stage(
        deduplicator.lsh.build_index,
        signatures
    )
    print(f"⏱️  Time: {index_time:.2f}s")
    print(f"📈 Throughput: {len(signatures)/index_time:.0f} samples/sec")
    print(f"\nTop functions:\n{index_stats[:1000]}")
    
    # Stage 4: Find duplicate groups (LIKELY BOTTLENECK)
    print("\n📊 STAGE 4: Find Duplicate Groups")
    print("-" * 80)
    groups, groups_time, groups_stats = profile_stage(
        deduplicator._find_duplicate_groups,
        signatures
    )
    print(f"⏱️  Time: {groups_time:.2f}s")
    print(f"📉 Groups found: {len(groups)}")
    print(f"📈 Throughput: {len(signatures)/groups_time:.0f} samples/sec")
    print(f"\nTop functions:\n{groups_stats[:1500]}")
    
    # Summary
    total_time = exact_time + minhash_time + index_time + groups_time
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Stage 1 (Exact):       {exact_time:8.2f}s  ({exact_time/total_time*100:5.1f}%)")
    print(f"Stage 2 (MinHash):     {minhash_time:8.2f}s  ({minhash_time/total_time*100:5.1f}%)")
    print(f"Stage 3 (LSH Index):   {index_time:8.2f}s  ({index_time/total_time*100:5.1f}%)")
    print(f"Stage 4 (Find Groups): {groups_time:8.2f}s  ({groups_time/total_time*100:5.1f}%)")
    print(f"{'─'*40}")
    print(f"TOTAL:                 {total_time:8.2f}s")
    print()
    print(f"Overall throughput: {len(samples)/total_time:.0f} samples/sec")
    print()
    
    # Extrapolate to full dataset
    full_size = 637005
    estimated_full_time = total_time * (full_size / len(samples))
    print(f"📊 Estimated time for {full_size:,} samples: {estimated_full_time/60:.1f} minutes")
    print()
    
    # Identify bottleneck
    stages = [
        ("Exact", exact_time),
        ("MinHash", minhash_time),
        ("LSH Index", index_time),
        ("Find Groups", groups_time)
    ]
    bottleneck = max(stages, key=lambda x: x[1])
    print(f"🔴 BOTTLENECK: {bottleneck[0]} ({bottleneck[1]/total_time*100:.1f}% of time)")
    print()


if __name__ == "__main__":
    main()
