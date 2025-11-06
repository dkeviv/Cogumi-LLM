"""
Quick benchmark to test vectorized similarity optimization.
"""
import time
import json
from src.utils.deduplication_parallel import ParallelDataDeduplicator, DeduplicationConfig

def load_test_data(n=10000):
    """Load first N samples from scored datasets."""
    samples = []
    
    # Load from openorca (largest dataset)
    try:
        with open('data/phase1/scored/openorca.jsonl', 'r') as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                data = json.loads(line)
                text = f"{data.get('instruction', '')} {data.get('response', '')}".strip()
                samples.append(text)
    except:
        print("⚠️  Using synthetic test data (real data not found)")
        samples = [f"Sample text number {i} with some content" for i in range(n)]
    
    return samples

print("🧪 Testing Vectorized Deduplication Optimization\n")
print("=" * 60)

# Test with 10K samples
print("\n📊 Testing with 10,000 samples...")
texts = load_test_data(10000)
print(f"✓ Loaded {len(texts)} texts")

config = DeduplicationConfig(
    similarity_threshold=0.8,
    num_workers=10
)
deduplicator = ParallelDataDeduplicator(config)

print("\n⏱️  Running deduplication...")
start = time.time()
unique_texts, stats = deduplicator.deduplicate_texts(texts)
elapsed = time.time() - start

print(f"\n✅ RESULTS:")
print(f"   Original: {stats['original_count']:,} samples")
print(f"   Unique: {stats['final_count']:,} samples")
print(f"   Exact duplicates: {stats['exact_duplicates_removed']:,}")
print(f"   Near duplicates: {stats['near_duplicates_removed']:,}")
print(f"   Retention: {stats['retention_rate']:.1%}")
print(f"   Time: {elapsed:.1f}s")
print(f"   Throughput: {stats['original_count']/elapsed:.0f} samples/sec")

# Estimate full dataset time
full_dataset_size = 637005
estimated_time = (full_dataset_size / stats['original_count']) * elapsed
print(f"\n🎯 Estimated time for {full_dataset_size:,} samples: {estimated_time/60:.1f} minutes")

if estimated_time < 600:  # Less than 10 minutes
    print("   ✅ OPTIMIZATION SUCCESSFUL! (target: <10 minutes)")
else:
    print("   ⚠️  Still slower than target")
