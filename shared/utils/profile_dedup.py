"""
Profile deduplication to find bottleneck
"""
import cProfile
import pstats
import io
from pathlib import Path
import json
from deduplication_parallel import ParallelMinHashLSH

def profile_dedup():
    """Profile deduplication on sample data"""
    # Load some sample data
    data_path = Path("data/phase1/scored/openorca_scored.jsonl")
    
    samples = []
    with open(data_path) as f:
        for i, line in enumerate(f):
            if i >= 50000:  # Use 50K samples for profiling
                break
            data = json.loads(line)
            text = f"{data.get('instruction', '')} {data.get('input', '')} {data.get('output', '')}"
            samples.append(text.strip())
    
    print(f"Profiling with {len(samples)} samples...")
    
    # Profile the deduplication
    profiler = cProfile.Profile()
    profiler.enable()
    
    lsh = ParallelMinHashLSH(threshold=0.8, num_perm=128)
    signatures = lsh.compute_signatures_parallel(samples)
    duplicate_groups = lsh._find_duplicate_groups(signatures)
    
    profiler.disable()
    
    # Print results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions
    print(s.getvalue())
    
    print(f"\nFound {len(duplicate_groups)} duplicate groups")

if __name__ == "__main__":
    profile_dedup()
