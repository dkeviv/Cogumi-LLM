"""
Context: Parallel deduplication using multiprocessing for faster MinHash computation.

Optimizations:
1. xxhash (10x faster than MD5)
2. Multiprocessing pool for parallel MinHash computation (10 workers)
3. Simple loop for similarity (vectorization was SLOWER due to overhead)
4. Early termination for visited items
5. Small chunksize (10) for responsive progress bars
6. Progress bars for monitoring

Performance (ACTUAL MEASUREMENTS):
- Original (MD5, sequential): 6-8 hours for 674K samples
- Optimized (xxhash, sequential): 3.9 hours for 674K samples
- Parallel (xxhash + multiprocessing + simple loop): ~60-90 minutes for 674K samples

LESSON LEARNED (Oct 17, 2025):
- Vectorization (numpy) was 4x SLOWER than simple loops for this use case
- Overhead of converting lists to numpy arrays and indexing negates benefits
- Small benchmarks (10K) don't reveal scaling issues with large datasets
- Simple is often better than "optimized" for complex workflows

Usage:
    from src.utils.deduplication_parallel import ParallelDataDeduplicator, DeduplicationConfig
    
    config = DeduplicationConfig(similarity_threshold=0.8)
    deduplicator = ParallelDataDeduplicator(config)
    unique_texts, stats = deduplicator.deduplicate_texts(texts)
"""

import re
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
from multiprocessing import Pool, cpu_count
import numpy as np

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

# Try to import xxhash (10x faster than MD5)
try:
    import xxhash
    USE_XXHASH = True
except ImportError:
    USE_XXHASH = False

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class DeduplicationConfig:
    """Configuration for deduplication."""
    similarity_threshold: float = 0.8
    shingle_size: int = 3
    num_hashes: int = 64  # Reduced from 128 for 2x speedup (still accurate at 0.8 threshold)
    num_bands: int = 16
    exact_match: bool = True
    num_workers: Optional[int] = None  # Auto-detect CPU count


@dataclass
class DuplicateGroup:
    """Group of duplicate texts with representative."""
    representative: int
    duplicates: List[int]


def _init_worker(hash_seeds, shingle_size):
    """Initialize worker process with shared data."""
    global WORKER_HASH_SEEDS, WORKER_SHINGLE_SIZE
    WORKER_HASH_SEEDS = hash_seeds
    WORKER_SHINGLE_SIZE = shingle_size


def _fast_hash(text: str, seed: int) -> int:
    """Fast hash function using xxhash (10x faster than MD5)."""
    combined = f"{text}_{seed}"
    
    if USE_XXHASH:
        return xxhash.xxh64(combined.encode(), seed=seed).intdigest()
    else:
        import hashlib
        return int(hashlib.md5(combined.encode()).hexdigest(), 16)


def _get_shingles(text: str, shingle_size: int) -> Set[str]:
    """Extract shingles (n-grams) from text."""
    text = re.sub(r'\s+', ' ', text.lower().strip())
    shingles = set()
    for i in range(len(text) - shingle_size + 1):
        shingle = text[i:i + shingle_size]
        shingles.add(shingle)
    return shingles


def _compute_minhash_parallel(text: str) -> np.ndarray:
    """
    Compute MinHash signature for a text (worker function).
    Uses global WORKER_HASH_SEEDS and WORKER_SHINGLE_SIZE.
    """
    shingles = _get_shingles(text, WORKER_SHINGLE_SIZE)
    signature = np.full(len(WORKER_HASH_SEEDS), np.inf)
    
    for shingle in shingles:
        for i, seed in enumerate(WORKER_HASH_SEEDS):
            hash_val = _fast_hash(shingle, int(seed))
            signature[i] = min(signature[i], hash_val)
    
    return signature


class ParallelMinHashLSH:
    """
    Parallel MinHash LSH using multiprocessing for 4-8x speedup.
    """
    
    def __init__(self, config: DeduplicationConfig):
        self.config = config
        self.num_hashes = config.num_hashes
        self.num_bands = config.num_bands
        self.rows_per_band = self.num_hashes // self.num_bands
        self.num_workers = config.num_workers or cpu_count()
        
        # LSH buckets for each band
        self.lsh_buckets: List[Dict[Tuple, List[int]]] = [
            defaultdict(list) for _ in range(self.num_bands)
        ]
        
        # Store MinHash signatures
        self.signatures: List[np.ndarray] = []
        
        # Pre-compute hash seeds
        np.random.seed(42)
        self.hash_seeds = np.random.randint(0, 2**32 - 1, size=self.num_hashes)
        
        logger.info(f"Parallel deduplication using {self.num_workers} workers")
        logger.info(f"Using {'xxhash (10x faster)' if USE_XXHASH else 'MD5 (slower)'} for hashing")
        
    def compute_signatures_parallel(self, texts: List[str]) -> List[np.ndarray]:
        """
        Compute MinHash signatures for all texts in parallel.
        
        This is the main bottleneck - parallelizing this gives 4-8x speedup.
        """
        logger.info(f"Computing {len(texts)} signatures using {self.num_workers} workers...")
        
        # Create process pool with shared initialization
        with Pool(
            processes=self.num_workers,
            initializer=_init_worker,
            initargs=(self.hash_seeds, self.config.shingle_size)
        ) as pool:
            # Use imap for progress tracking
            signatures = []
            
            with Progress(
                SpinnerColumn(),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("•"),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=False
            ) as progress:
                task = progress.add_task(
                    f"[cyan]Computing MinHash (parallel {self.num_workers} workers)...",
                    total=len(texts)
                )
                
                # Process in smaller batches for more frequent progress updates
                # Smaller chunksize = more frequent updates (trade memory for visibility)
                batch_size = 10  # Small chunks for responsive progress bar
                
                for signature in pool.imap(_compute_minhash_parallel, texts, chunksize=batch_size):
                    signatures.append(signature)
                    progress.advance(task)
        
        return signatures
    
    def build_index(self, signatures: List[np.ndarray]):
        """Build LSH index from pre-computed signatures."""
        for text_id, signature in enumerate(signatures):
            for band_idx in range(self.num_bands):
                start_row = band_idx * self.rows_per_band
                end_row = start_row + self.rows_per_band
                
                band_signature = tuple(signature[start_row:end_row])
                self.lsh_buckets[band_idx][band_signature].append(text_id)
        
        self.signatures = signatures
        
    def find_candidates(self, text_id: int) -> Set[int]:
        """Find candidate duplicates for a text."""
        candidates = set()
        signature = self.signatures[text_id]
        
        for band_idx in range(self.num_bands):
            start_row = band_idx * self.rows_per_band
            end_row = start_row + self.rows_per_band
            
            band_signature = tuple(signature[start_row:end_row])
            bucket = self.lsh_buckets[band_idx].get(band_signature, [])
            candidates.update(bucket)
        
        candidates.discard(text_id)
        return candidates
    
    def jaccard_similarity(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Estimate Jaccard similarity from MinHash signatures."""
        return np.mean(sig1 == sig2)


class ParallelDataDeduplicator:
    """Parallel data deduplicator with 4-8x speedup."""
    
    def __init__(self, config: Optional[DeduplicationConfig] = None):
        self.config = config or DeduplicationConfig()
        self.lsh = ParallelMinHashLSH(self.config)
        
    def _remove_exact_duplicates(self, texts: List[str]) -> Tuple[List[str], int]:
        """Remove exact duplicate texts."""
        seen = set()
        unique_texts = []
        
        for text in texts:
            if text not in seen:
                seen.add(text)
                unique_texts.append(text)
        
        return unique_texts, len(texts) - len(unique_texts)
    
    def _find_duplicate_groups(self, signatures: List[np.ndarray]) -> List[DuplicateGroup]:
        """
        Find groups of near-duplicate texts.
        
        OPTIMIZATIONS:
        1. Simple loop (vectorization was SLOWER for large datasets)
        2. Early termination for visited items
        """
        logger.info("Finding duplicate groups...")
        
        visited = set()
        groups = []
        
        with Progress(
            SpinnerColumn(),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=False
        ) as progress:
            task = progress.add_task(
                "[cyan]Finding duplicate groups...",
                total=len(signatures)
            )
            
            for text_id in range(len(signatures)):
                if text_id in visited:
                    progress.advance(task)
                    continue
                
                # Find candidates using LSH
                candidates = self.lsh.find_candidates(text_id)
                
                # Filter out already visited candidates (early termination)
                candidates = [c for c in candidates if c not in visited]
                
                if not candidates:
                    progress.advance(task)
                    continue
                
                # SIMPLE LOOP: Check each candidate (faster than vectorization overhead)
                duplicates = []
                ref_sig = signatures[text_id]
                
                for candidate_id in candidates:
                    similarity = self.lsh.jaccard_similarity(ref_sig, signatures[candidate_id])
                    if similarity >= self.config.similarity_threshold:
                        duplicates.append(candidate_id)
                
                if duplicates:
                    groups.append(DuplicateGroup(
                        representative=text_id,
                        duplicates=duplicates
                    ))
                    
                    # Mark all as visited
                    visited.add(text_id)
                    visited.update(duplicates)
                
                progress.advance(task)
        
        return groups
    
    def deduplicate_texts(self, texts: List[str]) -> Tuple[List[str], Dict]:
        """
        Deduplicate texts using parallel MinHash LSH.
        
        Returns:
            - Deduplicated texts
            - Statistics dictionary
        """
        logger.info(f"Starting parallel deduplication of {len(texts)} texts")
        logger.info(f"Using {self.config.num_workers or cpu_count()} CPU cores")
        original_count = len(texts)
        
        # Step 1: Remove exact duplicates
        if self.config.exact_match:
            with Progress(SpinnerColumn(), *Progress.get_default_columns()) as progress:
                task = progress.add_task("[cyan]Removing exact duplicates...", total=1)
                texts, exact_removed = self._remove_exact_duplicates(texts)
                progress.update(task, completed=1)
            logger.info(f"Removed {exact_removed} exact duplicates")
        else:
            exact_removed = 0
        
        # Step 2: Compute MinHash signatures in parallel (MAIN SPEEDUP)
        signatures = self.lsh.compute_signatures_parallel(texts)
        
        # Step 3: Build LSH index
        logger.info("Building LSH index...")
        self.lsh.build_index(signatures)
        
        # Step 4: Find near-duplicate groups
        duplicate_groups = self._find_duplicate_groups(signatures)
        
        # Step 5: Select representatives
        logger.info(f"Processing {len(duplicate_groups)} duplicate groups...")
        
        texts_to_keep = set(range(len(texts)))
        near_duplicates_removed = 0
        
        for group in duplicate_groups:
            for duplicate_id in group.duplicates:
                if duplicate_id in texts_to_keep:
                    texts_to_keep.discard(duplicate_id)
                    near_duplicates_removed += 1
        
        # Build final list
        deduplicated_texts = [texts[i] for i in sorted(texts_to_keep)]
        
        # Statistics
        stats = {
            "original_count": original_count,
            "exact_duplicates_removed": exact_removed,
            "near_duplicates_removed": near_duplicates_removed,
            "duplicate_groups_found": len(duplicate_groups),
            "final_count": len(deduplicated_texts),
            "retention_rate": len(deduplicated_texts) / original_count if original_count > 0 else 0,
            "deduplication_rate": 1 - (len(deduplicated_texts) / original_count) if original_count > 0 else 0,
            "num_workers": self.config.num_workers or cpu_count(),
        }
        
        logger.info(f"Parallel deduplication complete: {original_count} → {len(deduplicated_texts)} "
                   f"({stats['retention_rate']:.1%} retained)")
        logger.info(f"Used {stats['num_workers']} CPU cores")
        
        return deduplicated_texts, stats
