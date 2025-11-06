"""
⚠️ DEPRECATED - DO NOT USE ⚠️
=================================

This script is DEPRECATED and should NOT be used.

**Replaced by:** src/utils/deduplication_parallel.py
**Reason:** MD5 hashing too slow (6-8 hours for 674K samples)
**Migration:** Use ParallelDataDeduplicator instead of DataDeduplicator

**Performance Comparison:**
- This (MD5):           6-8 hours for 674K samples
- deduplication_optimized (xxhash): 56 minutes for 674K samples  
- deduplication_parallel (xxhash + multiprocessing): 9 minutes for 674K samples

**Archive Date:** October 17, 2025
**Archived For:** Historical reference only

=================================

Data Deduplication (LEGACY - MD5 based)
==================

This module provides advanced deduplication for training data.
Uses MinHash LSH for efficient near-duplicate detection.

Features:
- MinHash LSH for scalable similarity detection
- Content-aware deduplication
- Preserves data quality while removing redundancy
- Supports different similarity thresholds
"""

import hashlib
import re
from typing import List, Set, Dict, Tuple, Optional, Iterator
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationConfig:
    """Configuration for deduplication parameters."""
    similarity_threshold: float = 0.8  # Jaccard similarity threshold
    shingle_size: int = 3             # N-gram size for shingling
    num_hashes: int = 128             # Number of hash functions for MinHash
    num_bands: int = 16               # LSH bands for bucketing
    min_length: int = 50              # Minimum text length to consider
    exact_match: bool = True          # Whether to remove exact duplicates first


@dataclass
class DuplicateGroup:
    """Group of duplicate or near-duplicate items."""
    representative: str  # The item to keep (usually longest/highest quality)
    duplicates: List[str]  # Items to remove
    similarity_scores: List[float]  # Similarity to representative
    group_id: int


class MinHashLSH:
    """
    MinHash Locality Sensitive Hashing for efficient duplicate detection.
    
    This implementation uses MinHash signatures with LSH banding to find
    near-duplicate text efficiently in large datasets.
    """
    
    def __init__(self, config: DeduplicationConfig):
        self.config = config
        self.num_hashes = config.num_hashes
        self.num_bands = config.num_bands
        self.rows_per_band = self.num_hashes // self.num_bands
        
        # LSH buckets for each band
        self.lsh_buckets: List[Dict[Tuple, List[int]]] = [
            defaultdict(list) for _ in range(self.num_bands)
        ]
        
        # Store MinHash signatures
        self.signatures: List[np.ndarray] = []
        self.texts: List[str] = []
        
        # Pre-compute hash functions (using random seeds)
        np.random.seed(42)  # For reproducibility
        self.hash_seeds = np.random.randint(0, 2**32 - 1, size=self.num_hashes)
        
    def _get_shingles(self, text: str) -> Set[str]:
        """Extract shingles (n-grams) from text."""
        # Normalize text
        text = re.sub(r'\s+', ' ', text.lower().strip())
        
        # Create character n-grams (shingles)
        shingles = set()
        for i in range(len(text) - self.config.shingle_size + 1):
            shingle = text[i:i + self.config.shingle_size]
            shingles.add(shingle)
            
        return shingles
        
    def _compute_minhash(self, shingles: Set[str]) -> np.ndarray:
        """Compute MinHash signature for a set of shingles."""
        signature = np.full(self.num_hashes, np.inf)
        
        for shingle in shingles:
            # Hash the shingle with each hash function
            for i, seed in enumerate(self.hash_seeds):
                # Combine shingle hash with seed
                combined = f"{shingle}_{seed}"
                hash_val = int(hashlib.md5(combined.encode()).hexdigest(), 16)
                signature[i] = min(signature[i], hash_val)
                
        return signature
        
    def _add_to_lsh_buckets(self, signature: np.ndarray, text_id: int):
        """Add signature to LSH buckets."""
        for band_idx in range(self.num_bands):
            start_row = band_idx * self.rows_per_band
            end_row = start_row + self.rows_per_band
            
            # Create bucket key from signature band
            band_signature = tuple(signature[start_row:end_row])
            self.lsh_buckets[band_idx][band_signature].append(text_id)
            
    def add_text(self, text: str) -> int:
        """Add text to the LSH index."""
        if len(text) < self.config.min_length:
            return -1  # Skip short texts
            
        text_id = len(self.texts)
        self.texts.append(text)
        
        # Compute shingles and MinHash signature
        shingles = self._get_shingles(text)
        if not shingles:
            return -1
            
        signature = self._compute_minhash(shingles)
        self.signatures.append(signature)
        
        # Add to LSH buckets
        self._add_to_lsh_buckets(signature, text_id)
        
        return text_id
        
    def get_candidates(self, text_id: int) -> Set[int]:
        """Get candidate duplicates for a text using LSH."""
        if text_id >= len(self.signatures):
            return set()
            
        candidates = set()
        signature = self.signatures[text_id]
        
        # Check each LSH band
        for band_idx in range(self.num_bands):
            start_row = band_idx * self.rows_per_band
            end_row = start_row + self.rows_per_band
            
            band_signature = tuple(signature[start_row:end_row])
            
            # Get all texts in the same bucket
            bucket_candidates = self.lsh_buckets[band_idx].get(band_signature, [])
            candidates.update(bucket_candidates)
            
        # Remove self
        candidates.discard(text_id)
        return candidates
        
    def jaccard_similarity(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Compute Jaccard similarity from MinHash signatures."""
        return np.mean(sig1 == sig2)
        
    def find_duplicates(self, text_id: int) -> List[Tuple[int, float]]:
        """Find all duplicates of a given text above similarity threshold."""
        candidates = self.get_candidates(text_id)
        duplicates = []
        
        signature1 = self.signatures[text_id]
        
        for candidate_id in candidates:
            signature2 = self.signatures[candidate_id]
            similarity = self.jaccard_similarity(signature1, signature2)
            
            if similarity >= self.config.similarity_threshold:
                duplicates.append((candidate_id, similarity))
                
        return duplicates


class DataDeduplicator:
    """
    High-performance deduplication for training data.
    
    Uses MinHash LSH for efficient near-duplicate detection at scale.
    Preserves highest quality examples from duplicate groups.
    """
    
    def __init__(self, config: Optional[DeduplicationConfig] = None):
        self.config = config or DeduplicationConfig()
        self.lsh = MinHashLSH(self.config)
        self._exact_hash_map: Dict[str, int] = {}
        
    def _get_text_hash(self, text: str) -> str:
        """Get hash for exact duplicate detection."""
        normalized = re.sub(r'\s+', ' ', text.strip().lower())
        return hashlib.sha256(normalized.encode()).hexdigest()
        
    def _select_representative(self, duplicate_group: List[int]) -> int:
        """Select the best representative from a duplicate group."""
        # Quality scoring criteria (can be customized)
        def quality_score(text_id: int) -> Tuple[int, int]:
            text = self.lsh.texts[text_id]
            
            # Prefer longer texts (more information)
            length_score = len(text)
            
            # Prefer texts with better formatting (more punctuation/structure)
            structure_score = len(re.findall(r'[.!?;:]', text))
            
            return (length_score, structure_score)
            
        # Select text with highest quality score
        best_id = max(duplicate_group, key=quality_score)
        return best_id
        
    def deduplicate_texts(self, texts: List[str]) -> Tuple[List[str], Dict]:
        """
        Deduplicate a list of texts.
        
        Args:
            texts: List of text strings to deduplicate
            
        Returns:
            Tuple of (deduplicated_texts, stats_dict)
        """
        
        logger.info(f"Starting deduplication of {len(texts)} texts")
        original_count = len(texts)
        
        # Step 1: Remove exact duplicates if enabled
        if self.config.exact_match:
            texts, exact_removed = self._remove_exact_duplicates(texts)
            logger.info(f"Removed {exact_removed} exact duplicates")
        else:
            exact_removed = 0
            
        # Step 2: Build LSH index
        logger.info("Building MinHash LSH index...")
        text_ids = []
        for i, text in enumerate(texts):
            text_id = self.lsh.add_text(text)
            if text_id >= 0:  # Successfully added
                text_ids.append(text_id)
            else:
                text_ids.append(-1)  # Skipped (too short)
                
        # Step 3: Find near-duplicate groups
        logger.info("Finding near-duplicate groups...")
        duplicate_groups = self._find_duplicate_groups(text_ids)
        
        # Step 4: Select representatives and build final list
        logger.info(f"Processing {len(duplicate_groups)} duplicate groups...")
        
        # Track which texts to keep
        texts_to_keep = set(range(len(texts)))
        
        # Remove duplicates from each group (keep only representative)
        near_duplicates_removed = 0
        for group in duplicate_groups:
            representative = group.representative
            for duplicate_id in group.duplicates:
                if duplicate_id in texts_to_keep:
                    texts_to_keep.discard(duplicate_id)
                    near_duplicates_removed += 1
                    
        # Build final deduplicated list
        deduplicated_texts = [texts[i] for i in sorted(texts_to_keep)]
        
        # Statistics
        stats = {
            "original_count": original_count,
            "exact_duplicates_removed": exact_removed,
            "near_duplicates_removed": near_duplicates_removed,
            "duplicate_groups_found": len(duplicate_groups),
            "final_count": len(deduplicated_texts),
            "deduplication_ratio": len(deduplicated_texts) / original_count if original_count > 0 else 0,
            "similarity_threshold": self.config.similarity_threshold,
            "config": self.config
        }
        
        logger.info(f"Deduplication complete: {original_count} → {len(deduplicated_texts)} texts")
        logger.info(f"Removed {exact_removed + near_duplicates_removed} duplicates ({((exact_removed + near_duplicates_removed) / original_count * 100):.1f}%)")
        
        return deduplicated_texts, stats
        
    def _remove_exact_duplicates(self, texts: List[str]) -> Tuple[List[str], int]:
        """Remove exact duplicates (fast hash-based)."""
        seen_hashes = set()
        deduplicated = []
        removed_count = 0
        
        for text in texts:
            text_hash = self._get_text_hash(text)
            
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                deduplicated.append(text)
            else:
                removed_count += 1
                
        return deduplicated, removed_count
        
    def _find_duplicate_groups(self, text_ids: List[int]) -> List[DuplicateGroup]:
        """Find groups of near-duplicate texts."""
        processed = set()
        duplicate_groups = []
        group_id = 0
        
        for i, text_id in enumerate(text_ids):
            if text_id < 0 or text_id in processed:
                continue
                
            # Find all duplicates of this text
            duplicates = self.lsh.find_duplicates(text_id)
            
            if duplicates:
                # Create duplicate group
                group_members = [text_id] + [dup_id for dup_id, _ in duplicates]
                
                # Remove already processed members
                group_members = [tid for tid in group_members if tid not in processed]
                
                if len(group_members) > 1:
                    # Select representative (best quality)
                    representative = self._select_representative(group_members)
                    
                    # Create group
                    duplicates_list = [tid for tid in group_members if tid != representative]
                    similarity_scores = [
                        self.lsh.jaccard_similarity(
                            self.lsh.signatures[representative],
                            self.lsh.signatures[dup_id]
                        ) for dup_id in duplicates_list
                    ]
                    
                    group = DuplicateGroup(
                        representative=representative,
                        duplicates=duplicates_list,
                        similarity_scores=similarity_scores,
                        group_id=group_id
                    )
                    
                    duplicate_groups.append(group)
                    group_id += 1
                    
                # Mark all as processed
                processed.update(group_members)
            else:
                processed.add(text_id)
                
        return duplicate_groups
        
    def get_duplicate_statistics(self, duplicate_groups: List[DuplicateGroup]) -> Dict:
        """Get detailed statistics about duplicate groups."""
        
        if not duplicate_groups:
            return {"total_groups": 0}
            
        # Group size distribution
        group_sizes = [len(group.duplicates) + 1 for group in duplicate_groups]
        
        # Similarity distribution
        all_similarities = []
        for group in duplicate_groups:
            all_similarities.extend(group.similarity_scores)
            
        stats = {
            "total_groups": len(duplicate_groups),
            "total_duplicates_removed": sum(len(group.duplicates) for group in duplicate_groups),
            "avg_group_size": np.mean(group_sizes) if group_sizes else 0,
            "max_group_size": max(group_sizes) if group_sizes else 0,
            "avg_similarity": np.mean(all_similarities) if all_similarities else 0,
            "min_similarity": min(all_similarities) if all_similarities else 0,
            "group_size_distribution": {
                "2": sum(1 for size in group_sizes if size == 2),
                "3-5": sum(1 for size in group_sizes if 3 <= size <= 5),
                "6-10": sum(1 for size in group_sizes if 6 <= size <= 10),
                "11+": sum(1 for size in group_sizes if size > 10)
            }
        }
        
        return stats
        
    def deduplicate_dataset_streaming(
        self,
        texts_iterator: Iterator[str],
        batch_size: int = 1000
    ) -> Iterator[str]:
        """
        Deduplicate large dataset in streaming fashion.
        
        Useful for very large datasets that don't fit in memory.
        Processes data in batches and yields deduplicated results.
        """
        
        batch = []
        
        for text in texts_iterator:
            batch.append(text)
            
            if len(batch) >= batch_size:
                # Process batch
                deduplicated_batch, _ = self.deduplicate_texts(batch)
                
                # Yield results
                for deduplicated_text in deduplicated_batch:
                    yield deduplicated_text
                    
                # Reset batch
                batch = []
                
                # Reset LSH for next batch (to prevent memory growth)
                self.lsh = MinHashLSH(self.config)
                
        # Process final batch
        if batch:
            deduplicated_batch, _ = self.deduplicate_texts(batch)
            for deduplicated_text in deduplicated_batch:
                yield deduplicated_text


def deduplicate_training_data(
    texts: List[str],
    similarity_threshold: float = 0.8,
    preserve_order: bool = False
) -> Tuple[List[str], Dict]:
    """
    Convenience function for deduplicating training data.
    
    Args:
        texts: List of texts to deduplicate
        similarity_threshold: Jaccard similarity threshold (0.0 - 1.0)
        preserve_order: Whether to preserve original order
        
    Returns:
        Tuple of (deduplicated_texts, statistics)
    """
    
    config = DeduplicationConfig(similarity_threshold=similarity_threshold)
    deduplicator = DataDeduplicator(config)
    
    # Deduplicate
    deduplicated_texts, stats = deduplicator.deduplicate_texts(texts)
    
    # Preserve order if requested (requires mapping back)
    if preserve_order:
        # This is more complex and memory-intensive
        # For now, we return in the deduplicated order
        logger.warning("Order preservation not implemented yet, returning deduplicated order")
        
    return deduplicated_texts, stats