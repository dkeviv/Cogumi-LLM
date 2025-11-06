#!/usr/bin/env python3
"""
Phase 1B Step 4: Cluster Failures Using Sentence-BERT + KMeans

Takes the failures from Step 3 (step3_judge_outputs.py) and clusters them
into groups with similar characteristics using semantic embeddings.

Usage:
    python "Phase 1B_2_0/phase1b_cluster_failures.py" \\
        --failures ./data/phase1b/failures.jsonl \\
        --output ./data/phase1b/clusters.json \\
        --num_clusters 10

Expected Output:
    - clusters.json: Cluster assignments with sample failures
    - embeddings.npy: Failure embeddings (for visualization)
"""

import argparse
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from tqdm import tqdm
from typing import List, Dict

def load_failures(failures_path: str) -> List[Dict]:
    """Load failures from JSONL file."""
    failures = []
    with open(failures_path, 'r', encoding='utf-8') as f:
        for line in f:
            failures.append(json.loads(line))
    return failures


def embed_failures(failures: List[Dict], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """
    Embed failures using Sentence-BERT.
    
    Combines instruction + model_output for semantic representation.
    """
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    print("Creating embeddings...")
    texts = []
    for f in failures:
        # Combine instruction and model output for richer representation
        text = f"{f['instruction']}\n{f['model_output']}"
        texts.append(text)
    
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32
    )
    
    print(f"✅ Created {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    return embeddings


def cluster_embeddings(embeddings: np.ndarray, num_clusters: int):
    """Cluster embeddings using KMeans. Returns (labels, kmeans_model)."""
    print(f"Clustering into {num_clusters} groups...")
    
    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=42,
        n_init=10,
        max_iter=300
    )
    
    labels = kmeans.fit_predict(embeddings)
    
    print(f"✅ Clustering complete")
    print(f"   Inertia: {kmeans.inertia_:.2f}")
    
    return labels, kmeans


def create_cluster_summary(failures: List[Dict], labels: np.ndarray, num_samples: int = 10) -> Dict:
    """Create summary of each cluster with sample failures."""
    num_clusters = len(set(labels))
    
    clusters = {}
    for cluster_id in range(num_clusters):
        # Get failures in this cluster
        cluster_failures = [
            failures[i] for i, label in enumerate(labels) if label == cluster_id
        ]
        
        # Sample failures for display
        sample_failures = cluster_failures[:num_samples]
        
        # Calculate average score
        avg_score = sum(f['score'] for f in cluster_failures) / len(cluster_failures)
        
        clusters[f"cluster_{cluster_id}"] = {
            "id": cluster_id,
            "size": len(cluster_failures),
            "avg_score": avg_score,
            "samples": [
                {
                    "instruction": f['instruction'][:200] + "..." if len(f['instruction']) > 200 else f['instruction'],
                    "model_output": f['model_output'][:200] + "..." if len(f['model_output']) > 200 else f['model_output'],
                    "reference": f['reference'][:200] + "..." if len(f['reference']) > 200 else f['reference'],
                    "score": f['score'],
                    "reason": f['reason']
                }
                for f in sample_failures
            ]
        }
    
    return clusters


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1B: Cluster failures using Sentence-BERT + KMeans"
    )
    
    parser.add_argument(
        "--failures",
        type=str,
        required=True,
        help="Path to failures.jsonl from Step 3 (step3_judge_outputs.py)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save clusters.json"
    )
    
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=10,
        help="Number of clusters (default: 10)"
    )
    
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-BERT model to use (default: all-MiniLM-L6-v2)"
    )
    
    parser.add_argument(
        "--save_embeddings",
        action="store_true",
        help="Save embeddings to .npy file for visualization"
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("📊 PHASE 1B: FAILURE CLUSTERING")
    print("=" * 80)
    print(f"Failures: {args.failures}")
    print(f"Output: {args.output}")
    print(f"Clusters: {args.num_clusters}")
    print(f"Embedding model: {args.embedding_model}")
    print("=" * 80)
    print()
    
    # Load failures
    print("Loading failures...")
    failures = load_failures(args.failures)
    print(f"✅ Loaded {len(failures)} failures")
    print()
    
    # Embed failures
    embeddings = embed_failures(failures, args.embedding_model)
    
    # Save embeddings if requested
    if args.save_embeddings:
        embeddings_path = output_path.parent / "embeddings.npy"
        np.save(embeddings_path, embeddings)
        print(f"✅ Embeddings saved to: {embeddings_path}")
    
    print()
    
    # Cluster
    labels, kmeans = cluster_embeddings(embeddings, args.num_clusters)
    
    print()
    
    # Create cluster summary
    print("Creating cluster summary...")
    clusters = create_cluster_summary(failures, labels)
    
    # Save clusters
    output_data = {
        "num_clusters": args.num_clusters,
        "total_failures": len(failures),
        "clusters": clusters
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Clusters saved to: {args.output}")
    print()
    
    # Print cluster summary
    print("=" * 80)
    print("📊 CLUSTER SUMMARY")
    print("=" * 80)
    
    for cluster_name, cluster_data in sorted(clusters.items(), key=lambda x: x[1]['size'], reverse=True):
        print(f"\n{cluster_name}:")
        print(f"  Size: {cluster_data['size']} failures")
        print(f"  Avg Score: {cluster_data['avg_score']:.2f}/10")
        print(f"  Sample instruction: {cluster_data['samples'][0]['instruction'][:100]}...")
    
    print()
    print("=" * 80)
    print("🎯 NEXT STEP")
    print("=" * 80)
    print("Label clusters with Llama-405B:")
    print(f"  python scripts/phase1b_label_patterns.py \\")
    print(f"      --clusters {args.output} \\")
    print(f"      --output ./data/phase1b/patterns.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
