#!/usr/bin/env python3
"""
Phase 1B: Label Failure Patterns Using Llama-405B

Takes clustered failures and auto-labels each cluster with a descriptive
pattern name using Llama-405B as the labeling model (FREE via HF API).

Usage:
    python scripts/phase1b_label_patterns.py \\
        --clusters ./data/phase1b/clusters.json \\
        --output ./data/phase1b/patterns.json

Expected Output:
    - patterns.json: Each cluster labeled with failure pattern description
"""

import argparse
import json
from pathlib import Path
from huggingface_hub import InferenceClient
import time
from typing import Dict

LABELING_PROMPT_TEMPLATE = """You are an AI model analyst. Analyze these failed model outputs and identify the common failure pattern.

**Cluster Info:**
- Size: {size} failures
- Average Score: {avg_score:.2f}/10

**Sample Failures:**
{samples}

**Task:** Based on these failures, identify the common pattern/weakness.

Provide:
1. **Pattern Name:** A concise label (3-5 words) describing the failure type
2. **Description:** What specific weakness or error pattern do these failures share?
3. **Examples:** What types of inputs trigger this failure?

Format your response as:
PATTERN: [short name]
DESCRIPTION: [2-3 sentence explanation]
EXAMPLES: [types of inputs that trigger this]"""


def label_cluster_with_llama(cluster_data: Dict, client: InferenceClient, judge_model: str) -> Dict:
    """
    Label a cluster using Llama-405B via HuggingFace Inference API.
    """
    # Format samples for the prompt
    samples_text = ""
    for i, sample in enumerate(cluster_data['samples'][:5], 1):  # First 5 samples
        samples_text += f"\n**Failure {i}:**\n"
        samples_text += f"Instruction: {sample['instruction']}\n"
        samples_text += f"Model Output: {sample['model_output']}\n"
        samples_text += f"Score: {sample['score']}/10\n"
        samples_text += f"Issue: {sample['reason']}\n"
    
    prompt = LABELING_PROMPT_TEMPLATE.format(
        size=cluster_data['size'],
        avg_score=cluster_data['avg_score'],
        samples=samples_text
    )
    
    try:
        # Use chat_completion instead of text_generation for better API support
        messages = [{"role": "user", "content": prompt}]
        response = client.chat_completion(
            messages=messages,
            model=judge_model,  # Explicitly specify the model
            max_tokens=512,
            temperature=0.3  # Low temperature for consistent labeling
        )
        
        # Extract text from chat completion response
        response_text = response.choices[0].message.content
        
        # Parse response
        pattern_name = ""
        description = ""
        examples = ""
        
        for line in response_text.split('\n'):
            if line.startswith('PATTERN:'):
                pattern_name = line.replace('PATTERN:', '').strip()
            elif line.startswith('DESCRIPTION:'):
                description = line.replace('DESCRIPTION:', '').strip()
            elif line.startswith('EXAMPLES:'):
                examples = line.replace('EXAMPLES:', '').strip()
        
        if not pattern_name:
            # Fallback: use first line
            pattern_name = response_text.split('\n')[0].strip()[:50]
        
        return {
            "pattern_name": pattern_name,
            "description": description or response_text[:200],
            "examples": examples or "Various inputs",
            "raw_label": response_text
        }
        
    except Exception as e:
        print(f"⚠️  Labeling error: {e}")
        return {
            "pattern_name": f"Cluster {cluster_data['id']}",
            "description": f"Unlabeled cluster ({cluster_data['size']} failures)",
            "examples": "Various inputs",
            "raw_label": str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1B: Label failure patterns using Llama-405B"
    )
    
    parser.add_argument(
        "--clusters",
        type=str,
        required=True,
        help="Path to clusters.json from phase1b_cluster_failures.py"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save labeled patterns.json"
    )
    
    parser.add_argument(
        "--judge_model",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Judge model to use (default: Llama-3.3-70B-Instruct, use 405B for highest quality but 10x slower)"
    )
    
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace API token (optional, uses cached token if not provided)"
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("🏷️  PHASE 1B: PATTERN LABELING")
    print("=" * 80)
    print(f"Clusters: {args.clusters}")
    print(f"Output: {args.output}")
    print(f"Judge: {args.judge_model} (Llama-405B via HF - FREE)")
    print("=" * 80)
    print()
    
    # Load clusters
    print("Loading clusters...")
    with open(args.clusters, 'r', encoding='utf-8') as f:
        clusters_data = json.load(f)
    
    clusters = clusters_data['clusters']
    print(f"✅ Loaded {len(clusters)} clusters")
    print()
    
    # Initialize judge
    print(f"Initializing judge model: {args.judge_model}...")
    client = InferenceClient(token=args.hf_token)
    print("✅ Judge ready (HuggingFace Inference API - Zero Cost)")
    print()
    
    # Label each cluster
    print("Labeling clusters...")
    print("=" * 80)
    
    labeled_patterns = []
    
    for cluster_name, cluster_data in clusters.items():
        print(f"\nLabeling {cluster_name} ({cluster_data['size']} failures)...")
        
        label = label_cluster_with_llama(cluster_data, client, args.judge_model)
        
        pattern = {
            "cluster_id": cluster_data['id'],
            "cluster_name": cluster_name,
            "pattern_name": label['pattern_name'],
            "description": label['description'],
            "examples": label['examples'],
            "size": cluster_data['size'],
            "avg_score": cluster_data['avg_score'],
            "sample_failures": cluster_data['samples'][:3]  # Keep 3 samples
        }
        
        labeled_patterns.append(pattern)
        
        print(f"✅ Pattern: {label['pattern_name']}")
        print(f"   {label['description'][:100]}...")
        
        # Rate limiting
        time.sleep(1)
    
    print()
    print("=" * 80)
    
    # Save labeled patterns
    output_data = {
        "total_failures": clusters_data['total_failures'],
        "num_patterns": len(labeled_patterns),
        "patterns": labeled_patterns
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Labeled patterns saved to: {args.output}")
    print()
    
    # Print summary
    print("=" * 80)
    print("📊 PATTERN SUMMARY")
    print("=" * 80)
    
    for pattern in sorted(labeled_patterns, key=lambda x: x['size'], reverse=True):
        print(f"\n{pattern['pattern_name']} ({pattern['size']} failures)")
        print(f"  {pattern['description'][:150]}...")
    
    print()
    print("=" * 80)
    print("🎯 NEXT STEP")
    print("=" * 80)
    print("Generate targeted GPT-5 examples for Phase 1C:")
    print(f"  python scripts/phase1c_generate_targeted_data.py \\")
    print(f"      --patterns {args.output} \\")
    print(f"      --output ./data/phase1c/gpt5_targeted.jsonl")
    print("=" * 80)


if __name__ == "__main__":
    main()
