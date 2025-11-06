#!/usr/bin/env python3
"""
Phase 1B Benchmark - Standalone Terminal Version
Runs benchmark in terminal, resilient to SSH/browser disconnections.

Usage on Vast.ai:
    cd /workspace/data/Cogumi-LLM
    export OPENAI_API_KEY="your-key-here"
    nohup python scripts/run_phase1b_benchmark_standalone.py > benchmark_log.txt 2>&1 &
    
    # Monitor progress:
    tail -f benchmark_log.txt
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import argparse

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.automated_gpt4_benchmark import BenchmarkSuite


def main():
    parser = argparse.ArgumentParser(description='Run Phase 1B Benchmark')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model (auto-detects if not provided)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--samples', type=int, default=50,
                        help='Samples per category (default: 50)')
    parser.add_argument('--api_key', type=str, default=None,
                        help='OpenAI API key (or set OPENAI_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ ERROR: OpenAI API key not provided!")
        print("   Set via --api_key argument or OPENAI_API_KEY env var")
        sys.exit(1)
    
    # Auto-detect model path if not provided
    if args.model_path:
        model_path = args.model_path
    else:
        print("🔍 Auto-detecting model path...")
        possible_paths = [
            "/workspace/data/Cogumi-LLM/checkpoints/final",
            "/workspace/data/Cogumi-LLM/checkpoints/checkpoint-240240",
            "/workspace/Cogumi-LLM/checkpoints/final",
            "/data/Cogumi-LLM/checkpoints/final",
            "/workspace/checkpoints/final",
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                print(f"✅ Model found at: {model_path}")
                break
        
        if not model_path:
            print("❌ ERROR: Model not found at any expected location!")
            print("   Checked paths:")
            for path in possible_paths:
                print(f"     - {path}")
            print("\n   Specify manually with --model_path argument")
            sys.exit(1)
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = str(Path(model_path).parent.parent / "benchmark_results")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("🚀 PHASE 1B BENCHMARK - STANDALONE MODE")
    print("="*70)
    print(f"\n📋 Configuration:")
    print(f"   Model: {model_path}")
    print(f"   Output: {output_dir}")
    print(f"   Samples per category: {args.samples}")
    print(f"   Total comparisons: {args.samples * 6}")
    print(f"   Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    # Initialize benchmark suite
    print("🔄 Initializing benchmark suite...")
    try:
        suite = BenchmarkSuite(
            model_path=model_path,
            openai_key=api_key,
            output_dir=output_dir,
            device="auto"
        )
        print("✅ Suite initialized!")
        print(f"\n📊 Testing {len(suite.categories)} categories:")
        for cat, desc in suite.categories.items():
            print(f"   - {cat}: {desc}")
    except Exception as e:
        print(f"❌ Failed to initialize benchmark suite: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*70)
    print("⏳ Running benchmark (this will take ~6 hours)...")
    print("   Progress will be shown below:")
    print("="*70 + "\n")
    
    # Run benchmark
    try:
        results = suite.run_full_benchmark(
            categories=list(suite.categories.keys()),
            samples_per_category=args.samples
        )
        
        print("\n" + "="*70)
        print("✅ BENCHMARK COMPLETE!")
        print("="*70)
        print(f"   End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Results saved to: {output_dir}")
        print("")
        
        # Print summary
        if results:
            print("📊 Quick Summary:")
            for category, data in results.items():
                wins = data.get('wins', 0)
                losses = data.get('losses', 0)
                ties = data.get('ties', 0)
                total = wins + losses + ties
                if total > 0:
                    score = ((wins + ties * 0.5) / total) * 100
                    print(f"   {category:12s}: {score:5.1f}% ({wins}W {losses}L {ties}T)")
        
        print("\n✅ Benchmark completed successfully!")
        print(f"   View full results at: {output_dir}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user (Ctrl+C)")
        print(f"   Partial results may be saved at: {output_dir}")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
