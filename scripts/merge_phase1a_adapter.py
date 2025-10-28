#!/usr/bin/env python3
"""
Merge Phase 1A Adapter with Base Model

PURPOSE:
    Creates a fully merged Phase 1A model by combining the base model with
    Phase 1A adapter. This merged model should be used as the starting point
    for all subsequent training (Phase 1B, 1C, etc.).

WHY THIS IS CRITICAL:
    - Phase 1A training created only an adapter (LoRA weights)
    - Subsequent training needs the FULL Phase 1A knowledge, not just base
    - Without merging, Phase 1B trains on base only (loses Phase 1A knowledge)
    - This caused catastrophic forgetting in initial Phase 1B attempt

WHEN TO RUN:
    - After Phase 1A training completes
    - Before any Phase 1B training
    - Only needs to run ONCE (creates reusable merged model)

INPUT:
    - checkpoints/final/ (Phase 1A adapter)
    - Base model downloaded from HuggingFace automatically

OUTPUT:
    - checkpoints/phase1a_merged/ (Full merged model ready for Phase 1B)

USAGE:
    python scripts/merge_phase1a_adapter.py

PIPELINE STAGE: Phase 1A post-processing (bridge to Phase 1B)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
from pathlib import Path
import argparse

def merge_adapter(
    adapter_path: str = "checkpoints/final",
    output_path: str = "checkpoints/phase1a_merged"
):
    """
    Merge Phase 1A adapter with base model.
    
    Args:
        adapter_path: Path to Phase 1A adapter checkpoint
        output_path: Path to save merged model
    """
    
    print("=" * 80)
    print("🔧 MERGING PHASE 1A ADAPTER WITH BASE MODEL")
    print("=" * 80)
    print()
    
    # Check if adapter exists
    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found at: {adapter_path}")
    
    # Read adapter config to get base model
    adapter_config_path = adapter_path / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found at: {adapter_config_path}")
    
    with open(adapter_config_path) as f:
        adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path")
    
    print(f"📥 Loading base model: {base_model_name}")
    print("   (This may take a few minutes to download from HuggingFace)")
    print()
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=False
    )
    print("✅ Base model loaded")
    print()
    
    # Load adapter
    print(f"📥 Loading Phase 1A adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    print("✅ Adapter loaded")
    print()
    
    # Merge adapter into base
    print("🔄 Merging adapter with base model...")
    print("   (This creates a single unified model)")
    merged_model = model.merge_and_unload()
    print("✅ Merge complete!")
    print()
    
    # Save merged model
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Saving merged model to: {output_path}")
    merged_model.save_pretrained(str(output_path))
    
    # Save tokenizer
    print("💾 Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(output_path))
    
    print("✅ Merged model saved!")
    print()
    
    # Get model size
    import os
    model_files = list(output_path.glob("*.safetensors")) + list(output_path.glob("*.bin"))
    total_size = sum(f.stat().st_size for f in model_files)
    size_gb = total_size / (1024**3)
    
    print("=" * 80)
    print("✅ MERGE COMPLETE!")
    print("=" * 80)
    print(f"Merged model size: {size_gb:.2f} GB")
    print(f"Location: {output_path}")
    print()
    print("NEXT STEPS:")
    print("1. Use this merged model as base for Phase 1B training:")
    print(f"   --model_name {output_path}")
    print()
    print("2. This merged model contains:")
    print("   ✅ Base Llama 3.1 8B knowledge")
    print("   ✅ Phase 1A training (600K examples)")
    print("   ✅ Ready for Phase 1B targeted training")
    print()
    print("3. Delete Phase 1B adapter and retrain if needed:")
    print("   rm -rf checkpoints/phase1b_from_benchmark")
    print("   bash scripts/run_phase1b_benchmark_training.sh")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Merge Phase 1A adapter with base model')
    parser.add_argument('--adapter_path', type=str, default='checkpoints/final',
                       help='Path to Phase 1A adapter (default: checkpoints/final)')
    parser.add_argument('--output_path', type=str, default='checkpoints/phase1a_merged',
                       help='Path to save merged model (default: checkpoints/phase1a_merged)')
    
    args = parser.parse_args()
    
    merge_adapter(
        adapter_path=args.adapter_path,
        output_path=args.output_path
    )


if __name__ == "__main__":
    main()
