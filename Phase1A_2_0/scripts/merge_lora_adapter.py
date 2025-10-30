#!/usr/bin/env python3
"""
Merge LoRA Adapter with Base Model - Phase 1A Completion

Merges the trained LoRA adapter from Phase 1A training back into the base
Llama-3.1-8B-Instruct model to create the final 10GB merged model.

Usage:
    python scripts/merge_lora_adapter.py \
        --base_model models/llama-3.1-8b-base \
        --adapter_path data/checkpoints/phase1a_final \
        --output_path models/phase1a_merged_10gb
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def merge_lora_adapter(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    device_map: str = "auto",
    torch_dtype: str = "float16"
) -> None:
    """
    Merge LoRA adapter with base model.
    
    Args:
        base_model_path: Path to base Llama-3.1-8B-Instruct model
        adapter_path: Path to trained LoRA adapter checkpoint
        output_path: Path to save merged model
        device_map: Device mapping strategy ("auto", "cpu", "cuda:0")
        torch_dtype: Torch dtype for model ("float16", "bfloat16", "float32")
    """
    logger.info("=" * 80)
    logger.info("PHASE 1A: LoRA Adapter Merge")
    logger.info("=" * 80)
    
    # Convert torch_dtype string to actual dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    torch_dtype_obj = dtype_map.get(torch_dtype, torch.float16)
    
    # Validate paths
    # Note: base_model_path can be HuggingFace model ID (e.g., "meta-llama/...") 
    # OR local path - only validate if it looks like a local path
    is_hf_model = "/" in base_model_path and not base_model_path.startswith(".")
    
    if not is_hf_model:
        base_model_path_obj = Path(base_model_path)
        if not base_model_path_obj.exists():
            raise FileNotFoundError(f"Base model not found: {base_model_path}")
    
    adapter_path_obj = Path(adapter_path)
    if not adapter_path_obj.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")
    
    output_path = Path(output_path)
    
    logger.info(f"Base model: {base_model_path}")
    logger.info(f"Adapter: {adapter_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Device: {device_map}")
    logger.info(f"Dtype: {torch_dtype}")
    logger.info("")
    
    # Step 1: Load base model
    logger.info("Step 1/5: Loading base model...")
    logger.info(f"Loading from: {'HuggingFace' if is_hf_model else 'local path'}")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,  # Use string directly (works for both HF and local)
            device_map=device_map,
            torch_dtype=torch_dtype_obj,
            trust_remote_code=True
        )
        logger.info(f"✓ Base model loaded: {base_model.num_parameters():,} parameters")
    except Exception as e:
        logger.error(f"Failed to load base model: {e}")
        raise
    
    # Step 2: Load LoRA adapter
    logger.info("Step 2/5: Loading LoRA adapter...")
    try:
        model = PeftModel.from_pretrained(
            base_model,
            str(adapter_path),
            device_map=device_map
        )
        logger.info("✓ LoRA adapter loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load LoRA adapter: {e}")
        raise
    
    # Step 3: Merge adapter into base model
    logger.info("Step 3/5: Merging LoRA weights into base model...")
    logger.info("This may take 5-10 minutes...")
    try:
        merged_model = model.merge_and_unload()
        logger.info("✓ LoRA weights merged successfully")
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("✓ CUDA cache cleared")
    except Exception as e:
        logger.error(f"Failed to merge weights: {e}")
        raise
    
    # Step 4: Load tokenizer
    logger.info("Step 4/5: Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(base_model_path),
            trust_remote_code=True
        )
        logger.info("✓ Tokenizer loaded")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise
    
    # Step 5: Save merged model
    logger.info("Step 5/5: Saving merged model...")
    logger.info(f"Output directory: {output_path}")
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        merged_model.save_pretrained(
            str(output_path),
            safe_serialization=True,  # Use safetensors format
            max_shard_size="5GB"
        )
        logger.info("✓ Model saved")
        
        # Save tokenizer
        tokenizer.save_pretrained(str(output_path))
        logger.info("✓ Tokenizer saved")
        
    except Exception as e:
        logger.error(f"Failed to save merged model: {e}")
        raise
    
    # Verify output
    logger.info("")
    logger.info("Verifying output...")
    output_files = list(output_path.glob("*"))
    logger.info(f"✓ Output contains {len(output_files)} files")
    
    # Calculate approximate size
    total_size_gb = sum(f.stat().st_size for f in output_files if f.is_file()) / (1024**3)
    logger.info(f"✓ Total size: {total_size_gb:.2f} GB")
    
    # Success summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("MERGE COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Merged model saved to: {output_path}")
    logger.info(f"Model size: {total_size_gb:.2f} GB")
    logger.info("")
    logger.info("Next Steps:")
    logger.info("1. Validate model: python scripts/validate_base_model.py")
    logger.info("2. Test inference: python scripts/test_merged_model.py")
    logger.info("3. Proceed to Phase 1B: Failure Analysis")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter with base model (Phase 1A completion)"
    )
    
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path to base Llama-3.1-8B-Instruct model"
    )
    
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to trained LoRA adapter checkpoint"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save merged model"
    )
    
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "cuda:0"],
        help="Device mapping strategy (default: auto)"
    )
    
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for model (default: float16)"
    )
    
    args = parser.parse_args()
    
    try:
        merge_lora_adapter(
            base_model_path=args.base_model,
            adapter_path=args.adapter_path,
            output_path=args.output_path,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype
        )
        sys.exit(0)
    except Exception as e:
        logger.error(f"Merge failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
