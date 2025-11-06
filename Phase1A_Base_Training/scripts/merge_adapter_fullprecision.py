#!/usr/bin/env python3
"""
Merge Phase 1A Adapter (Full Precision) with Base Model

Properly merges LoRA adapter trained on full precision base.
No 4-bit quantization issues!

Input: 
- Base: meta-llama/Meta-Llama-3.1-8B-Instruct (bfloat16)
- Adapter: data/checkpoints/phase1a_fullprecision (trained on full precision)

Output:
- Merged model: checkpoints/phase1a_merged_fullprecision (bfloat16, ~16GB)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    adapter_path = "data/checkpoints/phase1a_fullprecision"
    output_path = "checkpoints/phase1a_merged_fullprecision"
    
    print("="*80)
    print("🔧 MERGING PHASE 1A ADAPTER (FULL PRECISION)")
    print("="*80)
    print(f"Base: {base_model_name}")
    print(f"Adapter: {adapter_path}")
    print(f"Output: {output_path}")
    print("="*80)
    print()
    
    print("📥 Loading base model (bfloat16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    print("✅ Base model loaded")
    
    print("📥 Loading Phase 1A adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    print("✅ Adapter loaded")
    
    print("🔄 Merging adapter with base...")
    merged_model = model.merge_and_unload()
    print("✅ Merge complete (no 4-bit issues!)")
    
    print("💾 Saving merged model...")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"✅ Merged model saved to: {output_path}")
    
    print()
    print("="*80)
    print("✅ MERGE SUCCESSFUL")
    print("="*80)
    print()
    print("Model size: ~16GB (bfloat16)")
    print()
    print("NEXT STEPS:")
    print("1. Benchmark merged model to verify quality")
    print("2. Train Phase 1B on this merged model")
    print("3. Optionally quantize for deployment (AFTER merge)")
    print("="*80)

if __name__ == "__main__":
    main()
