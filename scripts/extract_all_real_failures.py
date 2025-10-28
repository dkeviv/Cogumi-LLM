#!/usr/bin/env python3
"""
Extract ALL Real Failures for Phase 1B.2 Training

PURPOSE:
    Extract ALL failure examples from benchmark datasets. Uses dataset's ground
    truth answers - NO GPT-4/5 distillation needed! Training-ready output.

KEY INSIGHT:
    Datasets already have correct answers! No need for expensive distillation.
    - GSM8K: Has full solution with answer ✅
    - HumanEval/MBPP: Has canonical solution ✅  
    - MMLU: Has correct choice ✅

OUTPUT FORMAT:
    {"instruction": prompt, "output": correct_solution}
    Ready for immediate training!

PIPELINE STAGE: Phase 1B.2
"""

import json
import torch
from pathlib import Path
from typing import List, Dict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re
import sys


class FailureExtractor:
    """Extract ALL failures and format as training-ready data."""
    
    def __init__(self, model_path: str, output_dir: str):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("📥 Loading Phase 1A model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("✅ Model loaded")
    
    def generate_response(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate model response."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        return response
    
    def extract_answer(self, response: str) -> str:
        """Extract final answer from response."""
        if "####" in response:
            return response.split("####")[-1].strip()
        if "answer is" in response.lower():
            match = re.search(r'answer is[:\s]+([^\n\.]+)', response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        if "\\boxed{" in response:
            match = re.search(r'\\boxed\{([^}]+)\}', response)
            if match:
                return match.group(1).strip()
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        return lines[-1] if lines else ""
    
    def is_failure(self, model_answer: str, correct_answer: str) -> bool:
        """Check if model's answer is incorrect."""
        model_ans = str(model_answer).lower().strip()
        correct_ans = str(correct_answer).lower().strip()
        for char in ['$', ',', ' ']:
            model_ans = model_ans.replace(char, '')
            correct_ans = correct_ans.replace(char, '')
        return model_ans != correct_ans
    
    def extract_math_failures(self) -> List[Dict]:
        """Extract ALL math failures from GSM8K train set."""
        print("\n" + "="*80)
        print("📊 EXTRACTING ALL MATH FAILURES FROM GSM8K")
        print("="*80)
        
        print("Loading GSM8K train dataset...")
        dataset = load_dataset("gsm8k", "main", split="train")
        total = 7473  # GSM8K train size
        print(f"✅ Loaded {total} problems")
        
        failures = []
        successes = 0
        
        for i, example in enumerate(tqdm(dataset, desc="Testing GSM8K", total=total)):
            question = example['question']
            answer = example['answer']
            correct_answer = answer.split("####")[-1].strip() if "####" in answer else answer
            
            prompt = f"Solve this math problem step by step:\n\n{question}"
            model_response = self.generate_response(prompt, max_new_tokens=512)
            model_answer = self.extract_answer(model_response)
            
            if self.is_failure(model_answer, correct_answer):
                failures.append({
                    'instruction': prompt,
                    'output': answer  # Full solution from dataset
                })
            else:
                successes += 1
            
            if (i + 1) % 500 == 0:
                print(f"   {i+1}/{total} | Failures: {len(failures)} | Successes: {successes}")
        
        print(f"\n✅ Extracted {len(failures)} failures ({len(failures)/total*100:.1f}%)")
        print(f"   Phase 1A accuracy: {successes/total*100:.1f}%")
        return failures
    
    def save_training_data(self, failures: List[Dict], category: str):
        """Save training-ready data."""
        output_file = self.output_dir / f"{category}_training_ready.jsonl"
        
        with open(output_file, 'w') as f:
            for failure in failures:
                f.write(json.dumps(failure) + '\n')
        
        print(f"💾 Saved {len(failures)} examples to: {output_file}")
        return output_file


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints/phase1a_merged')
    parser.add_argument('--output_dir', type=str, default='data/phase1b2')
    parser.add_argument('--categories', nargs='+', default=['math'],
                       help='Categories: math, code, creativity')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🔍 PHASE 1B.2: EXTRACTING ALL REAL FAILURES")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Categories: {', '.join(args.categories)}")
    print(f"Output: {args.output_dir}")
    print("="*80)
    
    extractor = FailureExtractor(args.model_path, args.output_dir)
    
    results = {}
    
    if 'math' in args.categories:
        math_failures = extractor.extract_math_failures()
        extractor.save_training_data(math_failures, 'math')
        results['math'] = len(math_failures)
    
    # TODO: Add code and creativity extraction (similar pattern)
    
    print("\n" + "="*80)
    print("✅ EXTRACTION COMPLETE")
    print("="*80)
    for category, count in results.items():
        print(f"  {category.upper()}: {count} training examples")
    print()
    print("NEXT STEP:")
    print("  Train Phase 1B.2 directly on extracted failures:")
    print(f"  python train_phase1b_benchmark.py \\")
    print(f"    --model_name checkpoints/phase1a_merged \\")
    print(f"    --dataset_path '{args.output_dir}/*_training_ready.jsonl' \\")
    print(f"    --output_dir checkpoints/phase1b2 \\")
    print(f"    --num_train_epochs 2 \\")
    print(f"    --learning_rate 3e-6")
    print("="*80)


if __name__ == "__main__":
    main()
