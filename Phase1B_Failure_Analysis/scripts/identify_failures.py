#!/usr/bin/env python3
"""
Identify Model Failures for Targeted Training

Runs model on validation sets to identify:
1. Problems where model gets wrong answer
2. Problems where consistency <30%
3. Problems with high response variance

Outputs failure examples for Phase 1B.2 training.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple
import re
from collections import Counter

class FailureIdentifier:
    def __init__(self, model_path: str, output_dir: str):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            "unsloth/meta-llama-3.1-8b-instruct-bnb-4bit",
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True
        )
        
        # Load LoRA adapter if exists
        try:
            self.model = PeftModel.from_pretrained(self.model, model_path)
            print("✅ Loaded LoRA adapter")
        except:
            print("⚠️  No LoRA adapter found, using base model")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_solution(self, prompt: str, temp: float = 0.0, n_samples: int = 5) -> List[str]:
        """Generate n_samples solutions for consistency check."""
        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        outputs = []
        for _ in range(n_samples):
            output = self.model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=temp if temp > 0 else 0.01,  # Avoid zero division
                do_sample=(temp > 0),
                pad_token_id=self.tokenizer.eos_token_id
            )
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            # Extract only the assistant's response
            if "<|start_header_id|>assistant<|end_header_id|>" in response:
                response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
            outputs.append(response)
        
        return outputs
    
    def extract_answer(self, text: str, category: str) -> str:
        """Extract final answer from response."""
        if category == 'math':
            # Look for boxed answer or final number
            boxed = re.findall(r'\\boxed{([^}]+)}', text)
            if boxed:
                return boxed[-1].strip()
            
            # Look for "The answer is X" or "= X"
            patterns = [
                r'[Tt]he answer is:?\s*\$?([0-9,\.]+)',
                r'=\s*\$?([0-9,\.]+)\s*$',
                r'####\s*([0-9,\.]+)'
            ]
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    return match.group(1).strip()
        
        elif category == 'code':
            # Extract code block
            code_blocks = re.findall(r'```python\n(.*?)```', text, re.DOTALL)
            if code_blocks:
                return code_blocks[0].strip()
        
        return text.strip()
    
    def check_consistency(self, responses: List[str], category: str) -> Tuple[float, int]:
        """
        Calculate consistency percentage and number of unique responses.
        
        Returns:
            (consistency_percentage, num_unique_responses)
        """
        answers = [self.extract_answer(r, category) for r in responses]
        answer_counts = Counter(answers)
        most_common_count = answer_counts.most_common(1)[0][1] if answer_counts else 0
        
        consistency = (most_common_count / len(responses)) * 100
        num_unique = len(set(answers))
        
        return consistency, num_unique
    
    def is_answer_correct(self, response: str, ground_truth: str, category: str) -> bool:
        """Check if model's answer matches ground truth."""
        model_answer = self.extract_answer(response, category)
        
        if category == 'math':
            # Normalize numbers (remove commas, convert to float)
            try:
                model_num = float(model_answer.replace(',', ''))
                truth_num = float(ground_truth.replace(',', ''))
                return abs(model_num - truth_num) < 0.01
            except:
                return model_answer.lower() == ground_truth.lower()
        
        elif category == 'code':
            # For code, just check if solution was generated
            return len(model_answer) > 10
        
        return False
    
    def identify_math_failures(self, num_problems: int = 7473) -> List[Dict]:
        """Identify failures on GSM8K training set."""
        print("\n" + "="*80)
        print("IDENTIFYING MATH FAILURES (GSM8K Train)")
        print("="*80)
        
        dataset = load_dataset("gsm8k", "main", split="train")
        
        failures = []
        stats = {
            'total': 0,
            'incorrect': 0,
            'low_consistency': 0,
            'both': 0
        }
        
        # Sample evenly across difficulty if dataset is large
        if len(dataset) > num_problems:
            step = len(dataset) // num_problems
            indices = list(range(0, len(dataset), step))[:num_problems]
            dataset = dataset.select(indices)
        
        for example in tqdm(dataset, desc="Testing MATH problems"):
            stats['total'] += 1
            problem = example['question']
            ground_truth = example['answer'].split('####')[-1].strip()
            
            # Generate solutions (5 samples for consistency check)
            responses = self.generate_solution(problem, temp=0.7, n_samples=5)
            
            # Check consistency
            consistency, num_unique = self.check_consistency(responses, 'math')
            
            # Check correctness (use first response)
            is_correct = self.is_answer_correct(responses[0], ground_truth, 'math')
            
            # Identify as failure if:
            # 1. Answer is incorrect, OR
            # 2. Consistency <30%
            is_failure = (not is_correct) or (consistency < 30)
            
            if not is_correct:
                stats['incorrect'] += 1
            if consistency < 30:
                stats['low_consistency'] += 1
            if is_failure:
                if not is_correct and consistency < 30:
                    stats['both'] += 1
                
                failures.append({
                    'instruction': problem,
                    'ground_truth': ground_truth,
                    'category': 'math',
                    'consistency': consistency,
                    'num_unique': num_unique,
                    'is_correct': is_correct,
                    'sample_response': responses[0]
                })
        
        print(f"\n📊 MATH Failure Statistics:")
        print(f"   Total tested: {stats['total']}")
        print(f"   Incorrect answers: {stats['incorrect']} ({stats['incorrect']/stats['total']*100:.1f}%)")
        print(f"   Low consistency (<30%): {stats['low_consistency']} ({stats['low_consistency']/stats['total']*100:.1f}%)")
        print(f"   Both issues: {stats['both']}")
        print(f"   Total failures: {len(failures)} ({len(failures)/stats['total']*100:.1f}%)")
        
        return failures
    
    def identify_code_failures(self, num_problems: int = 374) -> List[Dict]:
        """Identify failures on MBPP training set."""
        print("\n" + "="*80)
        print("IDENTIFYING CODE FAILURES (MBPP Train)")
        print("="*80)
        
        dataset = load_dataset("mbpp", split="train")
        
        failures = []
        stats = {
            'total': 0,
            'no_solution': 0,
            'low_consistency': 0,
            'both': 0
        }
        
        # Use all MBPP problems (374 total)
        if len(dataset) > num_problems:
            dataset = dataset.select(range(num_problems))
        
        for example in tqdm(dataset, desc="Testing CODE problems"):
            stats['total'] += 1
            problem = example['text']
            
            # Generate solutions (5 samples for consistency check)
            prompt = f"Write a Python function to solve this problem:\n\n{problem}\n\nProvide only the function implementation."
            responses = self.generate_solution(prompt, temp=0.7, n_samples=5)
            
            # Check consistency
            consistency, num_unique = self.check_consistency(responses, 'code')
            
            # Check if solution was generated
            has_solution = len(self.extract_answer(responses[0], 'code')) > 10
            
            # Identify as failure if:
            # 1. No solution generated, OR
            # 2. Consistency <30%
            is_failure = (not has_solution) or (consistency < 30)
            
            if not has_solution:
                stats['no_solution'] += 1
            if consistency < 30:
                stats['low_consistency'] += 1
            if is_failure:
                if not has_solution and consistency < 30:
                    stats['both'] += 1
                
                failures.append({
                    'instruction': prompt,
                    'category': 'code',
                    'consistency': consistency,
                    'num_unique': num_unique,
                    'has_solution': has_solution,
                    'sample_response': responses[0]
                })
        
        print(f"\n📊 CODE Failure Statistics:")
        print(f"   Total tested: {stats['total']}")
        print(f"   No solution generated: {stats['no_solution']} ({stats['no_solution']/stats['total']*100:.1f}%)")
        print(f"   Low consistency (<30%): {stats['low_consistency']} ({stats['low_consistency']/stats['total']*100:.1f}%)")
        print(f"   Both issues: {stats['both']}")
        print(f"   Total failures: {len(failures)} ({len(failures)/stats['total']*100:.1f}%)")
        
        return failures
    
    def save_failures(self, failures: List[Dict], category: str):
        """Save identified failures to JSONL."""
        output_file = self.output_dir / f"{category}_failures.jsonl"
        
        with open(output_file, 'w') as f:
            for item in failures:
                f.write(json.dumps(item) + '\n')
        
        print(f"\n✅ Saved {len(failures)} failures to {output_file}")


def main():
    MODEL_PATH = "/workspace/data/Cogumi-LLM/checkpoints/final"  # Phase 1A model
    OUTPUT_DIR = "/workspace/data/Cogumi-LLM/failures"
    
    print("="*80)
    print("FAILURE IDENTIFICATION FOR TARGETED TRAINING")
    print("="*80)
    print("\nThis will:")
    print("1. Test model on GSM8K train (~7K problems)")
    print("2. Test model on MBPP train (~374 problems)")
    print("3. Identify problems where:")
    print("   - Answer is incorrect")
    print("   - Consistency <30%")
    print("\nEstimated time: 2-3 hours")
    print("="*80)
    
    identifier = FailureIdentifier(MODEL_PATH, OUTPUT_DIR)
    
    # Identify MATH failures
    math_failures = identifier.identify_math_failures(num_problems=2000)  # Sample 2K for speed
    identifier.save_failures(math_failures, 'math')
    
    # Identify CODE failures
    code_failures = identifier.identify_code_failures(num_problems=374)
    identifier.save_failures(code_failures, 'code')
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTotal failures identified: {len(math_failures) + len(code_failures)}")
    print(f"  - MATH: {len(math_failures)}")
    print(f"  - CODE: {len(code_failures)}")
    print(f"\nNext steps:")
    print("1. Review failure files in failures/")
    print("2. Generate self-consistent training data:")
    print("   python scripts/self_consistency_distillation.py --input failures/*.jsonl")
    print("3. Train Phase 1B.2:")
    print("   python train_qlora_optimized.py --data self_distillation/failures_*.jsonl")


if __name__ == "__main__":
    main()
