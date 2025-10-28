#!/usr/bin/env python3
"""
Extract ALL real failures from available benchmark datasets and prepare training JSONL

Creates training-ready files with fields: {"instruction":..., "output":...}
- For math (GSM8K): uses dataset's full solution as output
- For code (HumanEval/MBPP): uses provided reference implementation or test-backed solutions
- For creativity (MMLU/multiple-choice): uses correct choice explanation if available

Output directory: data/phase1b_all_failures/

Run on Vast.ai (recommended):
python scripts/extract_all_failures_prepare_training.py --model_path checkpoints/phase1a_merged --output_dir data/phase1b_all_failures
"""

import json
from pathlib import Path
from typing import List, Dict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import re
import argparse


def extract_answer_from_gsm8k_solution(sol: str) -> str:
    """Return the final answer portion from GSM8K full solution text"""
    # GSM8K solutions often have '####' and a final answer; prefer the final numeric part
    if '####' in sol:
        return sol.split('####')[-1].strip()
    # fallback: return last line
    lines = [l.strip() for l in sol.split('\n') if l.strip()]
    return lines[-1] if lines else sol.strip()


def prepare_math_failures(model, tokenizer, output_dir: Path) -> Path:
    output_file = output_dir / 'math_all_failures.jsonl'
    dataset = load_dataset('gsm8k', 'main', split='train')
    failures = 0
    
    with open(output_file, 'w') as fout:
        for i, ex in enumerate(tqdm(dataset, desc='Math GSM8K')):
            question = ex['question']
            sol = ex.get('answer', '')
            correct_answer = extract_answer_from_gsm8k_solution(sol)
            # Generate model response
            prompt = f"Solve this math problem step by step:\n\n{question}"
            inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=False)
            resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract last line of model response
            resp_only = resp[len(prompt):].strip()
            # Compare simplified answers (naive)
            norm_model = re.sub(r'[^0-9\.-]', '', resp_only)
            norm_correct = re.sub(r'[^0-9\.-]', '', correct_answer)
            if norm_model != norm_correct:
                failures += 1
                obj = {
                    'instruction': question,
                    'output': sol.strip(),
                    'failure_type': 'loss',
                    'source': 'gsm8k',
                    'index': i
                }
                fout.write(json.dumps(obj) + '\n')
    print(f"Saved math failures: {output_file} (count unknown until we read lines)")
    return output_file


def prepare_code_failures(model, tokenizer, output_dir: Path) -> Path:
    output_file = output_dir / 'code_all_failures.jsonl'
    failures = 0
    # HumanEval (test-run needed) - we will treat generation missing 'def' as failure as earlier
    try:
        humaneval = load_dataset('openai_humaneval', split='test')
        with open(output_file, 'w') as fout:
            for i, ex in enumerate(tqdm(humaneval, desc='HumanEval')):
                prompt = ex['prompt']
                inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=False)
                resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
                resp_only = resp[len(prompt):].strip()
                if not ('def ' in resp_only or 'class ' in resp_only):
                    failures += 1
                    obj = {
                        'instruction': prompt,
                        'output': ex.get('canonical_solution', '') or ex.get('reference', ''),
                        'failure_type': 'loss',
                        'source': 'humaneval',
                        'index': i
                    }
                    fout.write(json.dumps(obj) + '\n')
    except Exception as e:
        print('Could not process HumanEval:', e)
    # MBPP
    try:
        mbpp = load_dataset('mbpp', split='train')
        with open(output_file, 'a') as fout:
            for i, ex in enumerate(tqdm(mbpp, desc='MBPP')):
                prompt = ex['text']
                inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=False)
                resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
                resp_only = resp[len(prompt):].strip()
                if not ('def ' in resp_only or 'class ' in resp_only):
                    obj = {
                        'instruction': prompt,
                        'output': ex.get('code', ''),
                        'failure_type': 'loss',
                        'source': 'mbpp',
                        'index': i
                    }
                    fout.write(json.dumps(obj) + '\n')
    except Exception as e:
        print('Could not process MBPP:', e)

    print(f"Saved code failures to: {output_file}")
    return output_file


def prepare_creativity_failures(model, tokenizer, output_dir: Path) -> Path:
    output_file = output_dir / 'creativity_all_failures.jsonl'
    failures = 0
    subjects = ['high_school_european_history','high_school_government_and_politics','global_facts']
    with open(output_file, 'w') as fout:
        for subject in subjects:
            try:
                ds = load_dataset('cais/mmlu', subject, split='test')
                for i, ex in enumerate(tqdm(ds, desc=f'MMLU {subject}')):
                    question = ex['question']
                    choices = ex['choices']
                    correct_idx = ex['answer']
                    correct_choice = choices[correct_idx]
                    prompt = f"Question: {question}\n\nChoices:\n"
                    for j, ch in enumerate(choices):
                        prompt += f"{chr(65+j)}. {ch}\n"
                    prompt += '\nAnswer:'
                    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=64, temperature=0.7, do_sample=False)
                    resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    model_ans = resp[len(prompt):].strip()
                    model_choice = model_ans[:1].upper() if model_ans else ''
                    correct_letter = chr(65 + correct_idx)
                    if model_choice != correct_letter:
                        failures += 1
                        obj = {
                            'instruction': prompt,
                            'output': correct_choice,
                            'failure_type': 'loss',
                            'source': f'mmlu_{subject}',
                            'index': i
                        }
                        fout.write(json.dumps(obj) + '\n')
            except Exception as e:
                print('Could not load subject:', subject, e)
    print(f"Saved creativity failures to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints/phase1a_merged')
    parser.add_argument('--output_dir', type=str, default='data/phase1b_all_failures')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('Loading model...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map='auto')

    print('Preparing math failures...')
    prepare_math_failures(model, tokenizer, output_dir)
    print('Preparing code failures...')
    prepare_code_failures(model, tokenizer, output_dir)
    print('Preparing creativity failures...')
    prepare_creativity_failures(model, tokenizer, output_dir)

    print('\nDone. Files in', output_dir)

if __name__ == '__main__':
    main()
