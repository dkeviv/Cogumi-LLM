"""
Context: Export true failures for Phase 1C targeted distillation
- Extracts 2,139 genuine failures from deep analysis
- Groups by failure type (missing answer, logic error, wrong logic)
- Creates training dataset for Phase 1C
- Documents failure patterns for targeted teaching
"""

import json
from collections import defaultdict
from typing import Dict, List, Any

DEEP_ANALYSIS_FILE = "data/failure_analysis_deep.json"
TEST_FILE = "data/test_dataset_20k.jsonl"
MODEL_FILE = "data/model_outputs_20k.jsonl"
FAILURES_EXPORT_FILE = "data/phase1c_true_failures.jsonl"
PATTERNS_FILE = "data/failure_patterns_phase1c.json"

def load_all_data():
    """Load all necessary data."""
    with open(DEEP_ANALYSIS_FILE, 'r') as f:
        report = json.load(f)
    
    test_data = {}
    with open(TEST_FILE, 'r') as f:
        for idx, line in enumerate(f):
            test_data[idx] = json.loads(line)
    
    model_data = {}
    with open(MODEL_FILE, 'r') as f:
        for idx, line in enumerate(f):
            model_data[idx] = json.loads(line)
    
    return report, test_data, model_data


def categorize_failure(failure: Dict, test_item: Dict, model_item: Dict) -> str:
    """Categorize the type of failure."""
    analysis = failure.get('deep_analysis', {})
    category = failure.get('category', 'other')
    
    if category == 'math':
        model_numbers = analysis.get('model_numbers', [])
        ref_numbers = analysis.get('reference_numbers', [])
        
        if not model_numbers and ref_numbers:
            return 'missing_numerical_answer'
        elif model_numbers and not ref_numbers:
            return 'spurious_numbers'
        else:
            return 'wrong_calculation'
    
    elif category == 'code':
        code_quality = analysis.get('code_quality', {})
        if code_quality.get('logic_appears_sound'):
            return 'minor_logic_issue'
        else:
            return 'major_logic_error'
    
    elif category == 'reasoning':
        reasoning_quality = analysis.get('reasoning_quality', {})
        if not reasoning_quality.get('logic_valid'):
            return 'invalid_logic'
        else:
            return 'unsupported_conclusion'
    
    elif category == 'qa':
        qa_quality = analysis.get('qa_quality', {})
        if not qa_quality.get('answers_question'):
            return 'misses_question'
        else:
            return 'inaccurate_facts'
    
    else:
        return 'instruction_following'


def main():
    """Export true failures with categorization."""
    print("Loading data...")
    report, test_data, model_data = load_all_data()
    
    print("Extracting true failures...")
    all_deep = report['deep_analysis_all']
    true_failures = [r for r in all_deep if not r['is_likely_false_positive']]
    
    print(f"Found {len(true_failures)} true failures\n")
    
    # Categorize failures
    failure_types = defaultdict(list)
    category_stats = defaultdict(lambda: defaultdict(int))
    
    for failure in true_failures:
        failure_id = failure['id']
        category = failure['category']
        
        if failure_id not in test_data or failure_id not in model_data:
            continue
        
        failure_type = categorize_failure(failure, test_data[failure_id], model_data[failure_id])
        failure_types[failure_type].append(failure_id)
        category_stats[category][failure_type] += 1
    
    # Export failures as JSONL
    exported_count = 0
    with open(FAILURES_EXPORT_FILE, 'w') as f:
        for failure in true_failures:
            failure_id = failure['id']
            if failure_id not in test_data or failure_id not in model_data:
                continue
            
            failure_type = categorize_failure(failure, test_data[failure_id], model_data[failure_id])
            
            record = {
                "id": failure_id,
                "category": failure['category'],
                "failure_type": failure_type,
                "instruction": test_data[failure_id].get("instruction", ""),
                "reference_answer": test_data[failure_id].get("response", ""),
                "model_output": model_data[failure_id].get("model_output", ""),
                "quality_score": test_data[failure_id].get("quality_score", 0),
                "analysis": failure.get('deep_analysis', {}),
                "evidence": failure.get('evidence', [])
            }
            f.write(json.dumps(record) + "\n")
            exported_count += 1
    
    # Generate failure patterns report
    patterns_report = {
        "summary": {
            "total_true_failures": len(true_failures),
            "exported_failures": exported_count,
            "unique_failure_types": len(failure_types)
        },
        "failure_types": {
            ftype: {
                "count": len(ids),
                "percentage": round(100 * len(ids) / len(true_failures), 2),
                "ids_sample": ids[:10]
            }
            for ftype, ids in failure_types.items()
        },
        "by_category": {
            cat: {
                ftypes: count
                for ftypes, count in sorted(failures.items(), key=lambda x: x[1], reverse=True)
            }
            for cat, failures in category_stats.items()
        }
    }
    
    with open(PATTERNS_FILE, 'w') as f:
        json.dump(patterns_report, f, indent=2)
    
    # Print summary
    print("="*70)
    print("FAILURE CATEGORIZATION FOR PHASE 1C")
    print("="*70)
    
    print(f"\nTrue Failures by Type:")
    for ftype, ids in sorted(failure_types.items(), key=lambda x: len(x[1]), reverse=True):
        pct = 100 * len(ids) / len(true_failures)
        print(f"  {ftype:30}: {len(ids):5} ({pct:5.1f}%)")
    
    print(f"\nBy Category & Type:")
    for cat in sorted(category_stats.keys()):
        print(f"  {cat}:")
        for ftype, count in sorted(category_stats[cat].items(), key=lambda x: x[1], reverse=True):
            print(f"    {ftype:25}: {count:5}")
    
    print(f"\n{'='*70}")
    print(f"Exported {exported_count} failures to {FAILURES_EXPORT_FILE}")
    print(f"Failure patterns saved to {PATTERNS_FILE}")
    print(f"\nReady for Phase 1C targeted distillation ✅")


if __name__ == "__main__":
    main()
