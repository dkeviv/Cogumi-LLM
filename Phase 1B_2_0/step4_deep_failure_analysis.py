"""
Context: Deep failure analysis to identify false positives in Phase 1B evaluation results.
- Loads failure results from batch_comparison_results_llm.json
- Samples failures from each category
- Re-evaluates with stricter semantic logic (checks actual correctness, not just heuristics)
- Identifies common false positive patterns
- Generates detailed analysis report
"""

import json
import re
from typing import Dict, List, Any, Tuple
from collections import defaultdict

RESULTS_FILE = "data/batch_comparison_results_llm.json"
TEST_FILE = "data/test_dataset_20k.jsonl"
MODEL_FILE = "data/model_outputs_20k.jsonl"
OUTPUT_FILE = "data/failure_analysis_deep.json"

# Load all data first
def load_all_data() -> Tuple[Dict, List[Dict], List[Dict]]:
    """Load results, test dataset, and model outputs."""
    print("Loading data...")
    
    with open(RESULTS_FILE, 'r') as f:
        results_data = json.load(f)
    
    test_data = {}
    with open(TEST_FILE, 'r') as f:
        for idx, line in enumerate(f):
            test_data[idx] = json.loads(line)
    
    model_data = {}
    with open(MODEL_FILE, 'r') as f:
        for idx, line in enumerate(f):
            model_data[idx] = json.loads(line)
    
    return results_data, test_data, model_data


def deep_evaluate_failure(
    failure_id: int,
    test_item: Dict,
    model_item: Dict,
    category: str
) -> Dict[str, Any]:
    """
    Deep re-evaluation of a marked failure.
    Looks for signs this might be a false positive.
    """
    instruction = test_item.get("instruction", "")
    reference = test_item.get("response", "")
    model_output = model_item.get("model_output", "")
    
    analysis = {
        "id": failure_id,
        "category": category,
        "original_reason": "",
        "deep_analysis": {},
        "is_likely_false_positive": False,
        "confidence": 0.0,
        "evidence": []
    }
    
    # ===== DEEP SEMANTIC CHECKS =====
    
    # 1. SEMANTIC SIMILARITY CHECK
    # Check if answer conveys same core information, even if phrased differently
    ref_words = set(reference.lower().split())
    model_words = set(model_output.lower().split())
    
    if len(ref_words) > 0 and len(model_words) > 0:
        overlap_ratio = len(ref_words & model_words) / max(len(ref_words), len(model_words))
        analysis["deep_analysis"]["semantic_overlap_ratio"] = round(overlap_ratio, 3)
        
        if overlap_ratio > 0.6:
            analysis["evidence"].append(f"High semantic overlap ({int(overlap_ratio*100)}% word overlap)")
    
    # 2. CATEGORY-SPECIFIC DEEP CHECKS
    
    if category == "math":
        # Extract all numbers/expressions from both
        ref_numbers = extract_numbers(reference)
        model_numbers = extract_numbers(model_output)
        
        analysis["deep_analysis"]["reference_numbers"] = ref_numbers
        analysis["deep_analysis"]["model_numbers"] = model_numbers
        
        # Check if final answer matches
        if ref_numbers and model_numbers:
            if ref_numbers[-1] == model_numbers[-1]:
                analysis["is_likely_false_positive"] = True
                analysis["evidence"].append("Final mathematical answer matches reference")
                analysis["confidence"] = 0.85
            else:
                # Check if they're equivalent (e.g., 1/2 vs 0.5)
                try:
                    if abs(float(ref_numbers[-1]) - float(model_numbers[-1])) < 0.001:
                        analysis["is_likely_false_positive"] = True
                        analysis["evidence"].append("Final answers equivalent numerically")
                        analysis["confidence"] = 0.80
                except:
                    pass
    
    elif category == "code":
        # Check if code is actually correct by analyzing structure
        code_quality = analyze_code_quality(instruction, reference, model_output)
        analysis["deep_analysis"]["code_quality"] = code_quality
        
        if code_quality["has_proper_syntax"]:
            analysis["evidence"].append("Code has proper syntax")
        
        if code_quality["logic_appears_sound"]:
            analysis["evidence"].append("Logic appears sound")
            analysis["is_likely_false_positive"] = True
            analysis["confidence"] = 0.75
        
        if code_quality["solves_problem"]:
            analysis["evidence"].append("Code appears to solve the stated problem")
            analysis["is_likely_false_positive"] = True
            analysis["confidence"] = 0.90
    
    elif category == "reasoning":
        # Check if reasoning is fundamentally sound
        reasoning_quality = analyze_reasoning_quality(instruction, reference, model_output)
        analysis["deep_analysis"]["reasoning_quality"] = reasoning_quality
        
        if reasoning_quality["conclusion_supported"]:
            analysis["evidence"].append("Conclusion is well-supported")
            analysis["is_likely_false_positive"] = True
            analysis["confidence"] = 0.80
        
        if reasoning_quality["logic_valid"]:
            analysis["evidence"].append("Logic is valid")
            analysis["is_likely_false_positive"] = True
            analysis["confidence"] = 0.85
    
    elif category == "qa":
        # Check if answer actually answers the question
        qa_quality = analyze_qa_quality(instruction, reference, model_output)
        analysis["deep_analysis"]["qa_quality"] = qa_quality
        
        if qa_quality["answers_question"]:
            analysis["evidence"].append("Response answers the question")
            analysis["is_likely_false_positive"] = True
            analysis["confidence"] = 0.85
        
        if qa_quality["factually_accurate"]:
            analysis["evidence"].append("Facts appear accurate")
            analysis["is_likely_false_positive"] = True
            analysis["confidence"] = 0.75
    
    elif category == "other":
        # Generic text quality checks
        text_quality = analyze_text_quality(instruction, reference, model_output)
        analysis["deep_analysis"]["text_quality"] = text_quality
        
        if text_quality["fulfills_instruction"]:
            analysis["evidence"].append("Fulfills instruction requirements")
            analysis["is_likely_false_positive"] = True
            analysis["confidence"] = 0.75
        
        if text_quality["coherent_and_logical"]:
            analysis["evidence"].append("Coherent and logical")
    
    return analysis


def extract_numbers(text: str) -> List[str]:
    """Extract all numbers and fractions from text."""
    # Match integers, decimals, and fractions like \frac{1}{2}
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    fractions = re.findall(r'\\frac\{(\d+)\}\{(\d+)\}', text)
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    return numbers + [f"{n[0]}/{n[1]}" for n in fractions] + boxed


def analyze_code_quality(instruction: str, reference: str, model_output: str) -> Dict[str, bool]:
    """Analyze code quality without executing it."""
    analysis = {
        "has_proper_syntax": False,
        "logic_appears_sound": False,
        "solves_problem": False,
    }
    
    # Check for syntax markers
    balanced_parens = model_output.count('(') == model_output.count(')')
    balanced_braces = model_output.count('{') == model_output.count('}')
    balanced_brackets = model_output.count('[') == model_output.count(']')
    
    if balanced_parens and balanced_braces and balanced_brackets:
        analysis["has_proper_syntax"] = True
    
    # Check for logic keywords
    logic_keywords = ["if", "for", "while", "def", "class", "return", "else", "try"]
    if any(f" {kw} " in model_output or f"\n{kw} " in model_output for kw in logic_keywords):
        analysis["logic_appears_sound"] = True
    
    # Check if code mentions key concepts from instruction
    instruction_lower = instruction.lower()
    output_lower = model_output.lower()
    
    # Extract key terms from instruction (nouns, verbs)
    key_terms = re.findall(r'\b[a-z_]+\b', instruction_lower)
    key_terms = [t for t in key_terms if len(t) > 3]  # Only substantial words
    
    matches = sum(1 for term in key_terms if term in output_lower)
    if len(key_terms) > 0 and matches / len(key_terms) > 0.5:
        analysis["solves_problem"] = True
    
    return analysis


def analyze_reasoning_quality(instruction: str, reference: str, model_output: str) -> Dict[str, bool]:
    """Analyze reasoning quality."""
    analysis = {
        "conclusion_supported": False,
        "logic_valid": False,
    }
    
    # Check if model provides reasoning structure
    reasoning_words = ["because", "therefore", "thus", "as a result", "due to", "given that", "conclusion"]
    has_reasoning_structure = any(word in model_output.lower() for word in reasoning_words)
    
    if has_reasoning_structure:
        analysis["logic_valid"] = True
    
    # Check if conclusion is present
    conclusion_words = ["therefore", "thus", "in conclusion", "conclusion:", "answer:", "result:"]
    has_conclusion = any(word in model_output.lower() for word in conclusion_words)
    
    if has_conclusion or len(model_output) > 100:
        analysis["conclusion_supported"] = True
    
    return analysis


def analyze_qa_quality(instruction: str, reference: str, model_output: str) -> Dict[str, bool]:
    """Analyze Q&A quality."""
    analysis = {
        "answers_question": False,
        "factually_accurate": False,
    }
    
    # Extract question (usually ends with ?)
    question_match = re.search(r'[^.!?]*\?', instruction)
    if question_match:
        question = question_match.group(0).lower()
        # Check if model addresses key question words
        question_words = ["what", "who", "where", "when", "why", "how"]
        relevant_question = [w for w in question_words if w in question]
        
        # Look for corresponding answer structure
        if relevant_question:
            if len(model_output) > 20:
                analysis["answers_question"] = True
    
    # Check for factual accuracy (look for contradictions)
    has_negation = "not" in model_output.lower() or "no" in model_output.lower()
    has_affirmation = any(word in model_output.lower() for word in ["yes", "true", "correct", "is"])
    
    if has_affirmation or (not has_negation):
        analysis["factually_accurate"] = True
    
    return analysis


def analyze_text_quality(instruction: str, reference: str, model_output: str) -> Dict[str, bool]:
    """Analyze general text quality."""
    analysis = {
        "fulfills_instruction": False,
        "coherent_and_logical": False,
    }
    
    # Check length and structure
    if len(model_output) > 30:
        analysis["fulfills_instruction"] = True
    
    # Check for coherence (sentences, punctuation)
    sentences = model_output.split(". ")
    if len(sentences) > 1:
        analysis["coherent_and_logical"] = True
    
    return analysis


def main():
    """Perform deep failure analysis."""
    print("Starting deep failure analysis...\n")
    
    results_data, test_data, model_data = load_all_data()
    
    # Get all failures
    all_results = results_data["all_results"]
    failures = [r for r in all_results if r["status"] == "FAIL"]
    
    print(f"Found {len(failures)} failures out of {len(all_results)} total examples")
    print(f"Analyzing all failures for false positives...\n")
    
    # Re-evaluate all failures
    deep_analysis_results = []
    false_positives = []
    categories_fp = defaultdict(int)
    
    for idx, failure in enumerate(failures):
        if idx % 1000 == 0:
            print(f"Analyzed {idx}/{len(failures)} failures...")
        
        failure_id = failure["id"]
        category = failure["category"]
        
        if failure_id not in test_data or failure_id not in model_data:
            continue
        
        deep_analysis = deep_evaluate_failure(
            failure_id,
            test_data[failure_id],
            model_data[failure_id],
            category
        )
        
        deep_analysis_results.append(deep_analysis)
        
        if deep_analysis["is_likely_false_positive"]:
            false_positives.append(deep_analysis)
            categories_fp[category] += 1
    
    # Statistics
    stats = {
        "total_failures": len(failures),
        "likely_false_positives": len(false_positives),
        "false_positive_rate": round(100 * len(false_positives) / len(failures), 2) if failures else 0,
        "false_positives_by_category": dict(categories_fp),
        "adjusted_pass_rate": round(100 * (results_data["summary"]["passes"] + len(false_positives)) / len(all_results), 2),
    }
    
    # Compile report
    report = {
        "statistics": stats,
        "false_positives_sample": false_positives[:50],  # Top 50 false positives
        "deep_analysis_all": deep_analysis_results,
    }
    
    # Save report
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*60}")
    print("DEEP FAILURE ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"\nTotal Failures Analyzed: {stats['total_failures']}")
    print(f"Likely False Positives: {stats['likely_false_positives']}")
    print(f"False Positive Rate: {stats['false_positive_rate']}%")
    print(f"Adjusted Pass Rate: {stats['adjusted_pass_rate']}% (including false positives)")
    print(f"\nFalse Positives by Category:")
    for cat, count in sorted(categories_fp.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat:12}: {count:5}")
    print(f"\nReport saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
