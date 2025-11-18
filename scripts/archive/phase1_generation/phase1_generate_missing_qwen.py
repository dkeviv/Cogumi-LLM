#!/usr/bin/env python3
"""
Phase 1: Generate Missing Questions - Qwen 2.5 Coder 32B

Fallback strategy: Use Qwen 2.5 Coder 32B which is better at following
precise instructions and generating diverse outputs.
"""

import json
import random
import asyncio
import aiohttp
from pathlib import Path
from typing import Dict, List
from rich.console import Console
from rich.progress import Progress
from datetime import datetime

console = Console()

# Try Qwen 2.5 Coder 32B - better at following instructions
FREE_MODELS = [
    "qwen/qwen-2.5-coder-32b-instruct:free",
]

OPENROUTER_API_KEY = "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"

# Target gaps for lacking domains
DOMAIN_GAPS = {
    "Common Sense": 2500,
    "Instruction": 2500,
    "Reading": 2500,
    "Summarization": 2500
}

# Subtopics for diversity
DOMAIN_SUBTOPICS = {
    "Common Sense": [
        "Daily routines and schedules",
        "Social etiquette and manners",
        "Safety awareness",
        "Health and hygiene",
        "Basic cooking and food",
        "Weather and clothing",
        "Transportation and navigation",
        "Money and shopping",
        "Time management",
        "Household tasks",
        "Pet care basics",
        "Basic first aid",
        "Communication norms",
        "Environmental awareness",
        "Technology in daily life"
    ],
    "Instruction": [
        "Recipe following and cooking",
        "Assembly instructions",
        "Software installation steps",
        "Workout routines",
        "Craft and DIY projects",
        "Travel directions",
        "Game rules and gameplay",
        "Product usage guides",
        "Safety procedures",
        "Educational lesson plans",
        "Event planning checklists",
        "Cleaning and maintenance",
        "Form filling instructions",
        "Medical prescription following",
        "Emergency response protocols"
    ],
    "Reading": [
        "Main idea identification",
        "Supporting details extraction",
        "Inference from context",
        "Author's purpose and tone",
        "Vocabulary in context",
        "Text structure analysis",
        "Comparison and contrast",
        "Cause and effect in passages",
        "Fact vs opinion distinction",
        "Summarization of passages",
        "Drawing conclusions",
        "Predicting outcomes",
        "Character analysis",
        "Theme identification",
        "Critical evaluation"
    ],
    "Summarization": [
        "News article summarization",
        "Research paper abstracts",
        "Meeting minutes recap",
        "Book chapter summaries",
        "Technical documentation condensing",
        "Email thread summarization",
        "Video transcript summaries",
        "Legal document summaries",
        "Financial report highlights",
        "Customer feedback synthesis",
        "Product review aggregation",
        "Social media thread recaps",
        "Conference talk summaries",
        "Interview transcription summaries",
        "Policy document overviews"
    ]
}


def load_existing_questions(file_path: Path) -> set:
    """Load existing questions for deduplication."""
    seen = set()
    if file_path.exists():
        with open(file_path) as f:
            for line in f:
                q = json.loads(line.strip())
                seen.add(q['question'].strip().lower())
    return seen


def create_prompt(domain: str, batch_size: int) -> str:
    """Create diverse prompt with MEDIUM difficulty."""
    subtopic = random.choice(DOMAIN_SUBTOPICS[domain])
    
    # Add random seed/variation to make each batch truly unique
    variation = random.randint(1, 1000)
    
    return f"""Generate {batch_size} MEDIUM difficulty {domain} questions specifically about: "{subtopic}"

Domain: {domain}
Subtopic Focus: {subtopic}
Difficulty: MEDIUM
Variation Seed: {variation}

CRITICAL: Each question MUST be completely different from common questions.
Think creatively and generate UNUSUAL, CREATIVE scenarios within the subtopic.

Requirements:
- Questions MUST be specifically about the subtopic: {subtopic}
- Each question should be COMPLETELY UNIQUE and DIFFERENT
- MEDIUM complexity - more nuanced than basic questions
- Use CREATIVE and UNUSUAL scenarios
- Avoid common/generic questions
- Think outside the box
- English only
- No LaTeX math (use plain text like 'x^2' not '\\(x^2\\)')

Format: Return ONLY a JSON array of strings (the questions), nothing else.
Example: ["question 1", "question 2", "question 3"]

Generate {batch_size} CREATIVE and UNIQUE questions now:"""


async def call_api_async(session, model, prompt, semaphore):
    """Call OpenRouter API asynchronously."""
    async with semaphore:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 1.4,  # Very high for maximum diversity
        }
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        try:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                content = data['choices'][0]['message']['content'].strip()
                
                # Parse JSON array
                if content.startswith('[') and content.endswith(']'):
                    questions = json.loads(content)
                    return questions
                else:
                    # Try to find JSON array in content
                    start = content.find('[')
                    end = content.rfind(']') + 1
                    if start >= 0 and end > start:
                        questions = json.loads(content[start:end])
                        return questions
                
                return None
                
        except Exception as e:
            return None


async def generate_for_domain(
    domain: str,
    target_count: int,
    seen_questions: set,
    output_file: Path
) -> int:
    """Generate questions for a single domain."""
    
    # Over-request by 4x for maximum attempts
    request_count = int(target_count * 4.0)
    
    console.print(f"\n[cyan]Generating {request_count:,} CREATIVE {domain} questions (target {target_count:,} unique)...[/cyan]")
    
    batch_size = 20
    batches = (request_count + batch_size - 1) // batch_size
    semaphore = asyncio.Semaphore(20)
    
    all_questions = []
    duplicates_found = 0
    
    with Progress() as progress:
        task = progress.add_task(f"{domain}", total=batches)
        
        async with aiohttp.ClientSession() as session:
            for i in range(batches):
                model = random.choice(FREE_MODELS)
                prompt = create_prompt(domain, batch_size)
                
                questions = await call_api_async(session, model, prompt, semaphore)
                
                if questions:
                    for q in questions:
                        q_lower = q.strip().lower()
                        if q_lower not in seen_questions:
                            all_questions.append({
                                "question": q.strip(),
                                "difficulty": "easy",  # Still mark as easy for training
                                "domain": domain,
                                "model_used": model,
                                "generated_at": datetime.now().isoformat()
                            })
                            seen_questions.add(q_lower)
                        else:
                            duplicates_found += 1
                
                progress.advance(task)
                
                # Stop early if we hit target
                if len(all_questions) >= target_count:
                    progress.update(task, completed=batches)
                    break
    
    # Save to checkpoint
    checkpoint_file = output_file.parent / f"checkpoint_{domain.lower().replace(' ', '_')}_qwen.jsonl"
    with open(checkpoint_file, 'w') as f:
        for q in all_questions:
            f.write(json.dumps(q) + '\n')
    
    console.print(f"[green]✓ Generated {len(all_questions):,} unique questions for {domain}[/green]")
    if duplicates_found > 0:
        console.print(f"[yellow]⚠ Filtered {duplicates_found:,} duplicates ({duplicates_found/(len(all_questions)+duplicates_found)*100:.1f}%)[/yellow]")
    
    return len(all_questions)


async def main():
    """Main execution."""
    console.print("\n[bold cyan]Phase 1: Generate Missing Questions - Qwen 2.5 Coder 32B[/bold cyan]")
    console.print("=" * 70)
    console.print("\n[cyan]Strategy: Using Qwen Coder with CREATIVE prompts for maximum diversity[/cyan]")
    console.print("\n[cyan]Using model:[/cyan]")
    console.print("  • Qwen 2.5 Coder 32B Instruct")
    console.print(f"\n[cyan]Target domains:[/cyan]")
    for domain, gap in DOMAIN_GAPS.items():
        console.print(f"  • {domain}: +{gap:,} questions")
    
    # Load existing for deduplication
    existing_file = Path("data/phase1/questions_60k_shuffled.jsonl")
    console.print(f"\n[cyan]Loading existing questions for deduplication...[/cyan]")
    seen = load_existing_questions(existing_file)
    console.print(f"[green]✓ Loaded {len(seen):,} existing questions[/green]")
    
    output_dir = Path("data/phase1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate for each lacking domain
    total_generated = 0
    for domain, target in DOMAIN_GAPS.items():
        generated = await generate_for_domain(domain, target, seen, output_dir / "temp.jsonl")
        total_generated += generated
    
    console.print(f"\n[green]✓ Total generated: {total_generated:,} new unique questions[/green]")
    console.print(f"[green]✓ Total in dedup set: {len(seen):,}[/green]")
    
    console.print("\n[cyan]Next step: Consolidate all files[/cyan]")


if __name__ == "__main__":
    asyncio.run(main())
