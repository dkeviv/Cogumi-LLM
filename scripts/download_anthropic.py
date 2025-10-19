#!/usr/bin/env python3
"""Download Anthropic HH-RLHF conversation dataset."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

console = Console()

async def download_anthropic():
    console.print("\n[bold cyan]Downloading Anthropic HH-RLHF Dataset[/bold cyan]\n")
    
    # Download dataset
    console.print("Loading from HuggingFace...")
    dataset = await asyncio.to_thread(
        load_dataset,
        "Anthropic/hh-rlhf",
        split="train"
    )
    
    console.print(f"[green]✅ Downloaded {len(dataset):,} samples[/green]\n")
    
    # Parse and save
    output_path = Path("data/raw/anthropic-hh.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    console.print("Processing conversations...")
    saved_count = 0
    
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in dataset:
            try:
                text = sample.get("chosen", "")
                
                # Parse Human/Assistant conversation
                if "\n\nHuman:" in text and "\n\nAssistant:" in text:
                    parts = text.split("\n\nHuman:")
                    if len(parts) > 1:
                        human_part = parts[1].split("\n\nAssistant:")
                        if len(human_part) > 1:
                            instruction = human_part[0].strip()
                            assistant_part = human_part[1].split("\n\nHuman:")[0]
                            response = assistant_part.strip()
                            
                            if instruction and response:
                                f.write(json.dumps({
                                    "instruction": instruction,
                                    "response": response,
                                    "source": "Anthropic-HH"
                                }, ensure_ascii=False) + "\n")
                                saved_count += 1
            except Exception as e:
                continue
    
    console.print(f"\n[bold green]✅ Success![/bold green]")
    console.print(f"  Saved: {saved_count:,} samples")
    console.print(f"  Path: {output_path}")
    console.print(f"  Size: {output_path.stat().st_size / (1024*1024):.1f} MB")

if __name__ == "__main__":
    asyncio.run(download_anthropic())
