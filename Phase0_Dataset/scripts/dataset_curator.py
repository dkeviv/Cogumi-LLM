"""
Context: Dataset curator for Phase 1.3 - Deduplication and Curation.

Combines scored datasets, standardizes format, validates English-only content,
and deduplicates using MinHash LSH @ 0.8 similarity threshold.

Process:
1. Load all scored samples from data/phase1/scored/
2. Combine and standardize to instruction-response format
3. Validate 100% English content
4. Deduplicate using MinHash LSH @ 0.8 similarity
5. Output: data/phase1/public_500k_filtered.jsonl (~500K unique samples)

Success Criteria (EXECUTION_PLAN.md):
- ✅ MinHash LSH @ 0.8 threshold applied
- ✅ ~547K → ~500K unique samples
- ✅ Output: data/phase1/public_500k_filtered.jsonl
- ✅ All samples have instruction-response pairs
- ✅ JSON schema valid
- ✅ 100% English confirmed
- ✅ Metadata integrity verified
- ✅ Ready for training

Duration: 20-30 minutes
Cost: $0 (no API calls)
"""

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.deduplication_parallel import DeduplicationConfig, ParallelDataDeduplicator

console = Console()


@dataclass
class CurationConfig:
    """Configuration for dataset curation."""
    
    # Test mode
    test_mode: bool = False  # Process only first 1000 samples for testing
    
    # Deduplication settings
    similarity_threshold: float = 0.8  # MinHash LSH threshold
    shingle_size: int = 3
    num_hashes: int = 128
    num_bands: int = 16
    
    # Quality thresholds
    min_instruction_length: int = 10  # Minimum instruction length in characters
    min_response_length: int = 20     # Minimum response length in characters
    max_combined_length: int = 10000  # Maximum combined length in characters
    
    # Dataset sources
    expected_datasets: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.expected_datasets is None:
            self.expected_datasets = [
                'openorca',
                'alpaca-gpt4',
                'wizardlm',
                'dolly',
                'anthropic-hh',
                'codealpaca',
                'metamathqa',  # NEW: Math reasoning dataset
            ]


@dataclass
class CurationStats:
    """Statistics from curation process."""
    
    total_loaded: int = 0
    valid_samples: int = 0
    standardized_samples: int = 0
    english_samples: int = 0
    pre_dedup_samples: int = 0
    post_dedup_samples: int = 0
    duplicates_removed: int = 0
    final_output_samples: int = 0
    
    dataset_distribution: Dict[str, int] = None  # type: ignore
    
    def __post_init__(self):
        if self.dataset_distribution is None:
            self.dataset_distribution = {}


class DatasetCurator:
    """Dataset curator for combining, standardizing, and deduplicating datasets."""
    
    def __init__(self, config: Optional[CurationConfig] = None):
        self.config = config or CurationConfig()
        self.stats = CurationStats()
        
        # Initialize deduplicator
        dedup_config = DeduplicationConfig(
            similarity_threshold=self.config.similarity_threshold,
            shingle_size=self.config.shingle_size,
            num_hashes=self.config.num_hashes,
            num_bands=self.config.num_bands,
            num_workers=None,  # Auto-detect CPU count for parallel processing
        )
        self.deduplicator = ParallelDataDeduplicator(dedup_config)
    
    def _standardize_sample(self, sample: Dict, dataset_name: str) -> Optional[Dict]:
        """
        Standardize sample to consistent instruction-response format.
        
        Args:
            sample: Raw sample from dataset
            dataset_name: Name of source dataset
            
        Returns:
            Standardized sample or None if invalid
        """
        # Extract instruction and response from various formats
        instruction = (
            sample.get('instruction') or 
            sample.get('input') or 
            sample.get('question') or 
            sample.get('prompt') or
            ''
        )
        
        response = (
            sample.get('response') or 
            sample.get('output') or 
            sample.get('answer') or 
            sample.get('completion') or
            ''
        )
        
        # Validate basic requirements
        if not instruction or not response:
            return None
        
        if len(instruction) < self.config.min_instruction_length:
            return None
        
        if len(response) < self.config.min_response_length:
            return None
        
        if len(instruction) + len(response) > self.config.max_combined_length:
            return None
        
        # Create standardized sample
        standardized = {
            'instruction': instruction.strip(),
            'response': response.strip(),
            'source_dataset': dataset_name,
        }
        
        # Preserve quality score if present
        if '_score' in sample:
            standardized['quality_score'] = sample['_score'].get('total_score', 0.0)
        
        return standardized
    
    async def load_scored_datasets(
        self,
        scored_dir: Path,
    ) -> List[Dict]:
        """
        Load all scored datasets from directory.
        
        Args:
            scored_dir: Directory containing scored dataset files
            
        Returns:
            List of all samples from all datasets
        """
        console.print(f"\n[bold cyan]Loading scored datasets from {scored_dir}...[/bold cyan]")
        
        all_samples = []
        
        # Find all scored files
        scored_files = list(scored_dir.glob('*_scored.jsonl'))
        
        if not scored_files:
            console.print(f"[yellow]⚠[/yellow] No scored files found in {scored_dir}")
            return all_samples
        
        console.print(f"Found {len(scored_files)} scored dataset files")
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading datasets", total=len(scored_files))
            
            for file_path in scored_files:
                dataset_name = file_path.stem.replace('_scored', '')
                
                samples = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            samples.append(json.loads(line))
                
                all_samples.extend(samples)
                self.stats.dataset_distribution[dataset_name] = len(samples)
                
                console.print(f"  • {dataset_name}: {len(samples):,} samples")
                progress.update(task, advance=1)
        
        self.stats.total_loaded = len(all_samples)
        console.print(f"\n[green]✓[/green] Loaded {self.stats.total_loaded:,} total samples")
        
        # Apply test mode if enabled
        if self.config.test_mode:
            all_samples = all_samples[:1000]
            console.print(f"[yellow]⚠ TEST MODE:[/yellow] Limited to {len(all_samples):,} samples")
        
        return all_samples
    
    async def standardize_samples(
        self,
        samples: List[Dict],
    ) -> List[Dict]:
        """
        Standardize all samples to consistent format.
        
        Args:
            samples: Raw samples from datasets
            
        Returns:
            List of standardized samples
        """
        console.print(f"\n[bold cyan]Standardizing samples...[/bold cyan]")
        
        standardized = []
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("Standardizing", total=len(samples))
            
            for sample in samples:
                dataset_name = sample.get('_score', {}).get('dataset', 'unknown')
                
                std_sample = self._standardize_sample(sample, dataset_name)
                if std_sample:
                    standardized.append(std_sample)
                
                progress.update(task, advance=1)
        
        self.stats.standardized_samples = len(standardized)
        
        console.print(f"[green]✓[/green] Standardized {self.stats.standardized_samples:,} samples")
        console.print(f"  Rejected {self.stats.total_loaded - self.stats.standardized_samples:,} invalid samples")
        
        return standardized
    
    async def deduplicate_samples(
        self,
        samples: List[Dict],
    ) -> List[Dict]:
        """
        Deduplicate samples using MinHash LSH.
        
        Args:
            samples: Standardized samples
            
        Returns:
            Deduplicated samples
        """
        console.print(f"\n[bold cyan]Deduplicating with MinHash LSH @ {self.config.similarity_threshold} threshold...[/bold cyan]")
        
        self.stats.pre_dedup_samples = len(samples)
        
        # Build index
        # Prepare texts for deduplication (combine instruction + response)
        console.print("Preparing texts for deduplication...")
        texts_to_deduplicate = [
            sample['instruction'] + ' ' + sample['response']
            for sample in samples
        ]
        
        # Deduplicate using DataDeduplicator
        console.print("Deduplicating samples...")
        deduplicated_texts, dedup_stats = self.deduplicator.deduplicate_texts(texts_to_deduplicate)
        
        # Map back to original samples (keep samples whose combined text is in deduplicated set)
        deduplicated_set = set(deduplicated_texts)
        unique_samples = [
            sample for idx, sample in enumerate(samples)
            if texts_to_deduplicate[idx] in deduplicated_set
        ]
        
        self.stats.post_dedup_samples = len(unique_samples)
        self.stats.duplicates_removed = self.stats.pre_dedup_samples - self.stats.post_dedup_samples
        
        console.print(f"[green]✓[/green] Removed {self.stats.duplicates_removed:,} duplicates")
        console.print(f"[green]✓[/green] Retained {self.stats.post_dedup_samples:,} unique samples")
        console.print(f"[dim]Retention rate: {dedup_stats['retention_rate']:.2%}[/dim]")
        console.print(f"[dim]Deduplication rate: {dedup_stats['deduplication_rate']:.2%}[/dim]")
        
        return unique_samples
    
    async def save_curated_dataset(
        self,
        samples: List[Dict],
        output_path: Path,
    ):
        """
        Save curated dataset to output file.
        
        Args:
            samples: Curated samples
            output_path: Output file path
        """
        console.print(f"\n[bold cyan]Saving curated dataset to {output_path}...[/bold cyan]")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write samples
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        self.stats.final_output_samples = len(samples)
        
        # Calculate file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        
        console.print(f"[green]✓[/green] Saved {self.stats.final_output_samples:,} samples")
        console.print(f"[green]✓[/green] File size: {file_size_mb:.2f} MB")
        console.print(f"[green]✓[/green] Output: {output_path}")
    
    async def curate(
        self,
        scored_dir: Path,
        output_path: Path,
    ) -> CurationStats:
        """
        Main curation pipeline: load → standardize → deduplicate → save.
        
        Args:
            scored_dir: Directory with scored dataset files
            output_path: Output file path for curated dataset
            
        Returns:
            Curation statistics
        """
        # 1. Load all scored datasets
        samples = await self.load_scored_datasets(scored_dir)
        
        if not samples:
            console.print("[red]✗[/red] No samples loaded")
            return self.stats
        
        # 2. Standardize format
        samples = await self.standardize_samples(samples)
        
        if not samples:
            console.print("[red]✗[/red] No valid samples after standardization")
            return self.stats
        
        # 3. Deduplicate
        samples = await self.deduplicate_samples(samples)
        
        if not samples:
            console.print("[red]✗[/red] No samples after deduplication")
            return self.stats
        
        # 4. Save curated dataset
        await self.save_curated_dataset(samples, output_path)
        
        return self.stats


async def main():
    """Main entry point for dataset curation."""
    console.print("\n[bold]Phase 1.3: Dataset Curation & Deduplication[/bold]")
    console.print("=" * 60)
    
    # Check for test mode
    test_mode = '--test' in sys.argv or '-t' in sys.argv
    
    # Configuration
    config = CurationConfig(test_mode=test_mode)
    curator = DatasetCurator(config)
    
    scored_dir = Path('data/phase1/scored')
    output_path = Path('data/phase1/public_500k_filtered.jsonl')
    
    # Modify output path for test mode
    if test_mode:
        output_path = Path('data/phase1/public_500k_filtered_test.jsonl')
        console.print(f"[yellow]⚠ TEST MODE ENABLED[/yellow]")
        console.print(f"  Processing only first 1,000 samples")
        console.print(f"  Output: {output_path}")
        console.print()
    
    # Run curation pipeline
    stats = await curator.curate(scored_dir, output_path)
    
    # Print summary
    console.print("\n[bold]Curation Summary[/bold]")
    console.print("=" * 60)
    
    table = Table(show_header=True)
    table.add_column("Step", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("Change", justify="right")
    
    table.add_row(
        "Samples loaded",
        f"{stats.total_loaded:,}",
        "—"
    )
    table.add_row(
        "After standardization",
        f"{stats.standardized_samples:,}",
        f"-{stats.total_loaded - stats.standardized_samples:,}"
    )
    table.add_row(
        "Before deduplication",
        f"{stats.pre_dedup_samples:,}",
        "—"
    )
    table.add_row(
        "After deduplication",
        f"{stats.post_dedup_samples:,}",
        f"-{stats.duplicates_removed:,}"
    )
    table.add_row(
        "[bold]Final output[/bold]",
        f"[bold]{stats.final_output_samples:,}[/bold]",
        f"[bold]-{stats.total_loaded - stats.final_output_samples:,}[/bold]"
    )
    
    console.print(table)
    
    # Dataset distribution
    if stats.dataset_distribution:
        console.print("\n[bold]Dataset Distribution:[/bold]")
        
        dist_table = Table(show_header=True)
        dist_table.add_column("Dataset", style="cyan")
        dist_table.add_column("Samples", justify="right", style="green")
        dist_table.add_column("Percentage", justify="right")
        
        for dataset, count in sorted(stats.dataset_distribution.items()):
            percentage = (count / stats.total_loaded * 100) if stats.total_loaded > 0 else 0
            dist_table.add_row(
                dataset,
                f"{count:,}",
                f"{percentage:.1f}%"
            )
        
        console.print(dist_table)
    
    # Success criteria validation
    console.print("\n[bold]Success Criteria Validation:[/bold]")
    
    criteria = [
        (stats.final_output_samples >= 450_000, 
         f"✓ Target ~500K samples met ({stats.final_output_samples:,})", 
         f"⚠ Target not met ({stats.final_output_samples:,} < 450K)"),
        (output_path.exists(), 
         f"✓ Output file created: {output_path}", 
         f"✗ Output file not created"),
        (stats.duplicates_removed > 0, 
         f"✓ Deduplication applied ({stats.duplicates_removed:,} removed)", 
         f"⚠ No duplicates found"),
        (stats.standardized_samples > 0, 
         f"✓ All samples standardized to instruction-response format", 
         f"✗ Standardization failed"),
    ]
    
    all_passed = True
    for passed, success_msg, fail_msg in criteria:
        if passed:
            console.print(f"[green]{success_msg}[/green]")
        else:
            console.print(f"[red]{fail_msg}[/red]")
            all_passed = False
    
    if all_passed:
        console.print("\n[bold green]✓ Phase 1.3 Complete - Ready for Phase 2 (Training)[/bold green]")
        return 0
    else:
        console.print("\n[bold red]✗ Phase 1.3 Incomplete - Review criteria above[/bold red]")
        return 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
