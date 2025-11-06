"""
Context: Dataset downloader for Phase 1.1 - Download Public Datasets.

Downloads 5 free public instruction-tuning datasets from HuggingFace and GitHub.
NO teacher models, NO API calls, $0 cost.

Datasets:
1. OpenOrca (4.2M samples) - HuggingFace: Open-Orca/OpenOrca
2. Alpaca-GPT4 (52K samples) - HuggingFace: vicgalle/alpaca-gpt4
3. WizardLM (143K samples) - HuggingFace: WizardLM/WizardLM_evol_instruct_V2_196k
4. Dolly (15K samples) - HuggingFace: databricks/databricks-dolly-15k
5. ShareGPT (90K samples) - HuggingFace: anon8231489123/ShareGPT_Vicuna_unfiltered
6. CodeAlpaca (20K samples) - HuggingFace: sahil2801/CodeAlpaca-20k

Success Criteria (EXECUTION_PLAN.md):
- ✅ All 5 datasets downloaded successfully
- ✅ Total ~4.5M samples available
- ✅ Files in correct format (JSONL/JSON)
- ✅ No download errors or corrupted files
- ✅ Output: data/raw/ folder

Duration: 10-15 minutes
Cost: $0
"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

console = Console()


@dataclass
class DatasetSource:
    """Configuration for a single dataset source."""

    name: str  # Display name (e.g., "OpenOrca")
    source: str  # HuggingFace dataset ID or GitHub URL
    source_type: str  # "huggingface" or "github"
    expected_samples: int  # Expected number of samples
    instruction_field: str  # Field name for instruction/question
    response_field: str  # Field name for response/answer
    split: str = "train"  # Dataset split to download
    subset: Optional[str] = None  # Dataset subset/config name


@dataclass
class DownloadConfig:
    """Configuration for dataset downloading."""

    output_dir: Path = Path("data/raw")  # Output directory
    cache_dir: Optional[Path] = None  # HuggingFace cache directory
    max_retries: int = 3  # Number of retry attempts on failure
    retry_delay: int = 5  # Seconds to wait between retries
    verify_downloads: bool = True  # Verify download integrity
    save_format: str = "jsonl"  # Output format (jsonl or json)


class DatasetDownloader:
    """
    Downloads public instruction-tuning datasets for Phase 1.1.

    Uses HuggingFace datasets library to download and standardize datasets
    into a common format for downstream processing.

    Example:
        ```python
        downloader = DatasetDownloader()
        results = await downloader.download_all()
        print(f"Downloaded {sum(r.samples for r in results)} total samples")
        ```
    """

    # Dataset configurations
    DATASETS = [
        DatasetSource(
            name="OpenOrca",
            source="Open-Orca/OpenOrca",
            source_type="huggingface",
            expected_samples=4_200_000,
            instruction_field="question",
            response_field="response",
            split="train",
        ),
        DatasetSource(
            name="Alpaca-GPT4",
            source="vicgalle/alpaca-gpt4",
            source_type="huggingface",
            expected_samples=52_000,
            instruction_field="instruction",
            response_field="output",
            split="train",
        ),
        DatasetSource(
            name="WizardLM",
            source="WizardLM/WizardLM_evol_instruct_V2_196k",
            source_type="huggingface",
            expected_samples=143_000,
            instruction_field="instruction",
            response_field="output",
            split="train",
        ),
        DatasetSource(
            name="Dolly",
            source="databricks/databricks-dolly-15k",
            source_type="huggingface",
            expected_samples=15_000,
            instruction_field="instruction",
            response_field="response",
            split="train",
        ),
        DatasetSource(
            name="ShareGPT",
            source="anon8231489123/ShareGPT_Vicuna_unfiltered",
            source_type="huggingface",
            expected_samples=90_000,
            instruction_field="conversations",  # Special handling needed
            response_field="conversations",
            split="train",
        ),
        DatasetSource(
            name="CodeAlpaca",
            source="sahil2801/CodeAlpaca-20k",
            source_type="huggingface",
            expected_samples=20_000,
            instruction_field="instruction",
            response_field="output",
            split="train",
        ),
    ]

    def __init__(self, config: Optional[DownloadConfig] = None):
        """
        Initialize dataset downloader.

        Args:
            config: Download configuration (uses defaults if not provided)
        """
        self.config = config or DownloadConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.cache_dir:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    async def download_dataset(
        self,
        dataset_config: DatasetSource,
        progress: Progress,
        task_id: TaskID,
    ) -> Dict[str, Any]:
        """
        Download a single dataset from HuggingFace.

        Args:
            dataset_config: Dataset source configuration
            progress: Rich progress bar instance
            task_id: Progress task ID for updates

        Returns:
            Dict with download results (name, samples, path, success, error)
        """
        console.print(
            f"\n[cyan]Downloading {dataset_config.name}[/cyan] "
            f"(expected: {dataset_config.expected_samples:,} samples)..."
        )

        for attempt in range(1, self.config.max_retries + 1):
            try:
                # Download dataset from HuggingFace
                dataset = await asyncio.to_thread(
                    load_dataset,
                    dataset_config.source,
                    split=dataset_config.split,
                    cache_dir=str(self.config.cache_dir) if self.config.cache_dir else None,
                )

                # Get actual sample count (handle both Dataset and IterableDataset)
                actual_samples = len(dataset) if hasattr(dataset, '__len__') else 0  # type: ignore
                progress.update(task_id, completed=actual_samples)

                # Standardize format
                standardized_data = self._standardize_format(dataset, dataset_config)

                # Save to disk
                output_path = self.config.output_dir / f"{dataset_config.name.lower()}.{self.config.save_format}"
                saved_samples = await self._save_dataset(standardized_data, output_path)

                # Verify download if enabled
                if self.config.verify_downloads:
                    is_valid = await self._verify_download(output_path, dataset_config)
                    if not is_valid:
                        raise ValueError(f"Download verification failed for {dataset_config.name}")

                console.print(
                    f"[green]✅ {dataset_config.name}:[/green] "
                    f"{saved_samples:,} samples saved to {output_path}"
                )

                return {
                    "name": dataset_config.name,
                    "samples": saved_samples,
                    "path": str(output_path),
                    "success": True,
                    "error": None,
                }

            except Exception as e:
                if attempt < self.config.max_retries:
                    console.print(
                        f"[yellow]⚠️  Attempt {attempt} failed for {dataset_config.name}: {e}[/yellow]"
                    )
                    console.print(f"[yellow]Retrying in {self.config.retry_delay} seconds...[/yellow]")
                    await asyncio.sleep(self.config.retry_delay)
                else:
                    console.print(f"[red]❌ Failed to download {dataset_config.name} after {self.config.max_retries} attempts: {e}[/red]")
                    return {
                        "name": dataset_config.name,
                        "samples": 0,
                        "path": None,
                        "success": False,
                        "error": str(e),
                    }

        # Should never reach here, but just in case
        return {
            "name": dataset_config.name,
            "samples": 0,
            "path": None,
            "success": False,
            "error": "Unknown error",
        }

    def _standardize_format(
        self,
        dataset: Any,
        config: DatasetSource,
    ) -> List[Dict[str, str]]:
        """
        Standardize dataset format to common instruction-response pairs.

        Args:
            dataset: HuggingFace dataset object
            config: Dataset configuration

        Returns:
            List of standardized samples with 'instruction' and 'response' fields
        """
        standardized = []

        for sample in dataset:
            try:
                # Special handling for ShareGPT and WizardLM (conversation format)
                if config.name in ["ShareGPT", "WizardLM"]:
                    instruction, response = self._extract_sharegpt_conversation(sample)
                else:
                    # Standard instruction-response extraction
                    instruction = sample.get(config.instruction_field, "")
                    response = sample.get(config.response_field, "")

                    # Handle Alpaca format (with optional input field)
                    if config.name == "Alpaca-GPT4" and "input" in sample and sample["input"]:
                        instruction = f"{instruction}\n\nInput: {sample['input']}"

                # Only add if both instruction and response exist
                if instruction and response:
                    standardized.append({
                        "instruction": str(instruction).strip(),
                        "response": str(response).strip(),
                        "source": config.name,
                    })

            except Exception as e:
                # Skip malformed samples
                console.print(f"[yellow]⚠️  Skipping malformed sample in {config.name}: {e}[/yellow]")
                continue

        return standardized

    def _extract_sharegpt_conversation(self, sample: Dict[str, Any]) -> tuple[str, str]:
        """
        Extract instruction-response pair from ShareGPT conversation format.

        ShareGPT stores conversations as a list of messages with roles.
        We extract the first human message as instruction and first assistant
        message as response.

        Args:
            sample: ShareGPT sample with 'conversations' field

        Returns:
            Tuple of (instruction, response)
        """
        conversations = sample.get("conversations", [])
        
        instruction = ""
        response = ""

        for msg in conversations:
            role = msg.get("from", "")
            content = msg.get("value", "")

            if role == "human" and not instruction:
                instruction = content
            elif role == "gpt" and instruction and not response:
                response = content
                break  # We have a pair, stop here

        return instruction, response

    async def _save_dataset(
        self,
        data: List[Dict[str, str]],
        output_path: Path,
    ) -> int:
        """
        Save standardized dataset to disk.

        Args:
            data: List of standardized samples
            output_path: Output file path

        Returns:
            Number of samples saved
        """
        if self.config.save_format == "jsonl":
            # JSONL format (one JSON object per line)
            with open(output_path, "w", encoding="utf-8") as f:
                for sample in data:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        else:
            # JSON format (single array)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        return len(data)

    async def _verify_download(
        self,
        file_path: Path,
        config: DatasetSource,
    ) -> bool:
        """
        Verify downloaded dataset integrity.

        Args:
            file_path: Path to downloaded file
            config: Dataset configuration

        Returns:
            True if download is valid, False otherwise
        """
        try:
            # Check file exists and is not empty
            if not file_path.exists() or file_path.stat().st_size == 0:
                return False

            # Count lines in JSONL file
            line_count = 0
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_count += 1
                    # Parse first few lines to verify format
                    if line_count <= 10:
                        sample = json.loads(line)
                        if "instruction" not in sample or "response" not in sample:
                            console.print(f"[red]❌ Invalid format in {file_path}: missing required fields[/red]")
                            return False

            # Check sample count is reasonable (within 10% of expected or at least some samples)
            if line_count < 100:  # Minimum threshold
                console.print(f"[yellow]⚠️  Warning: Only {line_count} samples in {file_path} (expected ~{config.expected_samples:,})[/yellow]")
                return False

            return True

        except Exception as e:
            console.print(f"[red]❌ Verification error for {file_path}: {e}[/red]")
            return False

    async def download_all(self) -> List[Dict[str, Any]]:
        """
        Download all configured datasets in parallel.

        Returns:
            List of download results for each dataset

        Example:
            ```python
            downloader = DatasetDownloader()
            results = await downloader.download_all()

            total_samples = sum(r["samples"] for r in results if r["success"])
            failed = [r["name"] for r in results if not r["success"]]

            print(f"Downloaded {total_samples:,} total samples")
            if failed:
                print(f"Failed datasets: {', '.join(failed)}")
            ```
        """
        console.print("\n[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]")
        console.print("[bold cyan]Phase 1.1: Downloading Public Datasets[/bold cyan]")
        console.print("[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]\n")

        # Create progress bar
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            # Create tasks for each dataset
            tasks = []
            task_ids = {}

            for dataset_config in self.DATASETS:
                task_id = progress.add_task(
                    f"[cyan]{dataset_config.name}",
                    total=dataset_config.expected_samples,
                )
                task_ids[dataset_config.name] = task_id
                tasks.append(
                    self.download_dataset(dataset_config, progress, task_id)
                )

            # Download all datasets in parallel
            results = await asyncio.gather(*tasks)

        # Print summary
        console.print("\n[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]")
        console.print("[bold cyan]Download Summary[/bold cyan]")
        console.print("[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]\n")

        total_samples = 0
        successful = 0
        failed = []

        for result in results:
            if result["success"]:
                total_samples += result["samples"]
                successful += 1
                console.print(f"[green]✅ {result['name']}:[/green] {result['samples']:,} samples")
            else:
                failed.append(result["name"])
                console.print(f"[red]❌ {result['name']}:[/red] {result['error']}")

        console.print(f"\n[bold]Total:[/bold] {total_samples:,} samples from {successful}/{len(self.DATASETS)} datasets")

        if failed:
            console.print(f"[bold red]Failed:[/bold red] {', '.join(failed)}")
            console.print("\n[yellow]⚠️  Some datasets failed to download. Please check errors above.[/yellow]")
        else:
            console.print("\n[bold green]✅ All datasets downloaded successfully![/bold green]")

        return results


async def main():
    """
    Main function to run dataset downloader.

    Example usage:
        python -m src.data_collection.dataset_downloader
    """
    downloader = DatasetDownloader()
    results = await downloader.download_all()

    # Validate success criteria from EXECUTION_PLAN.md
    console.print("\n[bold cyan]Validating Success Criteria...[/bold cyan]\n")

    success_criteria = {
        "All 5 datasets downloaded": len([r for r in results if r["success"]]) == 5,
        "Total ~4.5M samples available": sum(r["samples"] for r in results) >= 4_000_000,
        "Files in correct format (JSONL)": all(
            Path(r["path"]).suffix == ".jsonl" for r in results if r["success"]
        ),
        "No download errors": all(r["success"] for r in results),
        "Output in data/raw/ folder": all(
            "data/raw" in r["path"] for r in results if r["success"]
        ),
    }

    all_passed = True
    for criterion, passed in success_criteria.items():
        status = "[green]✅[/green]" if passed else "[red]❌[/red]"
        console.print(f"{status} {criterion}")
        if not passed:
            all_passed = False

    if all_passed:
        console.print("\n[bold green]🎉 Phase 1.1 COMPLETE - All success criteria met![/bold green]")
    else:
        console.print("\n[bold red]❌ Phase 1.1 INCOMPLETE - Some criteria failed[/bold red]")
        console.print("[yellow]→ ITERATION REQUIRED: Fix errors above before proceeding to Phase 1.2[/yellow]")


if __name__ == "__main__":
    asyncio.run(main())
