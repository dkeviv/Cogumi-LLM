#!/usr/bin/env python3
"""Fix script to download WizardLM and ShareGPT datasets."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collection.dataset_downloader import (
    DatasetDownloader,
    DatasetSource,
    DownloadConfig,
)
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    DownloadColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

console = Console()

async def download_missing():
    console.print("\n[bold cyan]Downloading Missing Datasets[/bold cyan]\n")
    
    config = DownloadConfig(output_dir=Path("data/raw"), verify_downloads=True)
    downloader = DatasetDownloader(config)
    
    datasets_to_download = [ds for ds in downloader.DATASETS if ds.name in ["WizardLM", "ShareGPT"]]
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        tasks = []
        for dataset_config in datasets_to_download:
            task_id = progress.add_task(f"[cyan]{dataset_config.name}", total=dataset_config.expected_samples)
            tasks.append(downloader.download_dataset(dataset_config, progress, task_id))
        
        results = await asyncio.gather(*tasks)
    
    console.print("\n[bold cyan]Summary[/bold cyan]\n")
    for result in results:
        if result["success"]:
            console.print(f"[green]✅ {result['name']}:[/green] {result['samples']:,} samples")
        else:
            console.print(f"[red]❌ {result['name']}:[/red] {result['error']}")

if __name__ == "__main__":
    asyncio.run(download_missing())
