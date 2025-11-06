"""
⚠️ ARCHIVED - DO NOT USE ⚠️
================================

**Status**: DEPRECATED
**Reason**: Old implementation superseded by src/phase0_dataset/
**Use Instead**: src/phase0_dataset/ (complete and tested)

See: archive_old_src/README_ARCHIVE.md for details

---

ORIGINAL DOCSTRING (for historical reference):

Context: Data collection module for downloading and processing public datasets.

This module handles Phase 1 of the Cogumi-LLM pipeline:
- Download public instruction-tuning datasets
- Score and filter for quality
- Deduplicate and curate final 500K dataset

Public Datasets (NO teacher models, NO API calls):
- OpenOrca: 4.2M samples → select top 350K
- Alpaca-GPT4: 52K samples → select all
- WizardLM: 143K samples → select top 80K
- Dolly: 15K samples → select all
- ShareGPT: 90K samples → select top 50K
Total: ~547K → deduplicate to 500K unique

Cost: $0 (public datasets, no API calls)
Duration: 1.5-2 hours
Output: data/phase1/public_500k_filtered.jsonl
"""

from .dataset_downloader import DatasetDownloader, DownloadConfig
from .quality_scorer import QualityScorer, ScoringConfig
from .dataset_curator import DatasetCurator, CurationConfig

__all__ = [
    "DatasetDownloader",
    "DownloadConfig",
    "QualityScorer",
    "ScoringConfig",
    "DatasetCurator",
    "CurationConfig",
]
