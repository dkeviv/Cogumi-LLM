#!/usr/bin/env python3
"""
# Context: Phase 1B Utility - Claude Desktop chunking helper

Purpose:
- Split large paired JSONL files (test dataset and corresponding model outputs)
  into matched chunks suitable for Claude Desktop's ~31MB per-file limit.

Key features:
- Two split modes: by-count (N examples per chunk) or by-size (<= MB per file)
- Validates that test and model files have equal line counts (strict mode),
  with option to truncate to min length if desired
- Optional sampled ID consistency check when lines are JSON objects with an "id" field
- Streaming implementation (no full-file load), safe partial writes, and summary JSON
- Typed function signatures, clear docstrings, and basic progress logging

Usage examples:
- By count:  python split_for_claude.py --test ./test.jsonl --model ./model.jsonl \
             --out ./chunks --mode by-count --examples 6500
- By size:   python split_for_claude.py --test ./test.jsonl --model ./model.jsonl \
             --out ./chunks --mode by-size --max-mb 31

Notes:
- For very large files (>100K lines), the operation logs progress every 10K lines.
- Summary written to: {out}/split_summary.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Generator, Iterable, List, Optional, Tuple

import typer


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)


DEFAULT_MAX_MB = 31.0
PROGRESS_LOG_EVERY = 10_000


@dataclass
class ChunkInfo:
    """Metadata for a single chunk pair."""

    index: int
    start_line: int
    end_line_inclusive: int
    num_examples: int
    test_file: str
    model_file: str
    test_mb: float
    model_mb: float
    size_warning: bool


@dataclass
class SplitSummary:
    """Aggregate summary for the entire split run."""

    mode: str
    total_test_lines: int
    total_model_lines: int
    strict: bool
    truncated_to_min: bool
    examples_per_chunk: Optional[int]
    max_mb: Optional[float]
    chunks: List[ChunkInfo]


def _count_lines(file_path: Path) -> int:
    """Return total number of lines in a file."""

    with file_path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _iter_lines(file_path: Path) -> Generator[str, None, None]:
    """Yield lines from a file with UTF-8 decoding."""

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            yield line


def _maybe_parse_id(line: str) -> Optional[str]:
    """Try to parse a JSON line and extract an 'id' value, if present."""

    try:
        obj = json.loads(line)
        if isinstance(obj, dict) and "id" in obj:
            return str(obj["id"])
    except Exception:
        return None
    return None


def _bytes_of(line: str) -> int:
    return len(line.encode("utf-8"))


def split_by_count(
    test_path: Path,
    model_path: Path,
    output_dir: Path,
    examples_per_chunk: int,
    max_mb: float = DEFAULT_MAX_MB,
    strict: bool = True,
    id_sample_every: Optional[int] = 5000,
) -> SplitSummary:
    """Split paired files by fixed number of examples per chunk.

    Args:
        test_path: Path to test dataset JSONL.
        model_path: Path to model outputs JSONL.
        output_dir: Directory to write chunk pairs.
        examples_per_chunk: Target examples per chunk file.
        max_mb: Warning threshold per file in MB.
        strict: If True, require equal line counts; else truncate to min.
        id_sample_every: If provided, sample every N lines to compare 'id' fields.

    Returns:
        SplitSummary with per-chunk metadata.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    test_total = _count_lines(test_path)
    model_total = _count_lines(model_path)

    truncated_to_min = False
    if strict and test_total != model_total:
        raise ValueError(
            f"Line count mismatch: test={test_total} vs model={model_total}. "
            f"Use --strict false to truncate to min length."
        )
    if not strict and test_total != model_total:
        truncated_to_min = True
        total = min(test_total, model_total)
        logging.warning(
            "Line count mismatch (test=%d, model=%d). Truncating to %d lines.",
            test_total,
            model_total,
            total,
        )
    else:
        total = test_total

    chunks: List[ChunkInfo] = []
    num_chunks = (total + examples_per_chunk - 1) // examples_per_chunk

    test_iter = _iter_lines(test_path)
    model_iter = _iter_lines(model_path)

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * examples_per_chunk
        end_idx = min(start_idx + examples_per_chunk, total)

        test_chunk_path = output_dir / f"test_dataset_part{chunk_idx + 1}.jsonl"
        model_chunk_path = output_dir / f"model_outputs_part{chunk_idx + 1}.jsonl"

        test_bytes = 0
        model_bytes = 0
        written = 0

        with test_chunk_path.open("w", encoding="utf-8") as tf, model_chunk_path.open(
            "w", encoding="utf-8"
        ) as mf:
            for i in range(start_idx, end_idx):
                try:
                    t_line = next(test_iter)
                    m_line = next(model_iter)
                except StopIteration:
                    break

                if id_sample_every and (i % id_sample_every == 0):
                    t_id = _maybe_parse_id(t_line)
                    m_id = _maybe_parse_id(m_line)
                    if t_id is not None and m_id is not None and t_id != m_id:
                        logging.warning(
                            "ID mismatch at line %d: test id=%s, model id=%s",
                            i,
                            t_id,
                            m_id,
                        )

                tf.write(t_line)
                mf.write(m_line)
                test_bytes += _bytes_of(t_line)
                model_bytes += _bytes_of(m_line)
                written += 1

                if i and (i % PROGRESS_LOG_EVERY == 0):
                    logging.info("Processed %d/%d lines...", i, total)

        test_mb = test_bytes / (1024 * 1024)
        model_mb = model_bytes / (1024 * 1024)
        size_warning = test_mb > max_mb or model_mb > max_mb

        chunks.append(
            ChunkInfo(
                index=chunk_idx + 1,
                start_line=start_idx,
                end_line_inclusive=end_idx - 1,
                num_examples=written,
                test_file=test_chunk_path.name,
                model_file=model_chunk_path.name,
                test_mb=test_mb,
                model_mb=model_mb,
                size_warning=size_warning,
            )
        )

        print(
            f"\nChunk {chunk_idx + 1}:\n"
            f"  Examples: {start_idx} to {end_idx - 1} ({written} total)\n"
            f"  Test file: {test_chunk_path.name} ({test_mb:.1f} MB)\n"
            f"  Model file: {model_chunk_path.name} ({model_mb:.1f} MB)\n"
            + ("  ⚠️  WARNING: File(s) exceed 31MB limit!\n" if size_warning else "")
        )

    return SplitSummary(
        mode="by-count",
        total_test_lines=test_total,
        total_model_lines=model_total,
        strict=strict,
        truncated_to_min=truncated_to_min,
        examples_per_chunk=examples_per_chunk,
        max_mb=max_mb,
        chunks=chunks,
    )


def split_by_size(
    test_path: Path,
    model_path: Path,
    output_dir: Path,
    max_mb: float = DEFAULT_MAX_MB,
    strict: bool = True,
    id_sample_every: Optional[int] = 5000,
) -> SplitSummary:
    """Split paired files to ensure each resulting file stays under `max_mb`.

    Chunking rolls over when adding the next example would exceed the threshold
    for either the test or model chunk. Ensures paired chunk boundaries.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    test_total = _count_lines(test_path)
    model_total = _count_lines(model_path)

    truncated_to_min = False
    if strict and test_total != model_total:
        raise ValueError(
            f"Line count mismatch: test={test_total} vs model={model_total}. "
            f"Use --strict false to truncate to min length."
        )
    total = test_total
    if not strict and test_total != model_total:
        truncated_to_min = True
        total = min(test_total, model_total)
        logging.warning(
            "Line count mismatch (test=%d, model=%d). Truncating to %d lines.",
            test_total,
            model_total,
            total,
        )

    chunks: List[ChunkInfo] = []
    test_iter = _iter_lines(test_path)
    model_iter = _iter_lines(model_path)

    chunk_idx = 0
    processed = 0
    while processed < total:
        chunk_idx += 1
        start_idx = processed

        test_chunk_path = output_dir / f"test_dataset_part{chunk_idx}.jsonl"
        model_chunk_path = output_dir / f"model_outputs_part{chunk_idx}.jsonl"
        test_bytes = 0
        model_bytes = 0
        written = 0

        with test_chunk_path.open("w", encoding="utf-8") as tf, model_chunk_path.open(
            "w", encoding="utf-8"
        ) as mf:
            while processed < total:
                try:
                    t_line = next(test_iter)
                    m_line = next(model_iter)
                except StopIteration:
                    break

                t_b = _bytes_of(t_line)
                m_b = _bytes_of(m_line)
                next_test_mb = (test_bytes + t_b) / (1024 * 1024)
                next_model_mb = (model_bytes + m_b) / (1024 * 1024)

                # If adding this pair exceeds threshold in either file, roll to next chunk
                if (written > 0) and (next_test_mb > max_mb or next_model_mb > max_mb):
                    break

                if id_sample_every and (processed % id_sample_every == 0):
                    t_id = _maybe_parse_id(t_line)
                    m_id = _maybe_parse_id(m_line)
                    if t_id is not None and m_id is not None and t_id != m_id:
                        logging.warning(
                            "ID mismatch at line %d: test id=%s, model id=%s",
                            processed,
                            t_id,
                            m_id,
                        )

                tf.write(t_line)
                mf.write(m_line)
                test_bytes += t_b
                model_bytes += m_b
                written += 1
                processed += 1

                if processed and (processed % PROGRESS_LOG_EVERY == 0):
                    logging.info("Processed %d/%d lines...", processed, total)

        test_mb = test_bytes / (1024 * 1024)
        model_mb = model_bytes / (1024 * 1024)
        size_warning = test_mb > max_mb or model_mb > max_mb

        chunks.append(
            ChunkInfo(
                index=chunk_idx,
                start_line=start_idx,
                end_line_inclusive=processed - 1,
                num_examples=written,
                test_file=test_chunk_path.name,
                model_file=model_chunk_path.name,
                test_mb=test_mb,
                model_mb=model_mb,
                size_warning=size_warning,
            )
        )

        print(
            f"\nChunk {chunk_idx}:\n"
            f"  Examples: {start_idx} to {processed - 1} ({written} total)\n"
            f"  Test file: {test_chunk_path.name} ({test_mb:.1f} MB)\n"
            f"  Model file: {model_chunk_path.name} ({model_mb:.1f} MB)\n"
            + ("  ⚠️  WARNING: File(s) exceed 31MB limit!\n" if size_warning else "")
        )

    return SplitSummary(
        mode="by-size",
        total_test_lines=test_total,
        total_model_lines=model_total,
        strict=strict,
        truncated_to_min=truncated_to_min,
        examples_per_chunk=None,
        max_mb=max_mb,
        chunks=chunks,
    )


def _write_summary(summary: SplitSummary, output_dir: Path) -> Path:
    """Write a JSON summary to the output directory and return its path."""

    data = asdict(summary)
    # Convert dataclasses in list properly (already handled by asdict)
    summary_path = output_dir / "split_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return summary_path


def main(
    test: str = typer.Option(..., "--test", help="Path to test dataset JSONL"),
    model: str = typer.Option(..., "--model", help="Path to model outputs JSONL"),
    out: str = typer.Option(..., "--out", help="Output directory for chunks"),
    mode: str = typer.Option(
        "by-count", "--mode", help="Split strategy: 'by-count' or 'by-size'"
    ),
    examples: int = typer.Option(
        6667, "--examples", min=1, help="Examples per chunk (by-count mode)"
    ),
    max_mb: float = typer.Option(
        DEFAULT_MAX_MB, "--max-mb", help="Max MB per file (by-size mode + warnings)"
    ),
    strict: bool = typer.Option(
        True, "--strict/--no-strict", help="Require equal line counts"
    ),
    id_sample_every: Optional[int] = typer.Option(
        5000, "--id-sample-every", help="Sample frequency for 'id' consistency check"
    ),
) -> None:
    """CLI entrypoint for splitting paired JSONL files for Claude Desktop uploads."""

    test_path = Path(test)
    model_path = Path(model)
    output_dir = Path(out)

    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if mode not in {"by-count", "by-size"}:
        raise ValueError("--mode must be 'by-count' or 'by-size'")

    if mode == "by-count":
        summary = split_by_count(
            test_path,
            model_path,
            output_dir,
            examples_per_chunk=examples,
            max_mb=max_mb,
            strict=strict,
            id_sample_every=id_sample_every,
        )
    else:
        summary = split_by_size(
            test_path,
            model_path,
            output_dir,
            max_mb=max_mb,
            strict=strict,
            id_sample_every=id_sample_every,
        )

    summary_path = _write_summary(summary, output_dir)

    print(
        f"\n✅ Created {len(summary.chunks)} matched chunk pairs in {output_dir}\n"
        f"Summary: {summary_path}"
    )
    print("\nUpload to Claude Desktop in order:")
    for c in summary.chunks:
        print(
            f"  Batch {c.index}: {c.test_file} + {c.model_file} "
            f"({c.num_examples} examples)"
        )


if __name__ == "__main__":
    typer.run(main)
