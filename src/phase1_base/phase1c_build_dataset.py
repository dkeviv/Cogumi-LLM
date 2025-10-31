"""
Context: Cogumi-LLM Phase 1C Dataset Builder
Purpose: Convert authoritative Phase 1B FAIL items (GPT5judged haiku replay) into a targeted
         training dataset for Phase 1C distillation using available reference answers.

Notes:
- This bootstrap builder uses each item's reference_answer as the teacher output. It avoids
  external API calls and enables immediate test-mode training. Later, this file can be
  expanded to plug in GPT-5 enhanced answers when available.
- Adds progress bars for long operations, debug checkpoints, and a sample preview.
- Follows project conventions (async/await not needed here as no external I/O APIs are used).

Outputs:
- Primary: data/phase1c/targeted_failures_from_refs.jsonl (instruction/response pairs)
- Optional reverse pairs: data/phase1c/targeted_failures_reverse_from_refs.jsonl

Usage:
- Run via: python -m src.phase1_base.phase1c_build_dataset --help
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Tuple

from loguru import logger
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn


DEFAULT_INPUT_PATH = "Phase 1B_2_0/data/GPT5judged/phase1c_failures_gpt5judged.jsonl"
DEFAULT_OUTPUT_DIR = "data/phase1c"
DEFAULT_OUTPUT_FILE = "targeted_failures_from_refs.jsonl"
DEFAULT_OUTPUT_REVERSE_FILE = "targeted_failures_reverse_from_refs.jsonl"


@dataclass
class FailureItem:
    """Container for a single failure record from Phase 1B authoritative set.

    Attributes:
        id: Unique integer ID within the 20K evaluation set.
        category: Task type (e.g., 'code', 'math', 'qa', 'other').
        instruction: The original instruction/prompt text.
        reference_answer: The authoritative/ground-truth answer text.
        model_output: The student's model output that failed.
    """

    id: int
    category: str
    instruction: str
    reference_answer: str
    model_output: str


def read_jsonl(path: str) -> Iterator[dict]:
    """Stream-read a JSONL file line-by-line.

    Args:
        path: Absolute or relative path to a JSONL file.

    Yields:
        Parsed JSON objects per line.
    """

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def write_jsonl(records: Iterable[dict], path: str) -> None:
    """Write iterable of dicts to JSONL file, creating directory if needed.

    Args:
        records: Iterable of dictionaries to serialize.
        path: Destination file path.
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def to_failure_item(raw: dict) -> Optional[FailureItem]:
    """Convert a raw dict to FailureItem with guard rails.

    Returns None if any required field is missing or malformed.
    """

    required = ["id", "category", "instruction", "reference_answer", "model_output"]
    if any(k not in raw for k in required):
        return None
    try:
        return FailureItem(
            id=int(raw["id"]),
            category=str(raw["category"]),
            instruction=str(raw["instruction"]),
            reference_answer=str(raw["reference_answer"]),
            model_output=str(raw["model_output"]),
        )
    except Exception as exc:  # noqa: BLE001 - broad to guard dataset quirks
        logger.warning(f"Skipping record due to parse error: {exc}")
        return None


def build_forward_record(item: FailureItem) -> dict:
    """Create a forward training pair using the reference answer.

    Returns a standard instruction-tuned sample: {instruction, input, output}.
    """

    return {
        "instruction": item.instruction,
        "input": "",
        "output": item.reference_answer,
        "meta": {
            "source": "phase1c_failures_gpt5judged",
            "category": item.category,
            "id": item.id,
            "teacher": "reference_answer",
        },
    }


def build_reverse_record(item: FailureItem) -> dict:
    """Create a reverse training pair: given the answer, infer the question.

    This encourages bidirectional understanding as outlined in the plan.
    """

    reverse_instruction = (
        "Given the answer below, write a clear, self-contained question/instruction "
        "that this answer correctly addresses. Keep it faithful to the original task."
    )
    reverse_input = item.reference_answer
    reverse_output = item.instruction
    return {
        "instruction": reverse_instruction,
        "input": reverse_input,
        "output": reverse_output,
        "meta": {
            "source": "phase1c_failures_gpt5judged",
            "category": item.category,
            "id": item.id,
            "teacher": "reverse_from_reference",
        },
    }


def preview_samples(records: List[dict], n: int = 3) -> None:
    """Log a small sample for quick verification."""

    for i, rec in enumerate(records[:n]):
        logger.info(f"Sample {i+1}: instruction={rec['instruction'][:80]!r}… | output={rec['output'][:80]!r}…")


def build_datasets(
    input_path: str = DEFAULT_INPUT_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    make_reverse: bool = True,
    test_limit: Optional[int] = None,
) -> Tuple[str, Optional[str], int]:
    """Create forward (and optional reverse) datasets from authoritative failures.

    Args:
        input_path: Path to authoritative FAILS JSONL (contains reference_answer).
        output_dir: Directory to write output files.
        make_reverse: Whether to also generate reverse-pair dataset.
        test_limit: If provided, only process the first N items.

    Returns:
        Tuple of (forward_path, reverse_path_or_none, count_processed)
    """

    console = Console()
    forward_out = os.path.join(output_dir, DEFAULT_OUTPUT_FILE)
    reverse_out = os.path.join(output_dir, DEFAULT_OUTPUT_REVERSE_FILE) if make_reverse else None

    # Load and filter
    raw_iter = read_jsonl(input_path)
    items: List[FailureItem] = []
    for raw in raw_iter:
        item = to_failure_item(raw)
        if item is not None:
            items.append(item)
            if test_limit is not None and len(items) >= test_limit:
                break

    total = len(items)
    if total == 0:
        raise ValueError("No valid records found. Check input path or file contents.")

    forward_records: List[dict] = []
    reverse_records: List[dict] = []

    with Progress(
        TextColumn("[bold blue]Building Phase 1C dataset"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("processing", total=total)
        for idx, item in enumerate(items):
            forward_records.append(build_forward_record(item))
            if make_reverse:
                reverse_records.append(build_reverse_record(item))

            if idx % 1000 == 0 and idx > 0:
                logger.info(f"Progress: {idx}/{total} ({idx/total:.1%})")
            progress.advance(task)

    # Preview
    preview_samples(forward_records)
    if make_reverse:
        preview_samples(reverse_records)

    # Write
    write_jsonl(forward_records, forward_out)
    if make_reverse and reverse_out:
        write_jsonl(reverse_records, reverse_out)

    logger.success(
        f"Export complete → forward: {forward_out} | reverse: {reverse_out or 'N/A'} | count: {total}"
    )
    return forward_out, reverse_out, total


def main() -> None:
    """CLI entry point with simple argparse parameters.

    Supports test-mode via --limit to enable small, fast validation runs.
    """

    import argparse

    parser = argparse.ArgumentParser(description="Build Phase 1C targeted dataset from authoritative failures.")
    parser.add_argument("--input", dest="input_path", default=DEFAULT_INPUT_PATH, help="Path to failures JSONL")
    parser.add_argument("--out_dir", dest="output_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory path")
    parser.add_argument(
        "--no-reverse", dest="make_reverse", action="store_false", help="Disable generating reverse pairs"
    )
    parser.add_argument("--limit", dest="limit", type=int, default=None, help="Process only first N items for testing")

    args = parser.parse_args()

    logger.info("Starting Phase 1C dataset build…")
    logger.info(f"Input: {args.input_path}")
    logger.info(f"Output dir: {args.output_dir}")
    forward_path, reverse_path, count = build_datasets(
        input_path=args.input_path,
        output_dir=args.output_dir,
        make_reverse=args.make_reverse,
        test_limit=args.limit,
    )
    logger.info(f"Completed. Records processed: {count}")
    logger.info(f"Forward dataset: {forward_path}")
    if reverse_path:
        logger.info(f"Reverse dataset: {reverse_path}")


if __name__ == "__main__":
    main()
