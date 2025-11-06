"""
Context: Phase 1B utility to export FAIL cases from GPT5judged aggregate for Phase 1C.

This script reads the aggregated judgments produced by `step7_rejudge_gpt5.py`
from `data/GPT5judged/GPT5judged_all.jsonl`, filters out the items with
judgment == "FAIL", and writes a Phase 1C-ready JSONL containing the fields:

- id, category, instruction, reference_answer, model_output, judged_by, reason

It also emits a small stats JSON and an optional sample file for quick review.

Usage:
  python step8_export_gpt5judged_failures.py \
    --agg "Phase 1B_2_0/data/GPT5judged/GPT5judged_all.jsonl" \
    --out "Phase 1B_2_0/data/GPT5judged/phase1c_failures_gpt5judged.jsonl" \
    --stats "Phase 1B_2_0/data/GPT5judged/phase1c_failures_stats.json" \
    --sample "Phase 1B_2_0/data/GPT5judged/phase1c_failures_sample.jsonl" \
    --sample_size 100

Notes:
- Pure local I/O; no external APIs
- Includes basic validation and summary stats by category

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Iterable, List


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a .jsonl file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write an iterable of dicts to a .jsonl file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(agg_path: Path, out_path: Path, stats_path: Path | None, sample_path: Path | None, sample_size: int) -> None:
    """Export FAIL items from the GPT5judged aggregate to a Phase 1C-ready JSONL.

    Also produces optional stats and a small sample for quick inspection.
    """
    assert agg_path.exists(), f"Aggregate file not found: {agg_path}"

    failures: List[Dict[str, Any]] = []
    per_category: Dict[str, int] = {}

    for rec in iter_jsonl(agg_path):
        if str(rec.get("judgment", "")).upper() != "FAIL":
            continue
        category = (rec.get("category") or "other").lower()
        per_category[category] = per_category.get(category, 0) + 1

        failures.append(
            {
                "id": rec.get("id"),
                "category": category,
                "instruction": rec.get("instruction", ""),
                "reference_answer": rec.get("reference", ""),
                "model_output": rec.get("model_output", ""),
                "judged_by": rec.get("judged_by", "unknown"),
                "reason": rec.get("reason", ""),
            }
        )

    # Write failures
    write_jsonl(out_path, failures)

    # Optional stats
    if stats_path is not None:
        stats = {
            "total_failures": len(failures),
            "per_category": dict(sorted(per_category.items(), key=lambda x: x[0])),
            "source": str(agg_path),
            "output": str(out_path),
        }
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    # Optional sample
    if sample_path is not None:
        write_jsonl(sample_path, failures[: max(0, sample_size)])

    print("Export complete:")
    print(f"  Failures: {len(failures)} → {out_path}")
    if stats_path is not None:
        print(f"  Stats:    {stats_path}")
    if sample_path is not None:
        print(f"  Sample:   {sample_path} ({min(sample_size, len(failures))} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export FAIL cases from GPT5judged aggregate for Phase 1C")
    parser.add_argument("--agg", type=str, default="Phase 1B_2_0/data/GPT5judged/GPT5judged_all.jsonl")
    parser.add_argument("--out", type=str, default="Phase 1B_2_0/data/GPT5judged/phase1c_failures_gpt5judged.jsonl")
    parser.add_argument("--stats", type=str, default="Phase 1B_2_0/data/GPT5judged/phase1c_failures_stats.json")
    parser.add_argument("--sample", type=str, default="Phase 1B_2_0/data/GPT5judged/phase1c_failures_sample.jsonl")
    parser.add_argument("--sample_size", type=int, default=100)
    args = parser.parse_args()

    main(
        agg_path=Path(args.agg),
        out_path=Path(args.out),
        stats_path=Path(args.stats) if args.stats else None,
        sample_path=Path(args.sample) if args.sample else None,
        sample_size=int(args.sample_size),
    )
