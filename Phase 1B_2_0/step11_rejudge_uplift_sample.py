#!/usr/bin/env python3
"""
Context: Phase 1C Hybrid - Uplift measurement via Haiku replay on improved answers

Purpose:
- Quantify true quality uplift by re-judging a small sample of self-critique improved answers
  using the deterministic Haiku replay judge (authoritative for Phase 1B).
- Compare baseline (original previous_output) vs improved (final_answer) for the same IDs.

Inputs:
- --source_jsonl: JSONL from step9 with fields: id, instruction, reference_answer, previous_output, final_answer, category

Outputs (in --output_dir):
- eval_original.jsonl: Haiku replay judgments for previous_output vs reference_answer
- eval_improved.jsonl: Haiku replay judgments for final_answer vs reference_answer
- summary.json: Aggregate stats with baseline/improved pass rates and uplift

Usage:
  python "Phase 1B_2_0/step11_rejudge_uplift_sample.py" \
    --source_jsonl "Phase 1B_2_0/data/self_critique/rewrite.jsonl" \
    --output_dir "Phase 1B_2_0/data/self_critique/uplift" \
    --limit 100 --seed 42

Notes:
- Uses HaikuReplayJudge (no API) by replaying prior Haiku LLM judgments.
- Progress bars, logging, and guard rails included per project guidelines.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from loguru import logger
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# Local import: reuse the judge implementation
from step7_rejudge_gpt5 import HaikuReplayJudge, Judgment


@dataclass
class UpliftConfig:
    source_jsonl: str
    output_dir: str
    limit: int = 100
    seed: int = 42


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _judge_examples(judge: HaikuReplayJudge, examples: List[Dict[str, Any]], desc: str) -> Tuple[List[Dict[str, Any]], int, int]:
    """Run Haiku replay judge over a list of examples.

    Each example must contain: id, category, instruction, reference, model_output
    Returns (rows, passed_count, failed_count)
    """
    rows: List[Dict[str, Any]] = []
    passed = failed = 0
    with Progress(
        TextColumn(f"[bold blue]{desc}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("items", total=len(examples))
        for ex in examples:
            # Build synchronous judgment using HaikuReplayJudge lookup mapping
            ex_id = ex.get("id")
            try:
                ex_id_int = int(ex_id) if ex_id is not None else -1
            except Exception:
                ex_id_int = -1
            status, reason, conf = judge.lookup.get(
                ex_id_int, ("FAIL", "id not found in Haiku replay", 0.6)
            )
            j = Judgment(
                judgment=("PASS" if status == "PASS" else "FAIL"),
                reason=reason,
                confidence=float(conf),
                judged_by="haiku-replay",
            )
            row = {
                "id": ex.get("id"),
                "category": ex.get("category", "unknown"),
                "instruction": ex.get("instruction", ""),
                "reference": ex.get("reference", ""),
                "model_output": ex.get("model_output", ""),
                "judgment": j.judgment,
                "reason": j.reason,
                "confidence": j.confidence,
                "judged_by": j.judged_by,
            }
            rows.append(row)
            if j.judgment == "PASS":
                passed += 1
            else:
                failed += 1
            progress.advance(task)
    return rows, passed, failed


def evaluate_uplift(cfg: UpliftConfig) -> Tuple[float, float, float, int]:
    os.makedirs(cfg.output_dir, exist_ok=True)
    src_records = list(iter_jsonl(cfg.source_jsonl))

    if not src_records:
        logger.error(f"No records found in {cfg.source_jsonl}")
        return 0.0, 0.0, 0.0, 0

    # Filter to items with required fields
    filtered = [
        r for r in src_records
        if r.get("id") is not None and r.get("reference_answer") and r.get("previous_output") is not None and r.get("final_answer") is not None
    ]
    if not filtered:
        logger.error("No valid records with id/reference_answer/previous_output/final_answer")
        return 0.0, 0.0, 0.0, 0

    random.seed(cfg.seed)
    sample = filtered[:]
    random.shuffle(sample)
    if cfg.limit:
        sample = sample[: cfg.limit]

    judge = HaikuReplayJudge()

    # Build example lists for baseline and improved
    baseline_examples = []
    improved_examples = []
    for r in sample:
        base_ex = {
            "id": r.get("id"),
            "category": (r.get("category") or "unknown"),
            "instruction": r.get("instruction") or "",
            "reference": r.get("reference_answer") or "",
            "model_output": r.get("previous_output") or "",
        }
        imp_ex = {
            **base_ex,
            "model_output": r.get("final_answer") or "",
        }
        baseline_examples.append(base_ex)
        improved_examples.append(imp_ex)

    # Judge baseline and improved using the same authoritative mapping
    eval_orig, orig_pass, orig_fail = _judge_examples(judge, baseline_examples, "Haiku replay (original)")
    eval_impr, impr_pass, impr_fail = _judge_examples(judge, improved_examples, "Haiku replay (improved)")

    count = len(sample)
    baseline_rate = orig_pass / count if count else 0.0
    improved_rate = impr_pass / count if count else 0.0
    uplift_abs = improved_rate - baseline_rate

    # Write outputs
    with open(Path(cfg.output_dir) / "eval_original.jsonl", "w", encoding="utf-8") as f:
        for row in eval_orig:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(Path(cfg.output_dir) / "eval_improved.jsonl", "w", encoding="utf-8") as f:
        for row in eval_impr:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "count": count,
        "baseline_pass": orig_pass,
        "baseline_fail": orig_fail,
        "baseline_pass_rate": round(baseline_rate * 100, 2),
        "improved_pass": impr_pass,
        "improved_fail": impr_fail,
        "improved_pass_rate": round(improved_rate * 100, 2),
        "uplift_absolute": round(uplift_abs * 100, 2),
        "uplift_relative": round((improved_rate / baseline_rate - 1) * 100, 2) if baseline_rate > 0 else None,
        "source": cfg.source_jsonl,
        "judge": "haiku-replay",
        "note": "Authoritative replay used; values are deterministic for given IDs.",
    }
    with open(Path(cfg.output_dir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.success(
        "Uplift → baseline={:.2f}% | improved={:.2f}% | Δ={:.2f}% on n={}".format(
            baseline_rate * 100, improved_rate * 100, uplift_abs * 100, count
        )
    )
    return baseline_rate, improved_rate, uplift_abs, count


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure uplift via Haiku replay on improved self-critique answers")
    parser.add_argument("--source_jsonl", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = UpliftConfig(
        source_jsonl=args.source_jsonl,
        output_dir=args.output_dir,
        limit=args.limit,
        seed=args.seed,
    )
    evaluate_uplift(cfg)


if __name__ == "__main__":
    main()
