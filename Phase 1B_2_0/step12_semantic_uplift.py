#!/usr/bin/env python3
"""
Context: Phase 1C Hybrid - Semantic uplift measurement on self-critique outputs

Purpose:
- Estimate uplift by comparing baseline (previous_output) vs improved (final_answer)
  against the reference using the same local semantic evaluator as step10.

Inputs:
- --source_jsonl: JSONL from step9 containing: id, instruction, reference_answer, previous_output, final_answer, category

Outputs (in --output_dir):
- baseline_eval.jsonl: per-item similarity and pass for previous_output
- improved_eval.jsonl: per-item similarity and pass for final_answer
- summary.json: baseline vs improved pass rates and absolute/relative uplift

Usage:
  python "Phase 1B_2_0/step12_semantic_uplift.py" \
    --source_jsonl "Phase 1B_2_0/data/self_critique/rewrite.jsonl" \
    --output_dir "Phase 1B_2_0/data/self_critique/semantic_uplift" \
    --threshold 0.74 --limit 200 --batch_size 64

Notes:
- This is a proxy metric; authoritative LLM replay cannot score new outputs post-rewrite.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from loguru import logger
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from sentence_transformers import SentenceTransformer


@dataclass
class UpliftCfg:
    source_jsonl: str
    output_dir: str
    threshold: float = 0.74
    limit: int | None = None
    batch_size: int = 64
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def batched(items: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-12)
    b = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-12)
    return (a * b).sum(axis=1)


def evaluate_semantic_uplift(cfg: UpliftCfg) -> Tuple[float, float, float, int]:
    os.makedirs(cfg.output_dir, exist_ok=True)
    records = []
    for rec in iter_jsonl(cfg.source_jsonl):
        if rec.get("reference_answer") is None:
            continue
        if rec.get("previous_output") is None or rec.get("final_answer") is None:
            continue
        records.append(rec)
        if cfg.limit and len(records) >= cfg.limit:
            break

    count = len(records)
    if count == 0:
        logger.warning("No valid records found for semantic uplift.")
        with open(Path(cfg.output_dir) / "summary.json", "w", encoding="utf-8") as f:
            json.dump({"count": 0, "baseline": 0, "improved": 0, "uplift_abs": 0, "threshold": cfg.threshold}, f, indent=2)
        return 0.0, 0.0, 0.0, 0

    model = SentenceTransformer(cfg.model_name)
    refs = [r.get("reference_answer", "") for r in records]
    prevs = [r.get("previous_output", "") for r in records]
    finals = [r.get("final_answer", "") for r in records]

    with Progress(
        TextColumn("[bold blue]Encoding baseline/improved/refs"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
    ) as progress:
        t_prev = progress.add_task("prev", total=len(prevs))
        t_fin = progress.add_task("final", total=len(finals))
        t_ref = progress.add_task("ref", total=len(refs))

        prev_embs: List[np.ndarray] = []
        for ch in batched(prevs, cfg.batch_size):
            prev_embs.extend(model.encode(ch, batch_size=min(cfg.batch_size, len(ch)), show_progress_bar=False))
            progress.advance(t_prev, advance=len(ch))

        fin_embs: List[np.ndarray] = []
        for ch in batched(finals, cfg.batch_size):
            fin_embs.extend(model.encode(ch, batch_size=min(cfg.batch_size, len(ch)), show_progress_bar=False))
            progress.advance(t_fin, advance=len(ch))

        ref_embs: List[np.ndarray] = []
        for ch in batched(refs, cfg.batch_size):
            ref_embs.extend(model.encode(ch, batch_size=min(cfg.batch_size, len(ch)), show_progress_bar=False))
            progress.advance(t_ref, advance=len(ch))

    prev_arr = np.vstack(prev_embs)
    fin_arr = np.vstack(fin_embs)
    ref_arr = np.vstack(ref_embs)

    prev_sims = cosine_sim(prev_arr, ref_arr)
    fin_sims = cosine_sim(fin_arr, ref_arr)

    baseline_pass = int((prev_sims >= cfg.threshold).sum())
    improved_pass = int((fin_sims >= cfg.threshold).sum())
    baseline_rate = baseline_pass / count
    improved_rate = improved_pass / count
    uplift_abs = improved_rate - baseline_rate

    # Write eval files
    with open(Path(cfg.output_dir) / "baseline_eval.jsonl", "w", encoding="utf-8") as f:
        for r, s in zip(records, prev_sims.tolist()):
            f.write(json.dumps({
                "id": r.get("id"),
                "similarity": float(s),
                "pass": bool(s >= cfg.threshold),
                "threshold": cfg.threshold,
                "method": "local_semantic"
            }, ensure_ascii=False) + "\n")

    with open(Path(cfg.output_dir) / "improved_eval.jsonl", "w", encoding="utf-8") as f:
        for r, s in zip(records, fin_sims.tolist()):
            f.write(json.dumps({
                "id": r.get("id"),
                "similarity": float(s),
                "pass": bool(s >= cfg.threshold),
                "threshold": cfg.threshold,
                "method": "local_semantic"
            }, ensure_ascii=False) + "\n")

    summary = {
        "count": count,
        "baseline_pass": baseline_pass,
        "baseline_pass_rate": round(baseline_rate * 100, 2),
        "improved_pass": improved_pass,
        "improved_pass_rate": round(improved_rate * 100, 2),
        "uplift_absolute": round(uplift_abs * 100, 2),
        "threshold": cfg.threshold,
        "model": cfg.model_name,
    }
    with open(Path(cfg.output_dir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.success(
        "Semantic uplift → baseline={:.2f}% | improved={:.2f}% | Δ={:.2f}% on n={}".format(
            baseline_rate * 100, improved_rate * 100, uplift_abs * 100, count
        )
    )
    return baseline_rate, improved_rate, uplift_abs, count


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure semantic uplift for self-critique outputs")
    parser.add_argument("--source_jsonl", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--threshold", type=float, default=0.74)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    cfg = UpliftCfg(
        source_jsonl=args.source_jsonl,
        output_dir=args.output_dir,
        threshold=args.threshold,
        limit=args.limit,
        batch_size=args.batch_size,
        model_name=args.model_name,
    )
    evaluate_semantic_uplift(cfg)


if __name__ == "__main__":
    main()
