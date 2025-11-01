#!/usr/bin/env python3
"""
Context: Phase 1C Hybrid - Local Evaluation of Self-Critique Rewrites
Purpose: Evaluate self-critique outputs against reference answers using semantic similarity
         to estimate quality uplift before escalating to teacher models.

Inputs:
- JSONL from step9 with fields: id, instruction, reference_answer, previous_output, critique, final_answer

Outputs (in --output_dir):
- eval.jsonl: Per-item results with similarity_score and pass flag
- summary.json: Aggregate stats (count, passes, fails, pass_rate, threshold)
- improved_forward.jsonl: Forward dataset (instruction/input/output/meta) for pass=True, aligned with Phase 1C builder
- improved_reverse.jsonl: Reverse dataset (reverse instruction/input/output/meta) using final_answer as input and original instruction as output

Usage:
  python "Phase 1B_2_0/step10_evaluate_self_critique_local.py" \
    --input_jsonl "Phase 1B_2_0/data/self_critique/rewrite.jsonl" \
    --output_dir "Phase 1B_2_0/data/self_critique/eval" \
    --threshold 0.74 --limit 500 --batch_size 64

Notes:
- Uses sentence-transformers (MiniLM) for similarity. Fast and local.
- This is a pilot evaluator; authoritative re-judge remains Haiku replay when applicable.
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
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from sentence_transformers import SentenceTransformer


@dataclass
class EvalConfig:
    input_jsonl: str
    output_dir: str
    threshold: float = 0.74
    limit: int | None = None
    batch_size: int = 64
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def batched(items: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-12)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-12)
    return (a_norm * b_norm).sum(axis=1)


def evaluate_self_critique(cfg: EvalConfig) -> Tuple[int, int, float]:
    os.makedirs(cfg.output_dir, exist_ok=True)
    eval_path = str(Path(cfg.output_dir) / "eval.jsonl")
    improved_forward_path = str(Path(cfg.output_dir) / "improved_forward.jsonl")
    improved_reverse_path = str(Path(cfg.output_dir) / "improved_reverse.jsonl")

    records: List[Dict[str, Any]] = []
    for i, rec in enumerate(iter_jsonl(cfg.input_jsonl)):
        records.append(rec)
        if cfg.limit and len(records) >= cfg.limit:
            break

    total = len(records)
    if total == 0:
        logger.warning("No records to evaluate.")
        with open(Path(cfg.output_dir) / "summary.json", "w", encoding="utf-8") as f:
            json.dump({"count": 0, "passes": 0, "fails": 0, "pass_rate": 0.0, "threshold": cfg.threshold}, f, indent=2)
        return 0, 0, 0.0

    logger.info(f"Loaded {total} self-critique records. Threshold={cfg.threshold}")

    model = SentenceTransformer(cfg.model_name)

    instructions = [r.get("instruction", "") for r in records]
    finals = [r.get("final_answer", "") for r in records]
    refs = [r.get("reference_answer", "") for r in records]

    # Combine to reduce encoder passes: encode finals and refs separately
    with Progress(
        TextColumn("[bold blue]Encoding finals/refs"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task_f = progress.add_task("finals", total=len(finals))
        task_r = progress.add_task("refs", total=len(refs))

        final_embs: List[np.ndarray] = []
        for chunk in batched(finals, cfg.batch_size):
            embs = model.encode(chunk, batch_size=min(cfg.batch_size, len(chunk)), normalize_embeddings=False, show_progress_bar=False)
            final_embs.extend(embs)
            progress.advance(task_f, advance=len(chunk))

        ref_embs: List[np.ndarray] = []
        for chunk in batched(refs, cfg.batch_size):
            embs = model.encode(chunk, batch_size=min(cfg.batch_size, len(chunk)), normalize_embeddings=False, show_progress_bar=False)
            ref_embs.extend(embs)
            progress.advance(task_r, advance=len(chunk))

    final_arr = np.vstack(final_embs)
    ref_arr = np.vstack(ref_embs)

    sims = cosine_sim_matrix(final_arr, ref_arr)
    passes_mask = sims >= cfg.threshold

    passes = int(passes_mask.sum())
    fails = int(total - passes)
    pass_rate = passes / total if total else 0.0

    with open(eval_path, "w", encoding="utf-8") as eval_f, \
        open(improved_forward_path, "w", encoding="utf-8") as fwd_f, \
        open(improved_reverse_path, "w", encoding="utf-8") as rev_f:
        for rec, sim, is_pass in zip(records, sims.tolist(), passes_mask.tolist()):
            out = {
                "id": rec.get("id"),
                "category": rec.get("category", "unknown"),
                "instruction": rec.get("instruction", ""),
                "reference_answer": rec.get("reference_answer", ""),
                "previous_output": rec.get("previous_output", ""),
                "final_answer": rec.get("final_answer", ""),
                "similarity_score": float(sim),
                "pass": bool(is_pass),
                "method": "local_semantic",
                "threshold": cfg.threshold,
            }
            eval_f.write(json.dumps(out, ensure_ascii=False) + "\n")

            if is_pass and out["final_answer"]:
                # Emit forward record matching Phase 1C builder schema
                fwd = {
                    "instruction": out["instruction"],
                    "input": "",
                    "output": out["final_answer"],
                    "meta": {
                        "source": "self_critique",
                        "category": out["category"],
                        "teacher": "self_critique",
                        "quality": {
                            "evaluator": "local_semantic",
                            "similarity": out["similarity_score"],
                            "threshold": cfg.threshold,
                        },
                    },
                }
                fwd_f.write(json.dumps(fwd, ensure_ascii=False) + "\n")

                # Emit reverse record: given final_answer, infer original instruction
                reverse_instruction = (
                    "Given the answer below, write a clear, self-contained question/instruction "
                    "that this answer correctly addresses. Keep it faithful to the original task."
                )
                rev = {
                    "instruction": reverse_instruction,
                    "input": out["final_answer"],
                    "output": out["instruction"],
                    "meta": {
                        "source": "self_critique",
                        "category": out["category"],
                        "teacher": "reverse_from_self_critique",
                        "quality": {
                            "evaluator": "local_semantic",
                            "similarity": out["similarity_score"],
                            "threshold": cfg.threshold,
                        },
                    },
                }
                rev_f.write(json.dumps(rev, ensure_ascii=False) + "\n")

    with open(Path(cfg.output_dir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "count": total,
                "passes": passes,
                "fails": fails,
                "pass_rate": round(pass_rate * 100, 2),
                "threshold": cfg.threshold,
                "model": cfg.model_name,
            },
            f,
            indent=2,
        )

    logger.success(
        f"Evaluation complete → pass_rate={pass_rate*100:.2f}% | passes={passes} fails={fails} | outputs: {cfg.output_dir}"
    )
    return passes, fails, pass_rate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate self-critique outputs locally via semantic similarity")
    parser.add_argument("--input_jsonl", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--threshold", type=float, default=0.74)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformer for semantic similarity",
    )
    args = parser.parse_args()

    cfg = EvalConfig(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        threshold=args.threshold,
        limit=args.limit,
        batch_size=args.batch_size,
        model_name=args.model_name,
    )
    evaluate_self_critique(cfg)


if __name__ == "__main__":
    main()
