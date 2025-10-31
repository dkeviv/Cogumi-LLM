"""
Context: Phase 1B re-evaluation pipeline to reproduce batch judgments using a pluggable judge.

Purpose:
- Re-judge the 200 JSON batch files under `data/copilot_batches/` that contain 20K examples
- Write per-batch JSONL outputs to `data/GPT5judged/` with fields:
  id, category, instruction, reference, model_output, judgment, reason, confidence, judged_by
- Aggregate results and emit a summary with per-category pass/fail counts and overall pass rate

Notes:
- This script provides a Judge interface with three backends: "mock" (fast dry-run),
  "local" (placeholder for local model), and "gpt5" (external API integration).
- Follows Cogumi-LLM guidelines: progress bars, logging, error handling, and resumability.

Usage:
  python step7_rejudge_gpt5.py --mode mock --limit_batches 2
  python step7_rejudge_gpt5.py --mode gpt5

Outputs:
- data/GPT5judged/batch_XXX.jsonl (per-batch judgments)
- data/GPT5judged/summary.json (aggregated stats)

"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from rich.progress import Progress, TimeElapsedColumn, TimeRemainingColumn, BarColumn, TextColumn
from loguru import logger


# -------------- Configuration Constants --------------
DATA_DIR = Path(__file__).parent / "data"
BATCHES_DIR = DATA_DIR / "copilot_batches"
OUTPUT_DIR = DATA_DIR / "GPT5judged"
SUMMARY_FILE = OUTPUT_DIR / "summary.json"
AGG_FILE = OUTPUT_DIR / "GPT5judged_all.jsonl"


# -------------- Utilities --------------
def ensure_dirs() -> None:
    """Ensure required directories exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def list_batch_files() -> List[Path]:
    files = sorted(BATCHES_DIR.glob("batch_*.json"))
    return files


# -------------- Judge Interface --------------
@dataclass
class Judgment:
    judgment: str  # "PASS" | "FAIL"
    reason: str
    confidence: float
    judged_by: str


class Judge:
    async def assess(self, example: Dict[str, Any]) -> Judgment:
        raise NotImplementedError


class MockJudge(Judge):
    """Lightweight simulated judge for dry-runs and CI checks."""
    async def assess(self, example: Dict[str, Any]) -> Judgment:
        # Simple heuristic to simulate variation without external cost
        text = (example.get("model_output") or "") + " " + (example.get("reference") or "")
        score = (len(text) % 10) / 10.0
        verdict = "PASS" if score > 0.3 else "FAIL"
        return Judgment(
            judgment=verdict,
            reason="mock-eval based on length mod heuristic",
            confidence=0.5 + (score / 2.0),
            judged_by="mock",
        )


class LocalJudge(Judge):
    """Placeholder for a local model judge implementation."""
    async def assess(self, example: Dict[str, Any]) -> Judgment:
        # Placeholder: implement local model comparison if needed
        # For now, mirror mock behavior but mark judged_by
        text = (example.get("model_output") or "") + " " + (example.get("reference") or "")
        score = (len(text) % 7) / 7.0
        verdict = "PASS" if score > 0.35 else "FAIL"
        return Judgment(
            judgment=verdict,
            reason="local placeholder heuristic",
            confidence=0.5 + (score / 2.0),
            judged_by="local",
        )


class GPT5Judge(Judge):
    def __init__(self, concurrency: int = 8):
        self.semaphore = asyncio.Semaphore(concurrency)

    async def _call_api(self, example: Dict[str, Any]) -> Judgment:
        # NOTE: Replace with actual GPT-5 API call in your environment.
        # This placeholder simulates a conservative judgment pattern.
        model_out = example.get("model_output") or ""
        ref = example.get("reference") or ""
        overlap = len(set(model_out.split()) & set(ref.split()))
        threshold = max(3, int(len(ref.split()) * 0.05))
        verdict = "PASS" if overlap >= threshold else "FAIL"
        conf = 0.6 if verdict == "PASS" else 0.7
        return Judgment(
            judgment=verdict,
            reason=f"simulated gpt5 overlap={overlap} threshold={threshold}",
            confidence=conf,
            judged_by="gpt5-sim",
        )

    async def assess(self, example: Dict[str, Any]) -> Judgment:
        async with self.semaphore:
            return await self._call_api(example)


# -------------- Core Evaluation Logic --------------
async def evaluate_batch(
    judge: Judge,
    batch_path: Path,
) -> Tuple[Path, int, int, int]:
    """Evaluate a single batch and write JSONL output.

    Returns: (output_path, total, passed, failed)
    """
    data = read_json(batch_path)
    examples = data.get("examples", [])

    out_path = OUTPUT_DIR / (batch_path.stem + ".jsonl")
    rows: List[Dict[str, Any]] = []

    passed = failed = 0
    samples_preview: List[Dict[str, Any]] = []
    for idx, ex in enumerate(examples):
        j = await judge.assess(ex)
        row = {
            "id": ex.get("id"),
            "category": ex.get("category"),
            "instruction": ex.get("instruction"),
            "reference": ex.get("reference"),
            "model_output": ex.get("model_output"),
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
        # Capture small preview for verification hooks
        if idx < 5:
            samples_preview.append({k: row[k] for k in ("id", "category", "judgment", "confidence")})

    write_jsonl(out_path, rows)
    if samples_preview:
        logger.info(f"Sample judgments for {batch_path.name}: {samples_preview}")
    return out_path, len(examples), passed, failed


async def main(mode: str, limit_batches: Optional[int], resume: bool, clean: bool) -> None:
    ensure_dirs()
    logger.add(OUTPUT_DIR / "rejudge.log", rotation="2 MB")
    if mode == "mock":
        judge: Judge = MockJudge()
    elif mode == "local":
        judge = LocalJudge()
    elif mode == "gpt5":
        judge = GPT5Judge()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    batch_files = list_batch_files()
    if limit_batches:
        batch_files = batch_files[:limit_batches]

    # Cleanup old outputs if requested
    if clean:
        for f in OUTPUT_DIR.glob("batch_*.jsonl"):
            try:
                f.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete {f}: {e}")
        if AGG_FILE.exists():
            try:
                AGG_FILE.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete {AGG_FILE}: {e}")

    total_all = passed_all = failed_all = 0
    per_category: Dict[str, Dict[str, int]] = {}

    # Open aggregate file for append (create new if not exists)
    agg_mode = "a" if resume and AGG_FILE.exists() else "w"
    agg_file_handle = AGG_FILE.open(agg_mode, encoding="utf-8")

    with Progress(
        TextColumn("[bold blue]Re-judging"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("batches", total=len(batch_files))
        for i, batch in enumerate(batch_files, 1):
            out_path = OUTPUT_DIR / (batch.stem + ".jsonl")
            if resume and out_path.exists():
                logger.info(f"Skipping existing {out_path.name} (resume mode)")
                # still aggregate into global file to ensure completeness
                with out_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        d = json.loads(line)
                        agg_file_handle.write(json.dumps(d, ensure_ascii=False) + "\n")
                        cat = d.get("category", "unknown")
                        s = per_category.setdefault(cat, {"total": 0, "pass": 0, "fail": 0})
                        s["total"] += 1
                        if d.get("judgment") == "PASS":
                            s["pass"] += 1
                            passed_all += 1
                        else:
                            s["fail"] += 1
                            failed_all += 1
                        total_all += 1
                progress.advance(task)
                continue

            out_path, total, passed, failed = await evaluate_batch(judge, batch)
            total_all += total
            passed_all += passed
            failed_all += failed

            # Update category stats and aggregate file by reading the just-written file
            with out_path.open("r", encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    agg_file_handle.write(json.dumps(d, ensure_ascii=False) + "\n")
                    cat = d.get("category", "unknown")
                    s = per_category.setdefault(cat, {"total": 0, "pass": 0, "fail": 0})
                    s["total"] += 1
                    if d.get("judgment") == "PASS":
                        s["pass"] += 1
                    else:
                        s["fail"] += 1

            logger.info(
                f"[{i}/{len(batch_files)}] {batch.name} → {out_path.name} :: total={total} pass={passed} fail={failed}"
            )
            progress.advance(task)

    agg_file_handle.close()

    summary = {
        "mode": mode,
        "total": total_all,
        "pass": passed_all,
        "fail": failed_all,
        "pass_rate": round(100.0 * passed_all / max(1, total_all), 2),
        "per_category": per_category,
        "output_dir": str(OUTPUT_DIR),
    }
    with SUMMARY_FILE.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print("GPT5judged SUMMARY")
    print("=" * 70)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-judge copilot_batches with GPT5-like judge")
    parser.add_argument("--mode", choices=["mock", "local", "gpt5"], default="mock")
    parser.add_argument("--limit_batches", type=int, default=None, help="Limit number of batches for quick run")
    parser.add_argument("--resume", action="store_true", help="Skip existing batch outputs and append to aggregate")
    parser.add_argument("--clean", action="store_true", help="Delete old outputs in data/GPT5judged before running")
    args = parser.parse_args()

    asyncio.run(main(args.mode, args.limit_batches, args.resume, args.clean))
