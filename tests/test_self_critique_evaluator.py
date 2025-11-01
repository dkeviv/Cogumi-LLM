"""
Context: Tests for Phase 1C local evaluator (self-critique uplift).

This smoke test loads the evaluator module directly from file, injects a dummy
SentenceTransformer encoder (no network), and verifies that:
- Similarity-based pass/fail works at a chosen threshold
- improved_forward.jsonl contains only passing items
- Summary counts are correct
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
import sys


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("eval_module", str(path))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register before exec to satisfy dataclasses lookups in Python 3.14
    sys.modules[spec.name] = module  # type: ignore[index]
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


class DummyST:
    """Deterministic sentence-transformers stand-in.

    Maps texts to 2D vectors based on marker words to force different cosine sims:
    - Text containing 'GOOD' → [1, 0]
    - Text containing 'BAD' → [0, 1]
    - Otherwise → [1, 0]
    """

    def __init__(self, *_args, **_kwargs):
        pass

    def encode(self, texts, batch_size=32, normalize_embeddings=False, show_progress_bar=False):  # noqa: ARG002
        def vec(t: str):
            t = (t or "").upper()
            if "BAD" in t:
                return [0.0, 1.0]
            # Default and GOOD map to same vector
            return [1.0, 0.0]

        if isinstance(texts, str):
            return [vec(texts)]
        return [vec(t) for t in texts]


def test_local_evaluator_smoke(tmp_path: Path):
    # Arrange: create small input with one passing and one failing item
    input_path = tmp_path / "rewrite.jsonl"
    items = [
        {
            "id": "1",
            "instruction": "Do X",
            "reference_answer": "GOOD ref",
            "previous_output": "some prev",
            "final_answer": "GOOD answer",
        },
        {
            "id": "2",
            "instruction": "Do Y",
            "reference_answer": "GOOD ref",
            "previous_output": "some prev",
            "final_answer": "BAD answer",
        },
    ]
    with input_path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    out_dir = tmp_path / "eval"

    # Pre-inject dummy sentence_transformers module to avoid heavy dependency
    dummy_pkg = ModuleType("sentence_transformers")
    dummy_pkg.SentenceTransformer = DummyST  # type: ignore[attr-defined]
    sys.modules["sentence_transformers"] = dummy_pkg

    # Load evaluator module (will import our dummy)
    module_path = Path(__file__).parents[1] / "Phase 1B_2_0" / "step10_evaluate_self_critique_local.py"
    mod = _load_module(module_path)

    # Act: run evaluation with threshold between the two classes
    cfg = mod.EvalConfig(
        input_jsonl=str(input_path),
        output_dir=str(out_dir),
        threshold=0.5,
        limit=None,
        batch_size=8,
        model_name="dummy",
    )
    passes, fails, pass_rate = mod.evaluate_self_critique(cfg)

    # Assert: 1 pass, 1 fail
    assert passes == 1
    assert fails == 1
    assert abs(pass_rate - 0.5) < 1e-6

    # improved_forward.jsonl should contain exactly 1 record
    improved_path = out_dir / "improved_forward.jsonl"
    assert improved_path.exists()
    with improved_path.open("r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    assert len(lines) == 1

    # improved_reverse.jsonl should also contain exactly 1 record
    improved_rev_path = out_dir / "improved_reverse.jsonl"
    assert improved_rev_path.exists()
    with improved_rev_path.open("r", encoding="utf-8") as f:
        rev_lines = [ln for ln in f if ln.strip()]
    assert len(rev_lines) == 1

    # summary.json should reflect counts
    summary_path = out_dir / "summary.json"
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["count"] == 2
    assert summary["passes"] == 1
    assert summary["fails"] == 1
    assert 49.0 <= summary["pass_rate"] <= 51.0
