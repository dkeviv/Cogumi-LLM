"""
Context: Tests for Phase 1C dataset builder utility.
Purpose: Validate that the builder converts authoritative failures into forward (and reverse)
         instruction-tuning pairs with the correct schema, honoring test limits.
"""

from __future__ import annotations

import json
import os
from typing import List

import pytest

from src.phase1_base.phase1c_build_dataset import build_datasets


@pytest.mark.parametrize("limit", [5])
def test_build_datasets_smoke(limit: int) -> None:
    input_path = "Phase 1B_2_0/data/GPT5judged/phase1c_failures_gpt5judged.jsonl"
    out_dir = "data/phase1c_test"

    forward_path, reverse_path, count = build_datasets(
        input_path=input_path,
        output_dir=out_dir,
        make_reverse=True,
        test_limit=limit,
    )

    assert count == limit
    assert os.path.isfile(forward_path)
    assert reverse_path is not None and os.path.isfile(reverse_path)

    # Validate first few records have required keys
    def _read(path: str) -> List[dict]:
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(next(iter([line]))) for line in [f.readline() for _ in range(2)] if line]

    fw = _read(forward_path)
    rv = _read(reverse_path) if reverse_path else []

    for rec in fw:
        assert set(["instruction", "input", "output"]).issubset(rec.keys())
        assert isinstance(rec["instruction"], str) and isinstance(rec["output"], str)
        assert rec.get("meta", {}).get("teacher") == "reference_answer"

    for rec in rv:
        assert set(["instruction", "input", "output"]).issubset(rec.keys())
        assert isinstance(rec["instruction"], str) and isinstance(rec["output"], str)
        assert rec.get("meta", {}).get("teacher") == "reverse_from_reference"
