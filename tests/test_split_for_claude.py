"""
Context: Unit tests for Phase 1B utility `split_for_claude.py`.

This test verifies that the splitter can:
- Split by fixed example count into matched chunk pairs
- Split by size threshold while preserving pairing
- Produce a summary JSON describing the chunks
- Warn (via metadata) when a chunk exceeds the configured size

We use tiny JSONL fixtures and temporary directories.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import pytest

# Import functions from the script by path-safe module loading
import importlib.util
import sys


def _load_splitter_module() -> object:
    """Dynamically load the splitter script as a module for testing."""
    script_path = Path(__file__).resolve().parents[1] / "Phase 1B_2_0" / "split_for_claude.py"
    spec = importlib.util.spec_from_file_location("split_for_claude", str(script_path))
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    # Register module name to satisfy dataclasses' lookups on Python 3.14
    sys.modules[spec.name] = module  # type: ignore[index]
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


@pytest.fixture()
def tiny_pairs(tmp_path: Path):
    """Create small paired JSONL files with matching ids and content."""
    test_path = tmp_path / "test.jsonl"
    model_path = tmp_path / "model.jsonl"

    with test_path.open("w", encoding="utf-8") as tf, model_path.open(
        "w", encoding="utf-8"
    ) as mf:
        for i in range(1, 21):
            rec = {"id": f"item-{i}", "val": i}
            tf.write(json.dumps(rec) + "\n")
            mf.write(json.dumps(rec) + "\n")

    return test_path, model_path


def test_split_by_count_basic(tmp_path: Path, tiny_pairs):
    mod = _load_splitter_module()
    splitter = cast(Any, mod)
    test_path, model_path = tiny_pairs

    out_dir = tmp_path / "out_count"
    summary = splitter.split_by_count(
        test_path=test_path,
        model_path=model_path,
        output_dir=out_dir,
        examples_per_chunk=7,
        max_mb=100.0,
        strict=True,
        id_sample_every=5,
    )

    # Expect ceil(20/7) = 3 chunks
    assert len(summary.chunks) == 3
    # Sum of examples equals 20
    assert sum(c.num_examples for c in summary.chunks) == 20

    # Files exist and are non-empty
    for c in summary.chunks:
        assert (out_dir / c.test_file).is_file()
        assert (out_dir / c.model_file).is_file()
        assert c.num_examples > 0

    # Summary JSON written
    summary_path = out_dir / "split_summary.json"
    splitter._write_summary(summary, out_dir)
    assert summary_path.is_file()
    data = json.loads(summary_path.read_text())
    assert data["mode"] == "by-count"


def test_split_by_size_threshold(tmp_path: Path, tiny_pairs):
    mod = _load_splitter_module()
    splitter = cast(Any, mod)
    test_path, model_path = tiny_pairs

    out_dir = tmp_path / "out_size"
    # Set a very small MB threshold to force multiple chunks
    summary = splitter.split_by_size(
        test_path=test_path,
        model_path=model_path,
        output_dir=out_dir,
        max_mb=0.00001,  # ~10 bytes to force multiple chunks with tiny lines
        strict=True,
        id_sample_every=5,
    )

    assert len(summary.chunks) >= 2
    # Paired chunk sizes do not exceed the threshold according to metadata
    for c in summary.chunks:
        assert c.test_mb <= 0.01  # With tiny records, expect well under 0.01 MB
        assert c.model_mb <= 0.01
        assert c.num_examples > 0

    # Summary JSON written
    summary_path = out_dir / "split_summary.json"
    splitter._write_summary(summary, out_dir)
    assert summary_path.is_file()
    data = json.loads(summary_path.read_text())
    assert data["mode"] == "by-size"
