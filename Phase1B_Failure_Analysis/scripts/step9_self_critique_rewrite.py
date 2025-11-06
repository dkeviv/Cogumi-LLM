#!/usr/bin/env python3
"""
Context: Phase 1B → Hybrid Approach (Self-Critique Pass)
Purpose: For each authoritative FAIL case, prompt the local model to (a) briefly critique the
         prior failed answer against the reference, and (b) produce a corrected final answer.

Outputs:
- JSONL with one record per item including critique and final_answer.

Usage:
  python "Phase 1B_2_0/step9_self_critique_rewrite.py" \
    --model_path Phase1A_2_0/models/phase1a_merged_10gb \
    --failures_jsonl "Phase 1B_2_0/data/haiku_replay/phase1c_failures_haiku_replay.jsonl" \
    --output_jsonl "Phase 1B_2_0/data/self_critique/rewrite.jsonl" \
    --device cuda --load_in_4bit --resume --mode hf

Fast wiring pilot (no model):
    python "Phase 1B_2_0/step9_self_critique_rewrite.py" \
        --failures_jsonl "Phase 1B_2_0/data/haiku_replay/phase1c_failures_haiku_replay.jsonl" \
        --output_jsonl "Phase 1B_2_0/data/self_critique/rewrite.jsonl" \
        --mode dummy --limit 200

Notes:
- Keep outputs concise. This is a first-pass, low-cost corrective step. Hard cases escalate to GPT‑5.
- Deterministic-ish configs: low temperature, capped tokens, JSON-constrained responses.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
from loguru import logger
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from bitsandbytes import __version__ as _bnb_version  # noqa: F401
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:  # pragma: no cover
    _HAS_BNB = False


PROMPT_TEMPLATE = (
    "You are improving a previous answer by self-critique.\n"
    "Instruction:\n{instruction}\n\n"
    "Previous (failed) answer:\n{prev}\n\n"
    "Reference (correct) answer:\n{ref}\n\n"
    "Task: 1) Briefly critique what is wrong in the previous answer compared to the reference.\n"
    "2) Then write a concise, corrected final answer that directly satisfies the instruction.\n"
    "Respond in strict JSON with keys: critique, final_answer. No extra text."
)


def load_model(model_path: str, device: str = "cuda", load_in_4bit: bool = False) -> Tuple[Any, Any]:
    """Load the local model and tokenizer."""
    logger.info(f"Loading model from {model_path} on {device}…")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_kwargs = {"trust_remote_code": True}
    
    if load_in_4bit:
        if not _HAS_BNB:
            logger.warning("bitsandbytes not available; proceeding without 4-bit quantization")
            model_kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32
            model_kwargs["device_map"] = "auto" if device == "cuda" else None
        else:
            # For 4-bit quantization, let bitsandbytes handle device placement
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            )
            model_kwargs["quantization_config"] = quant_cfg
            model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32
        model_kwargs["device_map"] = "auto" if device == "cuda" else None

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    return model, tokenizer


def safe_json_extract(text: str) -> Dict[str, Any]:
    """Extract a JSON object with keys critique/final_answer from raw model output.

    Attempts to find the first {...} block; falls back to a best-effort parse.
    """
    try:
        # Try direct
        return json.loads(text)
    except Exception:
        pass
    # Try to find the first JSON object in output
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # Fallback: wrap as minimal
    return {"critique": text.strip()[:300], "final_answer": ""}


def generate_dummy_response(instruction: str, prev: str, ref: str) -> Dict[str, Any]:
    """Generate a deterministic dummy response for wiring tests without loading a model.

    Heuristic:
    - critique: short message pointing to mismatch vs reference
    - final_answer: trimmed reference (acts as a stand-in for corrected answer)
    """
    critique = "Previous answer diverges from key points in the reference; align content and be concise."
    final = (ref or "").strip()
    # Truncate excessively long references for safety
    if len(final) > 1200:
        final = final[:1200].rstrip() + "…"
    return {"critique": critique, "final_answer": final}


def generate_json_response(model, tokenizer, prompt: str, max_new_tokens: int = 220) -> Dict[str, Any]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt prefix if present
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt) :].strip()
    return safe_json_extract(decoded)


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-critique + rewrite failed answers (hybrid approach)")
    parser.add_argument("--model_path", required=False, type=str, help="HF model path (required if --mode hf)")
    parser.add_argument("--failures_jsonl", required=True, type=str)
    parser.add_argument("--output_jsonl", required=True, type=str)
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume: skip IDs already present in output_jsonl")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=160,
        help="Max new tokens for generation (lower for CPU runs)",
    )
    parser.add_argument(
        "--mode",
        choices=["hf", "dummy"],
        default="hf",
        help="hf: use local HF model; dummy: no-model deterministic outputs for wiring tests",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit (bitsandbytes) to reduce memory and speed up on GPUs",
    )
    args = parser.parse_args()

    os.makedirs(Path(args.output_jsonl).parent, exist_ok=True)

    model = None
    tokenizer = None
    if args.mode == "hf":
        assert args.model_path, "--model_path is required when --mode hf"
        model, tokenizer = load_model(args.model_path, args.device, load_in_4bit=args.load_in_4bit)
        # Move to MPS explicitly if requested (but not for quantized models)
        if args.device == "mps" and not args.load_in_4bit:
            try:
                model.to("mps")
            except Exception as e:
                logger.warning(f"Could not move model to MPS: {e}. Falling back to CPU.")

    failures = []
    for i, rec in enumerate(iter_jsonl(args.failures_jsonl)):
        failures.append(rec)
        if args.limit and len(failures) >= args.limit:
            break

    total = len(failures)
    logger.info(f"Loaded {total} failure items")

    # Resume support: collect existing IDs if requested
    processed_ids = set()
    if args.resume and os.path.exists(args.output_jsonl):
        try:
            with open(args.output_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        j = json.loads(line)
                        if "id" in j:
                            processed_ids.add(j["id"])
            logger.info(f"Resume enabled: found {len(processed_ids)} already written IDs; will skip them")
        except Exception as e:
            logger.warning(f"Failed to read existing output for resume: {e}")

    write_mode = "a" if args.resume and os.path.exists(args.output_jsonl) else "w"
    with open(args.output_jsonl, write_mode, encoding="utf-8") as out_f:
        with Progress(
            TextColumn("[bold blue]Self-critique rewrite"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("run", total=total)

            for idx, item in enumerate(failures):
                if processed_ids and item.get("id") in processed_ids:
                    progress.advance(task)
                    continue
                instruction = item.get("instruction", "")
                prev = item.get("model_output", "")
                ref = item.get("reference_answer", "")

                prompt = PROMPT_TEMPLATE.format(instruction=instruction, prev=prev, ref=ref)
                if args.mode == "hf":
                    result = generate_json_response(
                        model,
                        tokenizer,
                        prompt,
                        max_new_tokens=max(32, min(512, args.max_new_tokens)),
                    )
                else:
                    result = generate_dummy_response(instruction, prev, ref)
                record = {
                    "id": item.get("id"),
                    "category": item.get("category", "unknown"),
                    "instruction": instruction,
                    "reference_answer": ref,
                    "previous_output": prev,
                    "critique": result.get("critique", ""),
                    "final_answer": result.get("final_answer", ""),
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

                if idx and idx % 100 == 0:
                    logger.info(f"Progress: {idx}/{total} ({idx/total:.1%})")
                progress.advance(task)

    logger.success(f"Self-critique rewrite complete → {args.output_jsonl}")


if __name__ == "__main__":
    main()
