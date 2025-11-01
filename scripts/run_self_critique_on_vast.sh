#!/usr/bin/env bash
# Context: Run self-critique (step9) efficiently on a Vast.ai GPU instance
# Usage: bash scripts/run_self_critique_on_vast.sh <MODEL_DIR> <FAILURES_JSONL> <OUTPUT_JSONL> [LIMIT]
# Example:
#   bash scripts/run_self_critique_on_vast.sh /workspace/phase1a_merged_10gb \
#     "/workspace/Phase 1B_2_0/data/haiku_replay/phase1c_failures_haiku_replay.jsonl" \
#     "/workspace/Phase 1B_2_0/data/self_critique/rewrite_hf_full.jsonl" 7331

set -euo pipefail

MODEL_DIR=${1:?"Model directory required (HF checkpoint)"}
FAILURES=${2:?"Failures JSONL required"}
OUTPUT=${3:?"Output JSONL path required"}
LIMIT=${4:-}

# 1) Environment prep
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Ensure bitsandbytes present (GPU accel)
pip install bitsandbytes==0.42.0 || true

# 2) Optional cleanup
mkdir -p "$(dirname "$OUTPUT")"

# 3) Run step9 with 4-bit, CUDA, and resume
EXTRA_LIMIT=""
if [[ -n "$LIMIT" ]]; then
  EXTRA_LIMIT="--limit $LIMIT"
fi

python "Phase 1B_2_0/step9_self_critique_rewrite.py" \
  --model_path "$MODEL_DIR" \
  --failures_jsonl "$FAILURES" \
  --output_jsonl "$OUTPUT" \
  --device cuda --mode hf --load_in_4bit --resume \
  --max_new_tokens 128 $EXTRA_LIMIT

# 4) Evaluate locally (fast) and emit training-ready datasets
python "Phase 1B_2_0/step10_evaluate_self_critique_local.py" \
  --input_jsonl "$OUTPUT" \
  --output_dir "$(dirname "$OUTPUT")/eval" \
  --threshold 0.74 --batch_size 128

echo "Done. Outputs at $(dirname "$OUTPUT") and eval/"
