#!/bin/bash
# Train Phase 1B adapter on ALL extracted failures

set -e

echo "================================================================================"
echo "🔧 Phase 1B: Train LoRA Adapter on All Failures"
echo "================================================================================"

# Detect environment
if [ -d "/workspace" ]; then
    BASE_DIR="/workspace/data/Cogumi-LLM"
else
    BASE_DIR="$(pwd)"
fi

cd $BASE_DIR

OUTPUT_DIR="checkpoints/phase1b_all_failures"
MODEL_NAME="checkpoints/phase1a_merged"
DATASET_PATHS=("data/phase1b_all_failures/math_all_failures.jsonl" "data/phase1b_all_failures/code_all_failures.jsonl" "data/phase1b_all_failures/creativity_all_failures.jsonl")

# Verify files exist
for f in "${DATASET_PATHS[@]}"; do
    if [ ! -f "$f" ]; then
        echo "❌ Missing dataset file: $f"
        echo "Run extract_all_failures_prepare_training.py first"
        exit 1
    fi
done

# Merge dataset paths into one glob-friendly pattern or a single file
MERGED_DATA="data/phase1b_all_failures/all_failures_combined.jsonl"

# Combine JSONL files
python - <<'PY'
import json
from pathlib import Path
paths = ["data/phase1b_all_failures/math_all_failures.jsonl",
         "data/phase1b_all_failures/code_all_failures.jsonl",
         "data/phase1b_all_failures/creativity_all_failures.jsonl"]
out = Path('data/phase1b_all_failures/all_failures_combined.jsonl')
with out.open('w') as fo:
    for p in paths:
        for line in Path(p).read_text().splitlines():
            # convert to training format if needed
            try:
                obj = json.loads(line)
                inst = obj.get('instruction') or obj.get('prompt')
                outp = obj.get('output') or obj.get('full_solution') or obj.get('correct_choice')
                if inst and outp:
                    fo.write(json.dumps({'instruction': inst, 'output': outp}) + '\n')
            except Exception as e:
                print('skip line', e)
print('Wrote combined dataset to', out)
PY

# Training hyperparameters - conservative to avoid forgetting
EPOCHS=2
LR=3e-6
BATCH_SIZE=4
ACCUM=4

echo "Training with: model=$MODEL_NAME, output=$OUTPUT_DIR, epochs=$EPOCHS, lr=$LR"

python train_phase1b_benchmark.py \
    --model_name "$MODEL_NAME" \
    --dataset_path "data/phase1b_all_failures/all_failures_combined.jsonl" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $EPOCHS \
    --learning_rate $LR \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $ACCUM

echo ""
echo "================================================================================"
echo "✅ Training finished. Adapter saved to: $OUTPUT_DIR"
echo "Run validation: bash scripts/validate_phase1b1.sh"
echo "================================================================================"
