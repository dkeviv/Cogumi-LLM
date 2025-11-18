#!/bin/bash
# MAML Training v2 - Minimal Stable Configuration for 8B
# Run this after stopping the current training

# Navigate to project directory
cd /workspace

# Stop any running training (if not already stopped)
# Press Ctrl+C in the terminal running the old training

# Run corrected training (v2 - Minimal Stable for 8B)
python /workspace/scripts/phase1_train_maml_lora.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --data_file data/phase1/answers/training_data_clean.jsonl \
    --output_dir models/phase1_maml_lora_v2 \
    --num_epochs 3 \
    --inner_steps 1 \
    --tasks_per_batch 2 \
    --support_size 4 \
    --query_size 4 \
    --inner_learning_rate 1e-5 \
    --learning_rate 3e-6 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --gradient_accumulation_steps 8 \
    --patience 2 \
    --max_seq_length 2048

echo ""
echo "Training complete! Check models/phase1_maml_lora_v2/ for checkpoints"
