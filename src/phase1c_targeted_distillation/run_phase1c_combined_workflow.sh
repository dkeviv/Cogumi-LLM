#!/bin/bash
# Phase 1C/1D Combined Training - Complete Workflow
# 
# This script orchestrates the entire Phase 1C/1D pipeline:
# 1. Generate Claude examples for hard failures
# 2. Create bidirectional pairs for self-critique
# 3. Create bidirectional pairs for Claude examples  
# 4. Combine into unified training dataset
# 5. Run smart training with early stopping
# 6. Validate results
#
# Expected timeline: ~8-12 hours total
# Expected cost: ~$200-250

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
HARD_FAILURES="./Phase 1B_2_0/data/Phase 1B_2_0/phase1c_hard_failures.jsonl"
SELF_CRITIQUE="./Phase 1B_2_0/data/data/phase1c/phase1c_self_critique_train.jsonl"
OUTPUT_DIR="data/phase1c"
CHECKPOINT_DIR="data/checkpoints/phase1c_combined"
BASE_MODEL="Phase1A_2_0/models/phase1a_merged_10gb"

# API Provider (openai=cheaper, claude=better)
API_PROVIDER="${API_PROVIDER:-openai}"  # Default to OpenAI GPT-4o-mini
MODEL="${MODEL:-gpt-4o-mini}"           # Default model

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}PHASE 1C/1D COMBINED TRAINING - COMPLETE WORKFLOW${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo "Configuration:"
echo "  API Provider: $API_PROVIDER"
echo "  Model: $MODEL"
echo "  Hard failures: $HARD_FAILURES"
echo "  Self-critique: $SELF_CRITIQUE"
echo "  Output directory: $OUTPUT_DIR"
echo "  Base model: $BASE_MODEL"
echo ""

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

if [ ! -f "$HARD_FAILURES" ]; then
    echo -e "${RED}❌ Hard failures file not found: $HARD_FAILURES${NC}"
    exit 1
fi

if [ ! -f "$SELF_CRITIQUE" ]; then
    echo -e "${RED}❌ Self-critique file not found: $SELF_CRITIQUE${NC}"
    exit 1
fi

if [ ! -d "$BASE_MODEL" ]; then
    echo -e "${RED}❌ Base model not found: $BASE_MODEL${NC}"
    exit 1
fi

# Check API keys
if [ "$API_PROVIDER" == "openai" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}❌ OPENAI_API_KEY not set${NC}"
    exit 1
fi

if [ "$API_PROVIDER" == "claude" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "${RED}❌ ANTHROPIC_API_KEY not set${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites checked${NC}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ============================================================================
# STEP 1: Generate Claude examples (optional - can be skipped if already done)
# ============================================================================
CLAUDE_EXAMPLES="$OUTPUT_DIR/improved_examples.jsonl"

if [ -f "$CLAUDE_EXAMPLES" ]; then
    EXISTING_COUNT=$(wc -l < "$CLAUDE_EXAMPLES")
    echo -e "${YELLOW}⚠️  Found existing Claude examples: $EXISTING_COUNT generated${NC}"
    read -p "Skip generation? (y/n): " SKIP_CLAUDE
    
    if [ "$SKIP_CLAUDE" == "n" ]; then
        echo -e "${BLUE}Step 1: Generating Claude examples...${NC}"
        
        # Cost estimation first
        python src/phase1c_targeted_distillation/generate_claude_examples.py \
            --input "$HARD_FAILURES" \
            --output "$CLAUDE_EXAMPLES" \
            --api_provider "$API_PROVIDER" \
            --model "$MODEL" \
            --estimate_only
        
        echo ""
        read -p "Proceed with generation? (yes/no): " CONFIRM
        
        if [ "$CONFIRM" != "yes" ]; then
            echo -e "${RED}❌ Aborted by user${NC}"
            exit 1
        fi
        
        # Generate (can be interrupted and resumed)
        python src/phase1c_targeted_distillation/generate_claude_examples.py \
            --input "$HARD_FAILURES" \
            --output "$CLAUDE_EXAMPLES" \
            --api_provider "$API_PROVIDER" \
            --model "$MODEL" \
            --batch_size 10 \
            --delay 0.5
        
        echo -e "${GREEN}✅ Claude examples generated${NC}"
    else
        echo -e "${GREEN}✅ Using existing Claude examples${NC}"
    fi
else
    echo -e "${BLUE}Step 1: Generating Claude examples...${NC}"
    
    # Cost estimation first
    python src/phase1c_targeted_distillation/generate_claude_examples.py \
        --input "$HARD_FAILURES" \
        --output "$CLAUDE_EXAMPLES" \
        --api_provider "$API_PROVIDER" \
        --model "$MODEL" \
        --estimate_only
    
    echo ""
    read -p "Proceed with generation? (yes/no): " CONFIRM
    
    if [ "$CONFIRM" != "yes" ]; then
        echo -e "${RED}❌ Aborted by user${NC}"
        exit 1
    fi
    
    # Generate
    python src/phase1c_targeted_distillation/generate_claude_examples.py \
        --input "$HARD_FAILURES" \
        --output "$CLAUDE_EXAMPLES" \
        --api_provider "$API_PROVIDER" \
        --model "$MODEL" \
        --batch_size 10 \
        --delay 0.5
    
    echo -e "${GREEN}✅ Claude examples generated${NC}"
fi

echo ""

# ============================================================================
# STEP 2: Create bidirectional pairs for self-critique
# ============================================================================
echo -e "${BLUE}Step 2: Creating bidirectional pairs for self-critique...${NC}"

SELF_CRITIQUE_BIDIRECTIONAL="$OUTPUT_DIR/self_critique_bidirectional.jsonl"

python src/phase1c_targeted_distillation/create_bidirectional_pairs.py \
    --input "$SELF_CRITIQUE" \
    --output "$SELF_CRITIQUE_BIDIRECTIONAL" \
    --source_label "self_critique" \
    --validate

echo -e "${GREEN}✅ Self-critique bidirectional pairs created${NC}"
echo ""

# ============================================================================
# STEP 3: Create bidirectional pairs for Claude examples
# ============================================================================
echo -e "${BLUE}Step 3: Creating bidirectional pairs for Claude examples...${NC}"

CLAUDE_BIDIRECTIONAL="$OUTPUT_DIR/claude_bidirectional.jsonl"

python src/phase1c_targeted_distillation/create_bidirectional_pairs.py \
    --input "$CLAUDE_EXAMPLES" \
    --output "$CLAUDE_BIDIRECTIONAL" \
    --source_label "claude_generation" \
    --validate

echo -e "${GREEN}✅ Claude bidirectional pairs created${NC}"
echo ""

# ============================================================================
# STEP 4: Combine datasets
# ============================================================================
echo -e "${BLUE}Step 4: Combining datasets...${NC}"

COMBINED_TRAINING="$OUTPUT_DIR/combined_training_bidirectional.jsonl"

cat "$SELF_CRITIQUE_BIDIRECTIONAL" "$CLAUDE_BIDIRECTIONAL" > "$COMBINED_TRAINING"

# Validation
SELF_CRITIQUE_COUNT=$(wc -l < "$SELF_CRITIQUE_BIDIRECTIONAL")
CLAUDE_COUNT=$(wc -l < "$CLAUDE_BIDIRECTIONAL")
COMBINED_COUNT=$(wc -l < "$COMBINED_TRAINING")
EXPECTED_COUNT=$((SELF_CRITIQUE_COUNT + CLAUDE_COUNT))

echo "  Self-critique pairs: $SELF_CRITIQUE_COUNT"
echo "  Claude pairs: $CLAUDE_COUNT"
echo "  Combined total: $COMBINED_COUNT"
echo "  Expected: $EXPECTED_COUNT"

if [ "$COMBINED_COUNT" -eq "$EXPECTED_COUNT" ]; then
    echo -e "${GREEN}✅ Dataset combination verified${NC}"
else
    echo -e "${RED}❌ Dataset combination mismatch!${NC}"
    exit 1
fi

echo ""

# ============================================================================
# STEP 5: Smart training with early stopping
# ============================================================================
echo -e "${BLUE}Step 5: Starting smart training with early stopping...${NC}"
echo ""
echo "Training configuration:"
echo "  Base model: $BASE_MODEL"
echo "  Dataset: $COMBINED_TRAINING ($COMBINED_COUNT examples)"
echo "  Output: $CHECKPOINT_DIR"
echo "  Max epochs: 3 (will stop early if converged)"
echo "  Expected: 5-7 hours, ~\$15-20"
echo ""

read -p "Start training? (yes/no): " START_TRAINING

if [ "$START_TRAINING" != "yes" ]; then
    echo -e "${YELLOW}⏸️  Training skipped by user${NC}"
    echo ""
    echo "To run training manually:"
    echo "  python src/phase1c_targeted_distillation/train_phase1c_combined_smart.py \\"
    echo "    --model_name $BASE_MODEL \\"
    echo "    --dataset $COMBINED_TRAINING \\"
    echo "    --output_dir $CHECKPOINT_DIR"
    exit 0
fi

python src/phase1c_targeted_distillation/train_phase1c_combined_smart.py \
    --model_name "$BASE_MODEL" \
    --dataset "$COMBINED_TRAINING" \
    --output_dir "$CHECKPOINT_DIR" \
    --logging_dir "data/logs/phase1c_combined" \
    --max_epochs 3 \
    --patience 3 \
    --validation_split 0.05 \
    --eval_steps 500 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 3e-6 \
    --lora_r 64 \
    --lora_alpha 16

echo -e "${GREEN}✅ Training complete!${NC}"
echo ""

# ============================================================================
# STEP 6: Validation summary
# ============================================================================
echo -e "${BLUE}Step 6: Training summary${NC}"
echo ""

if [ -f "$CHECKPOINT_DIR/training_summary.json" ]; then
    echo "Training summary:"
    cat "$CHECKPOINT_DIR/training_summary.json"
    echo ""
fi

echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}PHASE 1C/1D COMPLETE!${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Merge LoRA adapter to base model"
echo "  2. Re-evaluate on 7,331 test examples"
echo "  3. Compare pass rate improvement (target: 63.34% → 88-92%)"
echo "  4. Proceed to Phase 1E (draft model distillation)"
echo ""
echo "Model location: $CHECKPOINT_DIR/final"
echo "TensorBoard logs: data/logs/phase1c_combined"
echo ""
