#!/bin/bash
# ⚠️ SUPERSEDED - USE NEWER VERSION ⚠️
# ========================================
#
# This script is SUPERSEDED by a better implementation.
#
# **Current Version:** scripts/vastai_validate_and_merge.sh (RECOMMENDED)
# **This Version:** Still functional but uses deprecated approach
#
# **Why Superseded:**
# - Uses test set extracted from training data (contaminated)
# - Model was trained on all data, so no true held-out test
# - New version uses independent benchmarks for real generalization test
#
# **When to Use New Version:**
# - For true generalization testing (benchmarks never seen in training)
# - When running on Vast.ai (optimized for that environment)
# - For comparable results (standard benchmarks)
#
# **Performance Comparison:**
# - Old: Tests training accuracy on contaminated data
# - New: Tests generalization on independent benchmarks
#
# **Archive Date:** November 18, 2025
# **Reason:** Replaced by benchmark-based validation approach
#
# See: docs/PHASE1D_BENCHMARK_VALIDATION.md for full explanation
#
# ========================================
#
# Phase 1D: Complete Validation & Merge Workflow
# 
# This script orchestrates:
# 1. Create test set (stratified sampling)
# 2. Validate LoRA model on test set
# 3. Merge LoRA weights into base model
# 4. Validate merged model on test set
# 5. Compare results (verify identical)
#
# Usage:
#   bash scripts/phase1_validate_and_merge.sh
#
# Expected Runtime: 30-45 minutes
# Expected Cost: $0 (using existing trained model)

set -e  # Exit on error

# Configuration
MODEL_PATH="models/phase1_maml_lora_v2/best"
MERGED_PATH="models/phase1_maml_lora_v2/merged"
TRAIN_FILE="data/phase1/answers/training_data_clean.jsonl"
TEST_FILE="data/phase1/test_set.jsonl"
RESULTS_DIR="results/phase1_validation"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Phase 1D: Validation & Merge Workflow${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Create test set (if not exists)
echo -e "${YELLOW}Step 1/5: Create Test Set${NC}"
if [ -f "$TEST_FILE" ]; then
    echo -e "${GREEN}✓${NC} Test set already exists: $TEST_FILE"
    echo "  Skipping test set creation (use --force to recreate)"
else
    echo "Creating stratified test set (200 examples)..."
    python scripts/phase1_create_test_set.py \
        --train_file "$TRAIN_FILE" \
        --output_file "$TEST_FILE" \
        --test_size 200 \
        --stratify_by difficulty,domain \
        --seed 42 \
        --no_backup
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} Test set created successfully"
    else
        echo -e "${RED}✗${NC} Test set creation failed"
        exit 1
    fi
fi
echo ""

# Step 2: Validate LoRA model
echo -e "${YELLOW}Step 2/5: Validate LoRA Model${NC}"
echo "Running validation on test set..."
python scripts/phase1_validate_maml.py \
    --model_path "$MODEL_PATH" \
    --test_file "$TEST_FILE" \
    --output_dir "$RESULTS_DIR" \
    --skip_merged

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} LoRA validation complete"
else
    echo -e "${RED}✗${NC} LoRA validation failed"
    exit 1
fi
echo ""

# Step 3: Merge LoRA weights
echo -e "${YELLOW}Step 3/5: Merge LoRA Weights${NC}"
if [ -d "$MERGED_PATH" ]; then
    echo -e "${GREEN}✓${NC} Merged model already exists: $MERGED_PATH"
    echo "  Skipping merge (delete $MERGED_PATH to re-merge)"
else
    echo "Merging LoRA adapter into base model..."
    python scripts/phase1_merge_lora.py \
        --lora_path "$MODEL_PATH" \
        --output_path "$MERGED_PATH" \
        --precision bfloat16 \
        --verify
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} Merge complete"
    else
        echo -e "${RED}✗${NC} Merge failed"
        exit 1
    fi
fi
echo ""

# Step 4: Validate merged model
echo -e "${YELLOW}Step 4/5: Validate Merged Model${NC}"
echo "Running validation on merged model..."
python scripts/phase1_validate_maml.py \
    --model_path "$MODEL_PATH" \
    --test_file "$TEST_FILE" \
    --output_dir "$RESULTS_DIR"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Merged validation complete"
else
    echo -e "${RED}✗${NC} Merged validation failed"
    exit 1
fi
echo ""

# Step 5: Display results summary
echo -e "${YELLOW}Step 5/5: Results Summary${NC}"
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Validation Results${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if results exist
if [ -f "$RESULTS_DIR/comparison.json" ]; then
    # Extract key metrics using Python
    python3 << EOF
import json
import sys

try:
    with open("$RESULTS_DIR/comparison.json", 'r') as f:
        results = json.load(f)
    
    print("\\n${GREEN}LoRA Model:${NC}")
    print(f"  Loss: {results['lora']['avg_loss']:.6f}")
    print(f"  Perplexity: {results['lora']['perplexity']:.4f}")
    print(f"  Examples: {results['lora']['num_examples']}")
    
    print("\\n${GREEN}Merged Model:${NC}")
    print(f"  Loss: {results['merged']['avg_loss']:.6f}")
    print(f"  Perplexity: {results['merged']['perplexity']:.4f}")
    print(f"  Examples: {results['merged']['num_examples']}")
    
    print("\\n${YELLOW}Comparison:${NC}")
    print(f"  Loss Difference: {results['loss_difference']:.8f}")
    print(f"  PPL Difference: {results['perplexity_difference']:.8f}")
    
    if results['identical']:
        print("\\n${GREEN}✓ Models are IDENTICAL (within tolerance)${NC}")
        print("  Merge was successful!")
        sys.exit(0)
    else:
        print("\\n${RED}✗ Models differ significantly${NC}")
        print("  This may indicate merge issues")
        sys.exit(1)

except Exception as e:
    print(f"\\n${RED}✗ Error reading results: {e}${NC}")
    sys.exit(1)
EOF

    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${BLUE}========================================${NC}"
        echo -e "${GREEN}✓ All validations passed!${NC}"
        echo -e "${BLUE}========================================${NC}"
        echo ""
        echo -e "${YELLOW}Next Steps:${NC}"
        echo "1. Extract base model responses for draft training:"
        echo "   python scripts/phase1_extract_base_responses.py"
        echo ""
        echo "2. Train draft model (Phase 1E):"
        echo "   python scripts/phase1e_train_draft.py"
        echo ""
        echo "3. Or proceed to compression (Phase 2):"
        echo "   bash scripts/setup_phase2_compression.sh"
        echo ""
        echo -e "${BLUE}Results saved in: $RESULTS_DIR${NC}"
    else
        echo ""
        echo -e "${RED}✗ Validation issues detected${NC}"
        echo "Check $RESULTS_DIR for detailed results"
        exit 1
    fi
else
    echo -e "${RED}✗ Results not found${NC}"
    echo "Validation may have failed earlier"
    exit 1
fi

else
    echo -e "${RED}✗${NC} Comparison results not found"
    echo "Previous steps may have failed"
    exit 1
fi
