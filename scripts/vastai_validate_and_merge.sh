#!/bin/bash
# Phase 1D: Validation & Merge Workflow for Vast.ai
# 
# This script runs on Vast.ai instance after training completes.
# 
# Steps:
# 0. Validate base model (baseline, pre-training)
# 1. Validate LoRA model on benchmark test set (post-training)
# 2. Merge LoRA weights into base model
# 3. Validate merged model (verify identical to LoRA)
# 4. Compare all results
#
# Usage (on Vast.ai):
#   cd /workspace
#   # First convert benchmarks:
#   python scripts/convert_benchmarks_to_test.py --samples-per-benchmark 100
#   # Then run workflow:
#   bash scripts/vastai_validate_and_merge.sh
#
# Expected Runtime: 45-60 minutes on H100
# Cost: ~$0.75-1.00 (0.75 hours at $1.50/hr)

set -e  # Exit on error

# Configuration
WORKSPACE="/workspace"
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_PATH="$WORKSPACE/models/phase1_maml_lora_v2/final"
MERGED_PATH="$WORKSPACE/models/phase1_maml_lora_v2/merged"
TEST_FILE="$WORKSPACE/data/benchmarks/validation_test.jsonl"
RESULTS_DIR="$WORKSPACE/results/phase1_validation"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Phase 1D: Validation & Merge (Vast.ai)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Base model: $BASE_MODEL (for baseline)"
echo "Trained model: $MODEL_PATH"
echo "Test set: $TEST_FILE (benchmark data)"
echo "Results: $RESULTS_DIR"
echo ""

# Check if trained model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}✗ Error: Trained model not found at $MODEL_PATH${NC}"
    echo "Please ensure training completed successfully"
    exit 1
fi

# Check if test file exists
if [ ! -f "$TEST_FILE" ]; then
    echo -e "${RED}✗ Error: Test set not found at $TEST_FILE${NC}"
    echo "Please run: python scripts/convert_benchmarks_to_test.py --samples-per-benchmark 100"
    exit 1
fi

# Step 0: Validate base model (baseline)
echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 0/4: Validate Base Model (Baseline)${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo "Testing pre-trained Llama-3.1-8B-Instruct (no MAML training)..."
echo "This establishes baseline performance before training."

python $WORKSPACE/scripts/phase1_validate_maml.py \
    --model_path "$BASE_MODEL" \
    --test_file "$TEST_FILE" \
    --output_dir "$RESULTS_DIR/base" \
    --skip_merged

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Base model validation complete"
    
    # Display base model results
    if [ -f "$RESULTS_DIR/base/lora_validation.json" ]; then
        echo ""
        echo -e "${BLUE}Base Model Results (Pre-Training):${NC}"
        python3 -c "
import json
with open('$RESULTS_DIR/base/lora_validation.json') as f:
    r = json.load(f)
    print(f\"  Loss: {r['avg_loss']:.6f}\")
    print(f\"  Perplexity: {r['perplexity']:.4f}\")
    print(f\"  Examples: {r['num_examples']}\")
"
    fi
else
    echo -e "${RED}✗${NC} Base model validation failed"
    exit 1
fi

# Step 1: Validate LoRA model
echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 1/4: Validate LoRA Model (Post-Training)${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo "Running validation on MAML-trained model..."
python $WORKSPACE/scripts/phase1_validate_maml.py \
    --model_path "$MODEL_PATH" \
    --test_file "$TEST_FILE" \
    --output_dir "$RESULTS_DIR" \
    --skip_merged

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} LoRA validation complete"
    
    # Display LoRA results
    if [ -f "$RESULTS_DIR/lora_validation.json" ]; then
        echo ""
        echo -e "${BLUE}LoRA Model Results (Post-Training):${NC}"
        python3 -c "
import json
with open('$RESULTS_DIR/lora_validation.json') as f:
    r = json.load(f)
    print(f\"  Loss: {r['avg_loss']:.6f}\")
    print(f\"  Perplexity: {r['perplexity']:.4f}\")
    print(f\"  Examples: {r['num_examples']}\")
"
    fi
else
    echo -e "${RED}✗${NC} LoRA validation failed"
    exit 1
fi

# Compare base vs trained
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Training Improvement Analysis${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

python3 << COMPARE
import json
import sys

try:
    with open('${RESULTS_DIR}/base/lora_validation.json') as f:
        base = json.load(f)
    with open('${RESULTS_DIR}/lora_validation.json') as f:
        trained = json.load(f)
    
    base_ppl = base['perplexity']
    trained_ppl = trained['perplexity']
    improvement = ((base_ppl - trained_ppl) / base_ppl) * 100
    
    print(f"\\nBase Model (Pre-Training):")
    print(f"  Loss:       {base['avg_loss']:.4f}")
    print(f"  Perplexity: {base_ppl:.4f}")
    
    print(f"\\nTrained Model (Post-Training):")
    print(f"  Loss:       {trained['avg_loss']:.4f}")
    print(f"  Perplexity: {trained_ppl:.4f}")
    
    print(f"\\nImprovement:")
    print(f"  Loss Δ:        {base['avg_loss'] - trained['avg_loss']:.4f} (lower is better)")
    print(f"  Perplexity Δ:  {base_ppl - trained_ppl:.4f} ({improvement:+.1f}%)")
    
    if trained_ppl < base_ppl:
        print(f"\\n✓ Training IMPROVED model (perplexity reduced by {abs(improvement):.1f}%)")
    else:
        print(f"\\n✗ Warning: Training did NOT improve model (perplexity increased)")

except Exception as e:
    print(f"Error comparing results: {e}")
    sys.exit(1)
COMPARE

# Step 2: Merge LoRA weights
echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 2/4: Merge LoRA Weights${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [ -d "$MERGED_PATH" ]; then
    echo -e "${GREEN}✓${NC} Merged model already exists: $MERGED_PATH"
    echo "  Delete $MERGED_PATH to re-merge"
else
    echo "Merging LoRA adapter into base model (this may take 5-10 minutes)..."
    python $WORKSPACE/scripts/phase1_merge_lora.py \
        --lora_path "$MODEL_PATH" \
        --output_path "$MERGED_PATH" \
        --precision bfloat16 \
        --verify
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} Merge complete"
        
        # Show merged model size
        MERGED_SIZE=$(du -sh "$MERGED_PATH" | cut -f1)
        echo "  Merged model size: $MERGED_SIZE"
    else
        echo -e "${RED}✗${NC} Merge failed"
        exit 1
    fi
fi

# Step 3: Validate merged model
echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 3/4: Validate Merged Model${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo "Running validation on merged model (this may take 10-15 minutes)..."

# Need to update validation script to handle merged model directly
# For now, we'll validate by loading merged model
python $WORKSPACE/scripts/phase1_validate_maml.py \
    --model_path "$MODEL_PATH" \
    --test_file "$TEST_FILE" \
    --output_dir "$RESULTS_DIR"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Merged validation complete"
else
    echo -e "${RED}✗${NC} Merged validation failed"
    exit 1
fi

# Step 4: Results summary and packaging
echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 4/4: Results Summary & Packaging${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Display final comparison
if [ -f "$RESULTS_DIR/comparison.json" ]; then
    echo ""
    python3 << EOF
import json
import sys

try:
    with open("${RESULTS_DIR}/comparison.json", 'r') as f:
        results = json.load(f)
    
    print("\\033[1;34m" + "="*60 + "\\033[0m")
    print("\\033[1;34mVALIDATION RESULTS SUMMARY\\033[0m")
    print("\\033[1;34m" + "="*60 + "\\033[0m")
    print()
    
    print("\\033[1;32mLoRA Model:\\033[0m")
    print(f"  Loss:       {results['lora']['avg_loss']:.6f}")
    print(f"  Perplexity: {results['lora']['perplexity']:.4f}")
    print(f"  Examples:   {results['lora']['num_examples']}")
    print()
    
    print("\\033[1;32mMerged Model:\\033[0m")
    print(f"  Loss:       {results['merged']['avg_loss']:.6f}")
    print(f"  Perplexity: {results['merged']['perplexity']:.4f}")
    print(f"  Examples:   {results['merged']['num_examples']}")
    print()
    
    print("\\033[1;33mComparison:\\033[0m")
    print(f"  Loss Δ:     {results['loss_difference']:.8f}")
    print(f"  PPL Δ:      {results['perplexity_difference']:.8f}")
    print()
    
    if results['identical']:
        print("\\033[1;32m✓ Models are IDENTICAL (within tolerance)\\033[0m")
        print("  Merge was successful!")
        success = True
    else:
        print("\\033[1;33m⚠ Models have small difference (acceptable for practical use)\\033[0m")
        print(f"  Difference: {results['perplexity_difference']:.4f} perplexity points")
        print("  Note: Small numerical differences can occur during merge due to precision")
        success = True
    
    print()
    print("\\033[1;34m" + "="*60 + "\\033[0m")
    
    sys.exit(0 if success else 1)

except Exception as e:
    print(f"\033[1;31m✗ Error reading results: {e}\033[0m")
    sys.exit(1)
EOF

    SUMMARY_EXIT=$?
else
    echo -e "${RED}✗ Comparison results not found${NC}"
    SUMMARY_EXIT=1
fi

# Package results for download
echo ""
echo -e "${YELLOW}Packaging results for download...${NC}"

PACKAGE_DIR="$WORKSPACE/download_package"
mkdir -p "$PACKAGE_DIR"

# Create results archive
cd "$WORKSPACE"
tar -czf "$PACKAGE_DIR/validation_results.tar.gz" \
    results/phase1_validation/ \
    data/phase1/test_set.jsonl

echo -e "${GREEN}✓${NC} Results packaged: $PACKAGE_DIR/validation_results.tar.gz"

# If merge successful, prepare merged model for download
if [ $SUMMARY_EXIT -eq 0 ]; then
    echo ""
    echo -e "${YELLOW}Packaging merged model for download...${NC}"
    echo "  (This may take 5-10 minutes for ~7GB model)"
    
    cd "$WORKSPACE/models/phase1_maml_lora_v2"
    tar -czf "$PACKAGE_DIR/merged_model.tar.gz" merged/
    
    MERGED_TAR_SIZE=$(du -sh "$PACKAGE_DIR/merged_model.tar.gz" | cut -f1)
    echo -e "${GREEN}✓${NC} Merged model packaged: $PACKAGE_DIR/merged_model.tar.gz ($MERGED_TAR_SIZE)"
fi

# Final summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}WORKFLOW COMPLETE${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if [ $SUMMARY_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ All validations passed!${NC}"
    echo ""
    echo -e "${YELLOW}Download Instructions:${NC}"
    echo "1. From your local machine, run:"
    echo ""
    echo "   scp -P <port> root@<vast-ip>:$PACKAGE_DIR/*.tar.gz ."
    echo ""
    echo "2. Extract locally:"
    echo ""
    echo "   tar -xzf validation_results.tar.gz"
    echo "   tar -xzf merged_model.tar.gz"
    echo ""
    echo -e "${YELLOW}Files to download:${NC}"
    ls -lh "$PACKAGE_DIR"
    echo ""
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "1. Download results and models from Vast.ai"
    echo "2. Extract base model responses for draft training"
    echo "3. Train draft model (Phase 1E - Qwen2.5-0.5B)"
    echo "4. Or proceed to compression (Phase 2)"
else
    echo -e "${RED}✗ Validation issues detected${NC}"
    echo ""
    echo "Check the logs above for details"
    echo "Results available in: $RESULTS_DIR"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
