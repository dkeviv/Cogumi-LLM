#!/bin/bash

# Phase 0 Dataset Creation - Environment Setup Script
# This script sets up the Python environment for Phase 0 dataset creation

set -e  # Exit on error

echo "==================================="
echo "Phase 0 Dataset Creation - Setup"
echo "==================================="
echo ""

# Check Python version
echo "1. Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found Python: $python_version"

if [[ "$python_version" < "3.10" ]]; then
    echo "   ❌ ERROR: Python 3.10+ required"
    exit 1
fi
echo "   ✅ Python version OK"
echo ""

# Create virtual environment
echo "2. Creating virtual environment..."
if [ -d "venv_phase0" ]; then
    echo "   ⚠️  Virtual environment already exists. Removing..."
    rm -rf venv_phase0
fi

python3 -m venv venv_phase0
echo "   ✅ Virtual environment created"
echo ""

# Activate virtual environment
echo "3. Activating virtual environment..."
source venv_phase0/bin/activate
echo "   ✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "4. Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel -q
echo "   ✅ Package managers upgraded"
echo ""

# Install dependencies
echo "5. Installing Phase 0 dependencies..."
echo "   This may take 2-3 minutes..."
pip install -r requirements.txt -q
echo "   ✅ Dependencies installed"
echo ""

# Verify installation
echo "6. Verifying installation..."
python -c "import xxhash, datasketch, rich, jsonlines, datasets; print('   ✅ Core packages OK')"
echo ""

# Check for API keys (optional)
echo "7. Checking API keys (optional)..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "   ⚠️  OPENAI_API_KEY not set (needed for GPT-4o, GPT-4-mini)"
else
    echo "   ✅ OPENAI_API_KEY found"
fi

if [ -z "$TOGETHER_API_KEY" ]; then
    echo "   ⚠️  TOGETHER_API_KEY not set (needed for Llama-405B)"
else
    echo "   ✅ TOGETHER_API_KEY found"
fi
echo ""

# Summary
echo "==================================="
echo "Setup Complete! ✅"
echo "==================================="
echo ""
echo "Next Steps:"
echo "  1. Activate environment: source venv_phase0/bin/activate"
echo "  2. Set API keys (if needed):"
echo "     export OPENAI_API_KEY='your-key-here'"
echo "     export TOGETHER_API_KEY='your-key-here'"
echo "  3. Verify dataset:"
echo "     python verify_dataset.py --input ../data/public_500k_filtered.jsonl"
echo ""
echo "See README.md for full usage instructions."
echo ""
