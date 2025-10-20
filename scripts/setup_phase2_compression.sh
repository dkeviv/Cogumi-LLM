#!/bin/bash
# Phase 2 Compression Environment Setup
# Run this while Phase 1 training is happening

set -e

echo "=================================="
echo "Phase 2 Compression Setup"
echo "=================================="

# Navigate to project root
cd "$(dirname "$0")/.."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo ""
echo "1. Installing Neural Magic SparseML..."
pip install sparseml[transformers]

echo ""
echo "2. Installing AutoAWQ..."
pip install autoawq

echo ""
echo "3. Installing llama.cpp for GGUF conversion..."
if [ ! -d "tools/llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp tools/llama.cpp
    cd tools/llama.cpp
    make
    cd ../..
else
    echo "   llama.cpp already exists, pulling latest..."
    cd tools/llama.cpp
    git pull
    make
    cd ../..
fi

echo ""
echo "4. Installing compression utilities..."
pip install zstandard
pip install onnx onnxruntime

echo ""
echo "5. Verifying installations..."
python -c "import sparseml; print(f'✅ SparseML {sparseml.__version__}')"
python -c "import awq; print(f'✅ AutoAWQ installed')"
python -c "import zstandard; print(f'✅ Zstandard {zstandard.__version__}')"

echo ""
echo "=================================="
echo "✅ Phase 2 environment ready!"
echo "=================================="
echo ""
echo "Next steps after training completes:"
echo "  1. Run: python src/phase2_compression/neural_magic_prune.py"
echo "  2. Run: python src/phase2_compression/awq_quantize.py"
echo "  3. Run: python src/phase2_compression/gguf_export.py"
