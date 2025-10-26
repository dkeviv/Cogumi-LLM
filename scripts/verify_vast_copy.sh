#!/bin/bash
# Quick verification script for new Vast.ai instance

echo "🔍 Checking for copied data..."
echo "================================"

# Check possible locations
echo -e "\n📂 Checking /workspace/Cogumi-LLM/:"
if [ -d "/workspace/Cogumi-LLM" ]; then
    echo "✅ Found /workspace/Cogumi-LLM"
    ls -lh /workspace/Cogumi-LLM/
else
    echo "❌ Not found at /workspace/Cogumi-LLM"
fi

echo -e "\n📂 Checking /data/Cogumi-LLM/:"
if [ -d "/data/Cogumi-LLM" ]; then
    echo "✅ Found /data/Cogumi-LLM"
    ls -lh /data/Cogumi-LLM/
else
    echo "❌ Not found at /data/Cogumi-LLM"
fi

echo -e "\n🔍 Searching for checkpoints..."
find / -name "checkpoint-240240" -type d 2>/dev/null | head -5

echo -e "\n📊 Disk usage:"
df -h

echo -e "\n✅ Verification complete!"
