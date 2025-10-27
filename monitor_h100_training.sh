#!/bin/bash
# H100 Training Monitor - Run from your Mac
# Updates every 30 seconds with training progress, GPU stats, and disk usage

VAST_IP="115.124.123.238"
VAST_PORT="17831"

echo "🔍 H100 Training Monitor"
echo "Instance: $VAST_IP:$VAST_PORT"
echo "Press Ctrl+C to stop"
echo "=========================="
echo ""

while true; do
    clear
    echo "⏰ $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================="
    echo ""
    
    echo "📊 GPU Status:"
    ssh -p $VAST_PORT root@$VAST_IP "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits" 2>/dev/null | awk -F, '{printf "   GPU Util: %s%% | Memory: %s/%s MB | Temp: %s°C | Power: %s W\n", $1, $2, $3, $4, $5}'
    echo ""
    
    echo "🚀 Latest Training Output (last 25 lines):"
    ssh -p $VAST_PORT root@$VAST_IP "tmux capture-pane -t training -p 2>/dev/null | tail -25" 2>/dev/null || echo "   ⚠️  Could not retrieve training output"
    echo ""
    
    echo "💾 Disk Space:"
    ssh -p $VAST_PORT root@$VAST_IP "df -h /data 2>/dev/null | tail -1" 2>/dev/null | awk '{printf "   Used: %s/%s (%s) | Available: %s\n", $3, $2, $5, $4}'
    echo ""
    
    echo "📁 Recent Checkpoints:"
    ssh -p $VAST_PORT root@$VAST_IP "ls -lht /data/Cogumi-LLM/checkpoints/ 2>/dev/null | grep checkpoint | head -3" 2>/dev/null || echo "   No checkpoints yet"
    echo ""
    
    echo "🔄 Training Process Status:"
    ssh -p $VAST_PORT root@$VAST_IP "ps aux | grep '[t]rain.py' >/dev/null 2>&1 && echo '   ✅ Training process is RUNNING' || echo '   ❌ Training process NOT found'" 2>/dev/null
    echo ""
    
    echo "Next update in 30 seconds... (Ctrl+C to stop)"
    sleep 30
done
