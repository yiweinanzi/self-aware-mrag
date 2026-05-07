#!/bin/bash
echo "🔍 实验快速检查 - $(date '+%H:%M:%S')"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
tail -100 /root/autodl-tmp/optimized_100samples.log | grep "运行 Self-Aware" | tail -1
echo ""
tail -20 /root/autodl-tmp/optimized_100samples.log | grep "uncertainty=" | tail -3
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
