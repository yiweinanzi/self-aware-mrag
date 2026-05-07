#!/bin/bash
# 实验监控脚本

echo "=========================================="
echo "实验监控 - $(date '+%H:%M:%S')"
echo "=========================================="

# 检查进程
if ps aux | grep -q "[r]un_all_baselines_100samples.py"; then
    echo "✅ 实验进程运行中"
else
    echo "❌ 实验进程未运行"
fi

echo ""
echo "最新进度："
tail -100 /root/autodl-tmp/optimized_100samples.log | grep -E "(运行 Self-Aware.*%\|)" | tail -1

echo ""
echo "不确定性统计（最近10个样本）："
tail -200 /root/autodl-tmp/optimized_100samples.log | grep "uncertainty=" | tail -10 | \
    awk '{print $1}' | grep -oP 'uncertainty=\K[0-9.]+' | \
    awk '{sum+=$1; count++} END {if(count>0) printf "平均: %.4f, 样本数: %d\n", sum/count, count}'

echo ""
echo "检索决策统计："
tail -200 /root/autodl-tmp/optimized_100samples.log | grep "should_retrieve=" | tail -10 | \
    grep -o "should_retrieve=[a-zA-Z]*" | sort | uniq -c

echo ""
echo "=========================================="

