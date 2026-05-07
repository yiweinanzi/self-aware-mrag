#!/bin/bash

echo "=========================================="
echo "监控OK-VQA全样本实验进度"
echo "=========================================="

LOG_FILE="/data0/home/zqwang/ACL/full_baselines_experiment_20251217_140254.log"
BASE_LOG="/data0/home/zqwang/ACL/FlashRAG/experiments/results_okvqa_baselines/baselines_20251217_220256.log"

while true; do
    echo ""
    echo "更新时间: $(date)"
    echo "----------------------------------------"

    # 检查主日志
    if [ -f "$LOG_FILE" ]; then
        echo "最新进度:"
        tail -10 "$LOG_FILE" | grep -E "(进度:|完成:|准确率:|✅)" | tail -5

        # 检查是否完成了任何方法
        echo ""
        echo "已完成的方法:"
        grep "准确率:" "$LOG_FILE" 2>/dev/null | tail -10
    else
        echo "日志文件尚未创建"
    fi

    # 检查基础日志
    if [ -f "$BASE_LOG" ]; then
        echo ""
        echo "当前方法详情:"
        tail -20 "$BASE_LOG" | grep -E "(实验完成|准确率:)" | tail -2
    fi

    echo ""
    echo "等待30秒后再次检查..."
    sleep 30
done