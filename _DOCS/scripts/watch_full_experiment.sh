#!/bin/bash
# 全数据集实验实时监控

LOG_FILE="/root/autodl-tmp/full_dataset_experiment.log"

while true; do
    clear
    echo "================================================"
    echo "🔍 MRAG-Bench 全数据集实验 (1353样本)"
    echo "================================================"
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # 进程状态
    if ps aux | grep -q "[p]ython.*run_all_baselines"; then
        UPTIME=$(ps -p $(pgrep -f "run_all_baselines") -o etime= | tr -d ' ')
        echo "✅ 实验运行中 | 时长: $UPTIME"
    else
        echo "❌ 实验未运行"
    fi
    
    echo ""
    echo "------------------------------------------------"
    echo "📊 最新进度"
    echo "------------------------------------------------"
    
    # 当前方法
    tail -100 "$LOG_FILE" | grep "评测方法:" | tail -1
    
    # 进度条
    tail -50 "$LOG_FILE" | grep "运行.*%\|" | tail -3
    
    echo ""
    echo "------------------------------------------------"
    echo "📈 已完成方法"
    echo "------------------------------------------------"
    grep "✅.*完成:" "$LOG_FILE" | tail -10
    
    echo ""
    echo "================================================"
    echo "按 Ctrl+C 退出监控"
    echo "================================================"
    
    sleep 60
done

