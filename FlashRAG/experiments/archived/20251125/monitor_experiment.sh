#!/bin/bash
# -*- coding: utf-8 -*-
# 消融实验监控脚本（bash版本）

set -e

echo "================================================================="
echo "消融实验监控"
echo "================================================================="
echo "开始监控: $(date)"

# 激活环境
echo "🔄 激活multirag环境..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multirag

echo "✅ 环境激活成功"

# 创建监控循环
while true; do
    clear

    echo "================================================================="
    echo "消融实验实时监控"
    echo "================================================================="
    echo "当前时间: $(date)"
    echo

    # 显示作业状态
    echo "📊 SLURM作业状态:"
    squeue -u $USER
    echo

    # 显示最新的实验日志
    echo "📋 最新实验日志 (最后30行):"
    LATEST_LOG=$(ls -t /data0/home/zqwang/ACL/FlashRAG/experiments/logs/srun_main_*.out 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        echo "日志文件: $LATEST_LOG"
        echo "----------------------------------------"
        tail -30 "$LATEST_LOG"
    else
        echo "未找到实验日志文件"
    fi
    echo

    # 显示实验结果
    echo "📁 实验结果状态:"
    RESULTS_DIR="/data0/home/zqwang/ACL/FlashRAG/experiments/results_ablation_okvqa"
    if [ -d "$RESULTS_DIR" ]; then
        echo "结果目录: $RESULTS_DIR"
        echo "文件数量: $(find $RESULTS_DIR -type f | wc -l)"
        echo "最新文件:"
        find $RESULTS_DIR -type f -printf "%T@ %p\n" | sort -n | tail -5 | while read timestamp file; do
            date -d "@$(echo $timestamp | cut -d' ' -f1)" "+%Y-%m-%d %H:%M:%S"
            echo "  $(basename "$file")"
        done
    else
        echo "结果目录不存在（实验可能还在准备中）"
    fi
    echo

    # 显示GPU状态
    echo "🎮 GPU状态:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    echo

    # 显示进程状态
    echo "⚙️  实验进程状态:"
    ps aux | grep -E "(python|ablation)" | grep -v grep | head -5
    echo

    echo "-----------------------------------------------------------------"
    echo "按 Ctrl+C 退出监控"
    echo "下次更新: 30秒后"
    echo

    sleep 30
done