#!/bin/bash
# 100样本消融实验脚本 - 带日志输出
# Author: Claude Code
# Date: 2025-11-25

set -e

# 环境设置
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

# 创建日志目录
LOG_DIR="/data0/home/zqwang/ACL/FlashRAG/experiments/logs"
mkdir -p "$LOG_DIR"

# 日志文件
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/ablation_100samples_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/ablation_100samples_${TIMESTAMP}.pid"

echo "================================================================================"
echo "100样本消融实验 - Qwen3-VL + FAISS检索"
echo "开始时间: $(date)"
echo "日志文件: $LOG_FILE"
echo "================================================================================"

# 启动实验并将所有输出重定向到日志文件
echo "启动100样本消融实验..."
nohup srun --partition=5090 --gres=gpu:1 python run_real_model_ablation.py > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "实验已启动！"
echo "PID: $(cat $PID_FILE)"
echo "监控日志: tail -f $LOG_FILE"
echo ""
echo "使用以下命令监控进度:"
echo "  tail -f $LOG_FILE"
echo "  grep -E '🔄|✅|❌|准确率|样本' $LOG_FILE"
echo ""
echo "停止实验:"
echo "  kill \$(cat $PID_FILE)"
echo ""
echo "================================================================================"