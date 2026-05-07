#!/bin/bash
# 2GPU版本100样本消融实验脚本
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
LOG_FILE="$LOG_DIR/ablation_2gpu_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/ablation_2gpu_${TIMESTAMP}.pid"

echo "================================================================================"
echo "2GPU 100样本消融实验 - Qwen3-VL + FAISS检索"
echo "开始时间: $(date)"
echo "GPU数量: 2 x RTX 5090 (64GB total)"
echo "日志文件: $LOG_FILE"
echo "================================================================================"

# 启动2GPU实验
echo "启动2GPU 100样本消融实验..."
nohup srun --partition=5090 --gres=gpu:2 python -c "
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

print(f'检测到GPU数量: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory/1024**3:.1f}GB)')

# 导入并运行实验
exec(open('run_real_model_ablation.py').read())
" > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"

echo "实验已启动！"
echo "PID: $(cat $PID_FILE)"
echo "监控日志: tail -f $LOG_FILE"
echo ""
echo "使用以下命令监控进度:"
echo "  tail -f $LOG_FILE"
echo "  grep -E '🔄|✅|❌|准确率|样本|GPU|device' $LOG_FILE"
echo ""
echo "停止实验:"
echo "  kill \$(cat $PID_FILE)"
echo ""
echo "================================================================================"