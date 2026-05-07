#!/bin/bash
# 最简单的2GPU版本 - 只使用fp16，避免所有复杂性
# Author: Claude Code
# Date: 2025-11-25

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

LOG_DIR="/data0/home/zqwang/ACL/FlashRAG/experiments/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/ablation_simple_2gpu_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/ablation_simple_2gpu_${TIMESTAMP}.pid"

echo "================================================================================"
echo "最简单2GPU版本 - fp16无量化"
echo "开始时间: $(date)"
echo "GPU数量: 2 x RTX 5090 (64GB total)"
echo "策略: fp16 + 自动device_map"
echo "日志文件: $LOG_FILE"
echo "================================================================================"

# 直接运行修改后的实验脚本
cat > /tmp/run_simple_2gpu.py << 'EOF'
import sys
import os
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

# 设置2GPU环境
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import torch
print(f'GPU数量: {torch.cuda.device_count()}')

# 直接修改配置后运行原实验
from run_real_model_ablation import CONFIG

# 关键修改
CONFIG['load_in_8bit'] = False  # 关闭8bit避免编译器问题
CONFIG['max_samples'] = 50     # 减少到50样本确保成功
CONFIG['max_new_tokens'] = 5   # 减少token数

print(f"配置: 8bit={CONFIG['load_in_8bit']}, 样本数={CONFIG['max_samples']}")

# 运行实验
from run_real_model_ablation import AblationExperiment
experiment = AblationExperiment(CONFIG)
experiment.run_experiment()
EOF

echo "启动最简单2GPU实验..."
nohup srun --partition=5090 --gres=gpu:2 python /tmp/run_simple_2gpu.py > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"

echo "实验已启动！"
echo "PID: $(cat $PID_FILE)"
echo "监控: tail -f $LOG_FILE"
echo "================================================================================"