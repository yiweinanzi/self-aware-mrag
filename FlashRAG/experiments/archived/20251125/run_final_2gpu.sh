#!/bin/bash
# 最终正确版本2GPU��验
# Author: Claude Code
# Date: 2025-11-25

set -e

echo "================================================================================"
echo "最终正确版本 - 2GPU Qwen3-VL实验"
echo "开始时间: $(date)"
echo "================================================================================"

# 创建日志目录
LOG_DIR="/data0/home/zqwang/ACL/FlashRAG/experiments/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/final_2gpu_${TIMESTAMP}.log"

# 创建Python脚本
cat > /tmp/final_2gpu.py << 'EOF'
import sys
import os
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

# 设置环境
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

print("=== 环境检查 ===")
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory = props.total_memory / 1024**3
        print(f"GPU {i}: {torch.cuda.get_device_name(i)} ({memory:.1f}GB)")
else:
    print("❌ CUDA不可用!")
    sys.exit(1)

print("=== 导入FlashRAG模块 ===")
from run_real_model_ablation import CONFIG, AblationExperiment

# 关键配置修改
CONFIG['load_in_8bit'] = False  # 避免编译器问题
CONFIG['max_samples'] = 20      # 先用20个样本测试
CONFIG['max_new_tokens'] = 5    # 减少token数
CONFIG['torch_dtype'] = 'float16'  # 使用float16

print(f"配置: 8bit={CONFIG['load_in_8bit']}, 样本数={CONFIG['max_samples']}")
print("=== 开始实验 ===")

try:
    experiment = AblationExperiment(CONFIG)
    experiment.run_experiment()
    print("=== 实验完成 ===")
except Exception as e:
    print(f"❌ 实验失败: {e}")
    import traceback
    traceback.print_exc()
EOF

echo "启动最终2GPU实验..."
srun --partition=5090 --gres=gpu:2 bash -c "
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments
python /tmp/final_2gpu.py
" > "$LOG_FILE" 2>&1

echo "实验已启动!"
echo "日志文件: $LOG_FILE"
echo "监控: tail -f $LOG_FILE"
echo "================================================================================"