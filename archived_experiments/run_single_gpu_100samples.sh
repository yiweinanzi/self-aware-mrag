#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multirag

echo "=== 单GPU 100样本真实模型验证测试 ==="
echo "目标：验证修复后准确率是否从1.3%提升到50%+"
echo "样本：100个OK-VQA样本"
echo "模型：真实Qwen3-VL-8B"
echo "时间: $(date)"
echo ""

# 运行单GPU测试
srun -n1 --partition=5090 --gres=gpu:1 \
     --cpus-per-task=8 \
     --mem=32G \
     --time=02:00:00 \
     --job-name=single_gpu_100samples \
     python /data0/home/zqwang/ACL/test_single_gpu_100samples.py