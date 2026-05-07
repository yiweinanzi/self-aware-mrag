#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "=== 优化2GPU全真实Qwen3-VL并行实验 ==="
echo "架构：2个GPU全部使用真实Qwen3-VL-8B模型"
echo "优势：确保充足内存，100%真实模型推理"
echo "数据集：2000个OK-VQA样本（比4GPU版本更多）"
echo "时间: $(date)"
echo ""

# 运行优化2GPU全真实Qwen3-VL实验
srun -n1 --partition=5090 --gpus=2 \
     --cpus-per-task=12 \
     --mem=80G \
     --time=06:00:00 \
     --job-name=2gpu_real_qwen3vl \
     python run_2gpu_real_qwen3vl.py