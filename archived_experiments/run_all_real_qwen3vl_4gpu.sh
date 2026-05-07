#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "=== 全真实Qwen3-VL 4GPU并行实验 ==="
echo "架构：4个GPU全部使用真实Qwen3-VL-8B模型"
echo "优势：真正的���行推理，无准确率稀释"
echo "数据集：1000个OK-VQA样本（测试）"
echo "时间: $(date)"
echo ""

# 运行全真实Qwen3-VL 4GPU实验
srun -n1 --partition=5090 --gpus=4 \
     --cpus-per-task=16 \
     --mem=128G \
     --time=08:00:00 \
     --job-name=all_real_qwen3vl_4gpu \
     python run_all_real_qwen3vl_4gpu.py