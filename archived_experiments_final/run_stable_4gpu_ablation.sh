#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "=== 稳定4GPU并行消融实验 ==="
echo "重新设计 - 简单 - 稳定 - 高效"
echo "数据集：全部OK-VQA样本(5046个)"
echo "架构：1个GPU完整模型 + 3个GPU简化模型"
echo "时间: $(date)"
echo ""

# 运行稳定的4GPU实验
srun -n1 --partition=5090 --gpus=4 \
     --cpus-per-task=16 \
     --mem=128G \
     --time=10:00:00 \
     --job-name=stable_4gpu_ablation \
     python run_stable_4gpu_ablation.py