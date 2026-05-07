#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "=== 4卡并行真实消融实验 ==="
echo "使用4个RTX 5090 GPU并行加速"
echo "数据集：全部OK-VQA样本(5046个)"
echo "时间: $(date)"
echo ""

# 运行4卡并行实验（使用推荐的语法）
srun -n1 --partition=5090 --gpus=4 \
     --cpus-per-task=16 \
     --mem=128G \
     --time=6:00:00 \
     --job-name=real_ablation_4gpu \
     python run_real_model_ablation_4gpu.py