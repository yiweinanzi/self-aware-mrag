#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "=== 优化真实消融实验 ==="
echo "使用高性能配置的RTX 5090 GPU"
echo "数据集：全部OK-VQA样本(5046个)"
echo "批次大小：8 (优化并行处理)"
echo "时间: $(date)"
echo ""

# 运行优化的单GPU实验
srun --partition=5090 \
     --gres=gpu:1 \
     --ntasks=1 \
     --cpus-per-task=12 \
     --mem=96G \
     --time=10:00:00 \
     --job-name=optimized_real_ablation \
     python run_real_model_ablation_optimized.py