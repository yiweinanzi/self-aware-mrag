#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "=== 优化4GPU并行真实消融实验 ==="
echo "内存优化 + 模型共享 + 真实数据集"
echo "数据集：全部OK-VQA样本(5046个)"
echo "批次大小：2 (内存优化)"
echo "时间: $(date)"
echo ""

# 设置CUDA内存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 运行优化的4GPU实验
srun -n1 --partition=5090 --gpus=4 \
     --cpus-per-task=16 \
     --mem=128G \
     --time=8:00:00 \
     --job-name=optimized_4gpu_ablation \
     python run_real_model_ablation_optimized_4gpu.py