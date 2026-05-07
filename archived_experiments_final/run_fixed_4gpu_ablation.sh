#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "=== 修复后的4GPU并行真实模型消融实验 ==="
echo "已修复：SimpleLLM模块 + 数据字段错误"
echo "数据集：全部OK-VQA样本(5046个)"
echo "架构：1个GPU完整模型 + 3个GPU简化模型"
echo "时间: $(date)"
echo ""

# 运行优化版4GPU实验（修复后）
srun -n1 --partition=5090 --gpus=4 \
     --cpus-per-task=16 \
     --mem=128G \
     --time=10:00:00 \
     --job-name=fixed_4gpu_ablation \
     python run_real_model_ablation_optimized_4gpu.py