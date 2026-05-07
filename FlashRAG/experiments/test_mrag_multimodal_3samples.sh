#!/bin/bash
#SBATCH --gres=gpu:2 --ntasks=1 --cpus-per-task=16 --mem=64G -p 5090
#SBATCH --job-name=mrag_multimodal_3samples
#SBATCH --output=mrag_multimodal_3samples_%j.out
#SBATCH --error=mrag_multimodal_3samples_%j.err

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "============================================"
echo "MRAG-Bench 3样本测试（验证多模态检索）"
echo "开始时间: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "============================================"

# 运行MRAG实验（3个样本，专注于验证多模态检索）
echo -e "\n开始运行MRAG实验（3个样本）..."
srun --gres=gpu:2 python run_all_baselines_MRAG.py \
    --max_samples 3

echo -e "\n实验完成时间: $(date)"
echo "结果保存在: ${PWD}/results_mrag_with_reranker"
echo "============================================"