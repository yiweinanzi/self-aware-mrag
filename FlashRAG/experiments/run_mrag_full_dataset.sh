#!/bin/bash
#SBATCH --gres=gpu:2 --ntasks=1 --cpus-per-task=16 --mem=64G -p 5090
#SBATCH --job-name=mrag_full_dataset
#SBATCH --output=mrag_full_dataset_%j.out
#SBATCH --error=mrag_full_dataset_%j.err

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "============================================"
echo "MRAG-Bench 完整数据集实验（多模态检索+Reranker）"
echo "开始时间: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "============================================"

# 运行MRAG完整数据集实验（约1353个样本）
echo -e "\n开始运行MRAG完整数据集实验..."
srun --gres=gpu:2 python run_all_baselines_MRAG.py \
    --max_samples 1353

echo -e "\n实验完成时间: $(date)"
echo "结果保存在: ${PWD}/results_mrag_with_reranker"
echo "============================================"