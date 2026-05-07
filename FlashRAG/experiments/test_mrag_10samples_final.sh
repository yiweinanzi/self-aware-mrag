#!/bin/bash
#SBATCH --gres=gpu:2 --ntasks=1 --cpus-per-task=16 --mem=64G -p 5090
#SBATCH --job-name=mrag_10samples_final
#SBATCH --output=mrag_10samples_final_%j.out
#SBATCH --error=mrag_10samples_final_%j.err

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "============================================"
echo "MRAG-Bench 10样本最终测试（多模态检索+Reranker）"
echo "开始时间: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "============================================"

# 运行MRAG实验（10个样本）
echo -e "\n开始运行MRAG实验（10个样本）..."
srun --gres=gpu:2 python run_all_baselines_MRAG.py \
    --max_samples 10

echo -e "\n实验完成时间: $(date)"
echo "结果保存在: ${PWD}/results_mrag_with_reranker"
echo "============================================"