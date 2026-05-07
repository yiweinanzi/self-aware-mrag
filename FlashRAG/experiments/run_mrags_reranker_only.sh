#!/bin/bash
#SBATCH --gres=gpu:2 --ntasks=1 --cpus-per-task=16 --mem=64G -p 5090
#SBATCH --job-name=mrags_reranker_only
#SBATCH --output=mrags_reranker_only_%j.out
#SBATCH --error=mrags_reranker_only_%j.err

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "============================================"
echo "MRAG-Bench Reranker测试（仅文本检索）"
echo "开始时间: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "============================================"

# 运行测试
echo -e "\n开始运行MRAG-Bench Reranker测试..."
srun --gres=gpu:2 python test_mrags_reranker_only.py

echo -e "\n实验完成时间: $(date)"
echo "============================================"