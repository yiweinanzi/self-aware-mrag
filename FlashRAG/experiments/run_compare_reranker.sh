#!/bin/bash
#SBATCH --gres=gpu:1 --ntasks=1 --cpus-per-task=8 --mem=32G -p 5090
#SBATCH --job-name=compare_reranker
#SBATCH --output=compare_reranker_%j.out
#SBATCH --error=compare_reranker_%j.err

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "============================================"
echo "比较BGE检索器使用和不使用Reranker的效果"
echo "开始时间: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "============================================"

# 运行比较测试
srun --gres=gpu:1 python compare_reranker.py

echo -e "\n实验完成时间: $(date)"
echo "============================================"