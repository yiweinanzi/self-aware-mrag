#!/bin/bash
#SBATCH --gres=gpu:2 --ntasks=1 --cpus-per-task=16 --mem=64G -p 5090
#SBATCH --job-name=test_reranker_fixed
#SBATCH --output=test_reranker_fixed_%j.out
#SBATCH --error=test_reranker_fixed_%j.err

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "============================================"
echo "测试BGE Reranker对MRAG-Bench的影响"
echo "开始时间: $(date)"
echo "============================================"

# 运行MRAG实验（启用reranker）
echo -e "\n开始运行MRAG实验..."
srun --gres=gpu:2 python run_all_baselines_MRAG.py \
    --max_samples 100

echo -e "\n实验完成时间: $(date)"
echo "结果保存在: ${PWD}/results_mrag_with_reranker"