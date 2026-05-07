#!/bin/bash
#SBATCH --gres=gpu:2 --ntasks=1 --cpus-per-task=16 --mem=64G -p 5090
#SBATCH --job-name=mrag_with_reranker
#SBATCH --output=mrag_with_reranker_%j.out
#SBATCH --error=mrag_with_reranker_%j.err

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "============================================"
echo "MRAG-Bench实验 - 使用BGE Reranker"
echo "开始时间: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "============================================"

# 运行MRAG实验（启用reranker）
echo -e "\n开始运行MRAG实验（使用BGE Reranker）..."
srun --gres=gpu:2 python run_all_baselines_MRAG.py \
    --max_samples 100 \
    --output_dir results_mrag_with_reranker

echo -e "\n实验完成时间: $(date)"
echo "结果保存在: ${PWD}/results_mrag_with_reranker"
echo "============================================"
echo "配置说明："
echo "- 使用BGE Reranker v2-M3模型"
echo "- Reranker批次大小: 32"
echo "- Reranker最大长度: 512"
echo "- 启用fp16加速"