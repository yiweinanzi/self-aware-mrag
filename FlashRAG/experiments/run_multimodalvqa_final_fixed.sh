#!/bin/bash
#SBATCH --gres=gpu:2 --ntasks=1 --cpus-per-task=16 --mem=64G -p 5090
#SBATCH --job-name=multimodalvqa_final
#SBATCH --output=/data0/home/zqwang/ACL/multimodalvqa_final_%j.out
#SBATCH --error=/data0/home/zqwang/ACL/multimodalvqa_final_%j.err

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "============================================"
echo "MultiModalQA 最终实验（10样本）"
echo "开始时间: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "============================================"

# 运行10样本实验
echo -e "\n开始运行最终实验..."
srun --gres=gpu:2 python run_all_baselines_MultimodalVQA.py \
    --max_samples 10 \
    --use_dataset_docs \
    --output_dir /data0/home/zqwang/ACL/results_multimodalqa_final

echo -e "\n实���完成时间: $(date)"
echo "结果保存在: /data0/home/zqwang/ACL/results_multimodalqa_final"
echo "============================================"
