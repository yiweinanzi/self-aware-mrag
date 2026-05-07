#!/bin/bash
#SBATCH --gres=gpu:1 --ntasks=1 --cpus-per-task=8 --mem=32G -p 5090
#SBATCH --job-name=multimodalvqa_debug
#SBATCH --output=/data0/home/zqwang/ACL/multimodalvqa_debug_%j.out
#SBATCH --error=/data0/home/zqwang/ACL/multimodalvqa_debug_%j.err

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "============================================"
echo "MultiModalQA 调试测试 - 查看Prompt"
echo "开始时间: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "============================================"

# 只运行1个样本查看prompt
echo -e "\n开始运行调试测试..."
srun --gres=gpu:1 python run_all_baselines_MultimodalVQA.py \
    --max_samples 1 \
    --use_dataset_docs \
    --output_dir /data0/home/zqwang/ACL/results_multimodalqa_debug_prompt

echo -e "\n实验完成时间: $(date)"
echo "============================================"