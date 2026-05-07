#!/bin/bash
#SBATCH --gres=gpu:2 --ntasks=1 --cpus-per-task=16 --mem=64G -p 5090
#SBATCH --job-name=multimodalvqa_debug
#SBATCH --output=/data0/home/zqwang/ACL/multimodalvqa_debug_%j.out
#SBATCH --error=/data0/home/zqwang/ACL/multimodalvqa_debug_%j.err

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "============================================"
echo "MultiModalQA 调试���试"
echo "开始时间: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "============================================"

# 运行MultiModalQA实验（只测试1个样本）
echo -e "\n开始运行MultiModalQA调试测试（1个样本）..."
srun --gres=gpu:2 python run_all_baselines_MultimodalVQA.py \
    --max_samples 1 \
    --use_dataset_docs \
    --output_dir /data0/home/zqwang/ACL/results_multimodalqa_debug

echo -e "\n实验完成时间: $(date)"
echo "结果保存在: /data0/home/zqwang/ACL/results_multimodalqa_debug"
echo "============================================"