#!/bin/bash
#SBATCH --gres=gpu:2 --ntasks=1 --cpus-per-task=16 --mem=64G -p 5090
#SBATCH --job-name=multimodalvqa_test_support
#SBATCH --output=/data0/home/zqwang/ACL/multimodalvqa_test_support_%j.out
#SBATCH --error=/data0/home/zqwang/ACL/multimodalvqa_test_support_%j.err

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "============================================"
echo "MultiModalQA 测试 - 支持度验证优化"
echo "开始时间: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "============================================"

# 只运行1个样本测试
echo -e "\n开始运行测试..."
srun --gres=gpu:2 python run_all_baselines_MultimodalVQA.py \
    --max_samples 1 \
    --use_dataset_docs \
    --output_dir /data0/home/zqwang/ACL/results_multimodalqa_test_support

echo -e "\n实验完成时间: $(date)"
echo "============================================"