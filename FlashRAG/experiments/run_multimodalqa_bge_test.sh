#!/bin/bash
#SBATCH --gres=gpu:2 --ntasks=1 --cpus-per-task=16 --mem=64G -p 5090
#SBATCH --job-name=multimodalqa_bge_test
#SBATCH --output=multimodalqa_bge_test_%j.out
#SBATCH --error=multimodalqa_bge_test_%j.err

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "============================================"
echo "MultiModalQA BGE检索实验 - 10样本"
echo "开始时间: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "输出目录: ${PWD}/results_multimodalqa_bge_test"
echo "============================================"

# 创建输出目录
mkdir -p results_multimodalqa_bge_test

# 运行MultiModalQA 10样本测试（使用BGE索引）
echo -e "\n开始运行MultiModalQA 10样本测试（BGE检索）..."
srun --ntasks=1 --gres=gpu:2 python run_all_baselines_MultimodalVQA.py \
    --max_samples 10 \
    --split dev \
    --output_dir results_multimodalqa_bge_test \
    --use_dataset_docs

echo -e "\n实验完成时间: $(date)"
echo "所有结果保存在: ${PWD}/results_multimodalqa_bge_test"