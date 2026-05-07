#!/bin/bash
#SBATCH --gres=gpu:2 --ntasks=1 --cpus-per-task=16 --mem=64G -p 5090
#SBATCH --job-name=multimodalqa_complete
#SBATCH --output=multimodalqa_complete_%j.out
#SBATCH --error=multimodalqa_complete_%j.err

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "============================================"
echo "MultiModalQA完整多模态检索实验 - 10样本"
echo "开始时间: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "输出目录: ${PWD}/results_multimodalqa_complete"
echo "============================================"

# 创建输出目录
mkdir -p results_multimodalqa_complete

# 运行MultiModalQA 10样本测试（完整多模态检索）
echo -e "\n开始运行MultiModalQA 10样本测试（完整多模态检索：BGE + CLIP）..."
srun --ntasks=1 --gres=gpu:2 python run_all_baselines_MultimodalVQA.py \
    --max_samples 10 \
    --split dev \
    --output_dir results_multimodalqa_complete

echo -e "\n实验完成时间: $(date)"
echo "所有结果保存在: ${PWD}/results_multimodalqa_complete"