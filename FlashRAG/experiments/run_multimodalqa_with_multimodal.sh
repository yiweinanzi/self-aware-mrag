#!/bin/bash
#SBATCH --gres=gpu:2 --ntasks=1 --cpus-per-task=16 --mem=64G -p 5090
#SBATCH --job-name=multimodalqa_multimodal
#SBATCH --output=multimodalqa_multimodal_%j.out
#SBATCH --error=multimodalqa_multimodal_%j.err

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "============================================"
echo "MultiModalQA多模态检索实验 - 10样本"
echo "开始时间: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "输出目录: ${PWD}/results_multimodalqa_multimodal"
echo "============================================"

# 创建输出目录
mkdir -p results_multimodalqa_multimodal

# 运行MultiModalQA 10样本测试（多模态检索）
echo -e "\n开始运行MultiModalQA 10样本测试（多模态检索）..."
# 使用 --ntasks=1 确保只有一个进程运行，避免重复日志
# 使用 --use_dataset_docs 启用数据集文档模式
# 使用 --use_multimodal_retrieval 启用多模态检索
srun --ntasks=1 --gres=gpu:2 python run_all_baselines_MultimodalVQA_multimodal.py \
    --max_samples 10 \
    --split dev \
    --output_dir results_multimodalqa_multimodal \
    --use_dataset_docs \
    --use_multimodal_retrieval

echo -e "\n实验完成时间: $(date)"
echo "所有结果保存在: ${PWD}/results_multimodalqa_multimodal"