#!/bin/bash
#SBATCH --gres=gpu:2 --ntasks=1 --cpus-per-task=32 --mem=128G -p 5090
#SBATCH --job-name=build_multimodal_indexes
#SBATCH --output=build_multimodal_indexes_%j.out
#SBATCH --error=build_multimodal_indexes_%j.err
#SBATCH --time=12:00:00

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "============================================"
echo "构建MultiModalQA完整多模态索引"
echo "开始时间: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "输出目录: /data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa"
echo "============================================"

# 创建索引目录
mkdir -p /data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa

# 运行索引构建脚本
python build_multimodalqa_indexes_complete.py

echo -e "\n索引构建完成时间: $(date)"
echo "所有索引保存在: /data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa"