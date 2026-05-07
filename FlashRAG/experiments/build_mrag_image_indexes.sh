#!/bin/bash
#SBATCH --gres=gpu:1 --ntasks=1 --cpus-per-task=16 --mem=32G -p 5090
#SBATCH --job-name=build_mrag_image_index
#SBATCH --output=build_mrag_image_index_%j.out
#SBATCH --error=build_mrag_image_index_%j.err
#SBATCH --time=04:00:00

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "============================================"
echo "构建MRAG-Bench图像索引"
echo "开始时间: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "============================================"

# 创建输出目录
mkdir -p /data0/home/zqwang/ACL/FlashRAG/indexes/mrag_bench

# 运行索引构建
python build_mrag_image_indexes.py

echo -e "\n索引构建完成时间: $(date)"
echo "索引保存在: /data0/home/zqwang/ACL/FlashRAG/indexes/mrag_bench/"