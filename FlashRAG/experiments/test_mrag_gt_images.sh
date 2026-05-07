#!/bin/bash
#SBATCH --gres=gpu:2 --ntasks=1 --cpus-per-task=16 --mem=64G -p 5090
#SBATCH --job-name=test_mrag_gt
#SBATCH --output=test_mrag_gt_%j.out
#SBATCH --error=test_mrag_gt_%j.err

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "============================================"
echo "测试MRAG-Bench使用gt_images"
echo "开始时间: $(date)"
echo "============================================"

# 运行测试（使用数据集提供的gt_images）
python test_mrag_gt_images.py 2>&1 | tee test_mrag_gt_${SLURM_JOB_ID}.log

echo -e "\n测试完成时间: $(date)"