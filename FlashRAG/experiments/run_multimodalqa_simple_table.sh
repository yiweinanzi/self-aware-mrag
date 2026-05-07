#!/bin/bash
#SBATCH --gres=gpu:2 --ntasks=1 --cpus-per-task=16 --mem=64G -p 5090
#SBATCH --job-name=multimodalqa_simple_table
#SBATCH --output=/data0/home/zqwang/ACL/multimodalqa_simple_table_%j.out
#SBATCH --error=/data0/home/zqwang/ACL/multimodalqa_simple_table_%j.err

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "============================================"
echo "MultiModalQA 简化表格处理实验（基于MOQAGPT）"
echo "开始时间: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "============================================"

echo -e "\n开始运行简化表格处理实验..."
srun --gres=gpu:2 python run_all_baselines_MultimodalVQA.py \
    --max_samples 10 \
    --use_dataset_docs \
    --simple_table \
    --output_dir /data0/home/zqwang/ACL/results_multimodalqa_simple_table

echo -e "\n实验完成时间: $(date)"
echo "结果保存在: /data0/home/zqwang/ACL/results_multimodalqa_simple_table"
echo "============================================"