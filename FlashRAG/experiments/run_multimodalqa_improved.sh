#!/bin/bash
#SBATCH --gres=gpu:2 --ntasks=1 --cpus-per-task=16 --mem=64G -p 5090
#SBATCH --job-name=multimodalqa_improved
#SBATCH --output=/data0/home/zqwang/ACL/multimodalqa_improved_%j.out
#SBATCH --error=/data0/home/zqwang/ACL/multimodalqa_improved_%j.err

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "============================================"
echo "MultiModalQA 改进版实验"
echo "开始时间: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "============================================"

echo -e "\n开始运行改进版实验（移除答案长度限制）..."
srun --gres=gpu:2 python run_all_baselines_MultimodalVQA.py \
    --max_samples 10 \
    --use_dataset_docs \
    --simple_table \
    --output_dir /data0/home/zqwang/ACL/results_multimodalqa_improved

echo -e "\n实验完成时间: $(date)"
echo "结果保存在: /data0/home/zqwang/ACL/results_multimodalqa_improved"
echo "============================================"