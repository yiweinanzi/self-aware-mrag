#!/bin/bash
#SBATCH --gres=gpu:2 --ntasks=1 --cpus-per-task=16 --mem=64G -p 5090
#SBATCH --job-name=mrag_larger
#SBATCH --output=mrag_larger_%j.out
#SBATCH --error=mrag_larger_%j.err

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "开始运行MRAG更大样本实验..." | tee mrag_larger_${SLURM_JOB_ID}.out
echo "时间戳: $(date)" | tee -a mrag_larger_${SLURM_JOB_ID}.out
echo "样本数: 100" | tee -a mrag_larger_${SLURM_JOB_ID}.out

# 运行MRAG实验（100个样本）
python3 run_all_baselines_MRAG.py --max_samples 100 2>&1 | tee -a mrag_larger_${SLURM_JOB_ID}.out