#!/bin/bash
#SBATCH --gres=gpu:1 --ntasks=1 --cpus-per-task=8 --mem=32G -p 5090
#SBATCH --job-name=mrag_medium
#SBATCH --output=mrag_medium_%j.out
#SBATCH --error=mrag_medium_%j.err

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

# 创建运行日志
echo "============================================" | tee mrag_medium_${SLURM_JOB_ID}.out
echo "MRAG-Bench 中等规模实验" | tee -a mrag_medium_${SLURM_JOB_ID}.out
echo "============================================" | tee -a mrag_medium_${SLURM_JOB_ID}.out
echo "开始时间: $(date)" | tee -a mrag_medium_${SLURM_JOB_ID}.out
echo "样本数: 50" | tee -a mrag_medium_${SLURM_JOB_ID}.out

# 运行MRAG实验（50个样本）
echo -e "\n运行实验..." | tee -a mrag_medium_${SLURM_JOB_ID}.out
python3 run_all_baselines_MRAG.py --max_samples 50 2>&1 | tee -a mrag_medium_${SLURM_JOB_ID}.out

echo -e "\n实验完成时间: $(date)" | tee -a mrag_medium_${SLURM_JOB_ID}.out