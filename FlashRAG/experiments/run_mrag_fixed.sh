#!/bin/bash
#SBATCH --gres=gpu:2 --ntasks=1 --cpus-per-task=16 --mem=64G -p 5090
#SBATCH --job-name=mrag_fixed
#SBATCH --output=mrag_fixed_%j.out
#SBATCH --error=mrag_fixed_%j.err

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

# 设置时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "开始运行修复版MRAG实验..." | tee mrag_fixed_${SLURM_JOB_ID}.out
echo "时间戳: $TIMESTAMP" | tee -a mrag_fixed_${SLURM_JOB_ID}.out

# 运行修复版MRAG实验
python3 run_all_baselines_MRAG.py --max_samples 10 2>&1 | tee -a mrag_fixed_${SLURM_JOB_ID}.out