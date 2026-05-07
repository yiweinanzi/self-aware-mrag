#!/bin/bash
#SBATCH --gres=gpu:2 --ntasks=1 --cpus-per-task=16 --mem=64G -p 5090
#SBATCH --job-name=mrag_optimized
#SBATCH --output=mrag_optimized_%j.out
#SBATCH --error=mrag_optimized_%j.err

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

# 运行优化版MRAG实验
cd /data0/home/zqwang/ACL/FlashRAG/experiments
python3 run_all_baselines_MRAG_optimized.py --max_samples 10 2>&1 | tee mrag_optimized_${SLURM_JOB_ID}.out