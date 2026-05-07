#!/bin/bash
#SBATCH --gres=gpu:2 --ntasks=1 --cpus-per-task=16 --mem=64G -p 5090
#SBATCH --job-name=mrag_improved
#SBATCH --output=mrag_improved_%j.out
#SBATCH --error=mrag_improved_%j.err

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments
python3 run_all_baselines_MRAG.py --max_samples 10