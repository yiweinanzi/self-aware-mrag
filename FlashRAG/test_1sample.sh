#!/bin/bash

#SBATCH --job-name=test_1sample
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=test_1sample.log
#SBATCH --error=test_1sample.err

echo "========================================"
echo "运行原始脚本测试（1个样本）"
echo "========================================"

# 初始化conda
eval "$(conda shell.bash hook)"
conda activate multirag

# 设置环境
export CUDA_VISIBLE_DEVICES=0

echo "Python: $(python --version)"
echo "环境: $CONDA_DEFAULT_ENV"

# 运行脚本
cd /data0/home/zqwang/ACL/FlashRAG
python experiments/run_all_baselines_100samples.py