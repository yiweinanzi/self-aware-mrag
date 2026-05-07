#!/bin/bash

#SBATCH --job-name=test_single_sample
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=test_output.log
#SBATCH --error=test_error.log

echo "========================================"
echo "在salloc节点运行单样本测试"
echo "========================================"
echo "节点: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "开始时间: $(date)"
echo "========================================"

# 初始化conda
eval "$(conda shell.bash hook)"
conda activate multirag

echo "环境信息:"
echo "  Python: $(python --version)"
echo "  Conda环境: $CONDA_DEFAULT_ENV"
echo "  工作目录: $(pwd)"

# 检查torch
python -c "import torch; print('PyTorch:', torch.__version__)" 2>&1 || echo "PyTorch未安装"

# 检查CUDA
nvidia-smi

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

echo ""
echo "========================================"
echo "运行单样本测试"
echo "========================================"

# 运行Python测试
python test_single_sample.py

echo ""
echo "========================================"
echo "测试完成"
echo "========================================"
echo "结束时间: $(date)"