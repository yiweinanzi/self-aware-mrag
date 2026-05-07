#!/bin/bash
#SBATCH --job-name=multi_datasets_baselines
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/multi_datasets_%j.out
#SBATCH --error=logs/multi_datasets_%j.err

# 创建日志目录
mkdir -p logs

# 打印作业信息
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo "=========================================="

# 激活conda环境
source ~/.bashrc
conda activate multirag

# 检查环境
echo "Python: $(python --version)"
echo "CUDA: $(nvcc --version | head -n 1)"
nvidia-smi

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 运行实验
echo "=========================================="
echo "Running multi-dataset baseline comparison..."
echo "=========================================="

# 首先运行单个数据集测试
echo "Testing with OK-VQA dataset..."
python experiments/run_all_baselines_100samples.py

# 如果成功，运行多数据集实验
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Running on 4 datasets..."
    echo "=========================================="

    # 运行所有4个数据集（使用较少样本以节省时间）
    python experiments/run_all_baselines_multi_datasets.py \
        --datasets okvqa mrag-bench \
        --methods Self-Aware-MRAG ViDoRAG \
        --max-samples 50 \
        --gpu-id 0
else
    echo "Single dataset test failed, skipping multi-dataset run"
fi

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="