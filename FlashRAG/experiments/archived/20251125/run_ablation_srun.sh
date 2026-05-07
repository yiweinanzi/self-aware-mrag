#!/bin/bash
# -*- coding: utf-8 -*-
# 使用srun申请GPU节点运行消融实验

set -e

# ============================================================================
# 配置参数
# ============================================================================

# SLURM配置
#SBATCH --job-name=ablation_okvqa
#SBATCH --partition=5090
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=/data0/home/zqwang/ACL/FlashRAG/experiments/logs/srun_%j.out
#SBATCH --error=/data0/home/zqwang/ACL/FlashRAG/experiments/logs/srun_%j.err

# 实验配置
CONDA_ENV="multirag"
WORKDIR="/data0/home/zqwang/ACL/FlashRAG/experiments"
LOG_DIR="/data0/home/zqwang/ACL/FlashRAG/experiments/logs"

# ============================================================================
# 主程序
# ============================================================================

echo "================================================================="
echo "消融实验 - SLURM版本"
echo "================================================================="
echo "开始时间: $(date)"
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURM_NODELIST"
echo "GPU数量: $SLURM_JOB_GPUS"
echo

# 创建日志目录
mkdir -p $LOG_DIR

# 激活conda环境
echo "🔄 激活conda环境: $CONDA_ENV"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

if [ $? -ne 0 ]; then
    echo "❌ 无法激活环境: $CONDA_ENV"
    exit 1
fi

echo "✅ 环境激活成功: $(which python)"

# 检查GPU
echo "🔍 检查GPU状态:"
nvidia-smi

# 进入工作目录
cd $WORKDIR

# 运行快速测试
echo "🧪 运行快速测试..."
python quick_test_run.py > $LOG_DIR/quick_test_$SLURM_JOB_ID.log 2>&1

if [ $? -ne 0 ]; then
    echo "❌ 快速测试失败，查看日志: $LOG_DIR/quick_test_$SLURM_JOB_ID.log"
    exit 1
fi

echo "✅ 快速测试通过"

# 运行消融实验
echo "🚀 开始运行消融实验..."
echo "使用GPU数量: $SLURM_JOB_GPUS"
echo "预计运行时间: 15-25小时"

# 设置CUDA设备
if [ -n "$SLURM_JOB_GPUS" ]; then
    export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS
fi

# 运行消融实验
python run_ablation_study_okvqa.py > $LOG_DIR/ablation_$SLURM_JOB_ID.log 2>&1

# 检查实验结果
if [ $? -eq 0 ]; then
    echo "✅ 消融实验完成!"
    echo "查看结果: ls -la /data0/home/zqwang/ACL/FlashRAG/experiments/results_ablation_okvqa/"
else
    echo "❌ 消融实验失败，查看日志: $LOG_DIR/ablation_$SLURM_JOB_ID.log"
    tail -50 $LOG_DIR/ablation_$SLURM_JOB_ID.log
    exit 1
fi

echo
echo "================================================================="
echo "实验完成!"
echo "================================================================="
echo "结束时间: $(date)"
echo "作业ID: $SLURM_JOB_ID"
echo "GPU使用: $SLURM_JOB_GPUS"
echo "================================================================="