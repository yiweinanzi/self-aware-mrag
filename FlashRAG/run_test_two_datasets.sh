#!/bin/bash

#SBATCH --job-name=test_two_datasets
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=test_two_datasets_%j.out
#SBATCH --error=test_two_datasets_%j.err

echo "=========================================="
echo "测试两个数据集的完整对比实验"
echo "OK-VQA 和 MRAG-Bench"
echo "7个方法，10个样本"
echo "=========================================="
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURM_NODELIST"
echo "开始时间: $(date)"

# 初始化conda
eval "$(conda shell.bash hook)"
conda activate multirag

echo ""
echo "环境信息:"
echo "  Python: $(python --version)"
echo "  环境: $CONDA_DEFAULT_ENV"
echo "  GPU信息:"
nvidia-smi

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 运行测试
echo ""
echo "=========================================="
echo "运行测试脚本"
echo "=========================================="

cd /data0/home/zqwang/ACL/FlashRAG
python test_two_datasets_10samples.py 2>&1 | tee test_two_datasets.log

# 检查是否成功
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "测试成功完成！"
    echo "=========================================="
    echo "结果保存在: experiments/test_two_datasets_results/"
    echo "查看报告: ls -la experiments/test_two_datasets_results/*.md"
else
    echo ""
    echo "=========================================="
    echo "测试失败！"
    echo "=========================================="
    echo "查看日志: test_two_datasets.log"
fi

echo ""
echo "结束时间: $(date)"