#!/bin/bash
#SBATCH -J okvqa_test
#SBATCH -p 5090
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:2
#SBATCH -t 02:00:00
#SBATCH -o logs/okvqa_test_%j.out
#SBATCH -e logs/okvqa_test_%j.err

# 创建日志目录
mkdir -p logs

# 加载环境
echo "Loading environment..."
eval "$(conda shell.bash hook)"
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0,1

# 运行OK-VQA测试
echo "Running OK-VQA real dataset test..."
echo "Start time: $(date)"
python experiments/run_all_baselines_OK_VQA.py
echo "End time: $(date)"

echo "OK-VQA test completed!"