#!/bin/bash

echo "=========================================="
echo "测试两个数据集的对比实验"
echo "基于 run_all_baselines_100samples.py 修改"
echo "=========================================="

# 先测试OK-VQA
echo ""
echo "测试 OK-VQA 数据集..."
echo "=========================================="

salloc -p cpu -n 1 --gres=gpu:1 --time=02:00:00 bash -c '
eval "$(conda shell.bash hook)"
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG

echo "运行 OK-VQA 测试..."
python experiments/run_two_datasets_10samples.py --dataset okvqa --max-samples 10
'