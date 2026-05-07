#!/bin/bash

echo "========================================"
echo "最简测试：修改原始脚本使用1个样本"
echo "========================================"

# 先备份原始脚本
cp experiments/run_all_baselines_100samples.py experiments/run_all_baselines_100samples.py.bak

# 修改样本数为1
sed -i "s/'max_samples': 100/'max_samples': 1/" experiments/run_all_baselines_100samples.py

echo "已修改脚本，使用1个样本"

# 提交作业
sbatch --wrap="
eval \\"\\\$(conda shell.bash hook)\\"
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG
echo '开始运行原始脚本（1个样本）'
python experiments/run_all_baselines_100samples.py
echo '运行完成'
" --job-name=simple_test --partition=cpu --gres=gpu:1 --time=03:00:00 --output=simple_test.log --error=simple_test.err

echo "作业已提交，查看输出：tail -f simple_test.log"