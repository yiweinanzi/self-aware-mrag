#!/bin/bash

echo "========================================"
echo "最终测试：使用salloc直接运行"
echo "========================================"

# 申请salloc节点并运行
salloc -p cpu -n 1 --gres=gpu:1 --time=02:00:00 bash -c '
echo "=== 在salloc节点中 ==="
eval "$(conda shell.bash hook)"
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG
echo "环境信息:"
which python
python --version

echo ""
echo "========================================"
echo "运行原始脚本测试（1个样本）"
echo "========================================"

# 运行脚本
python experiments/run_all_baselines_100samples.py 2>&1 | tee final_test_output.log

echo ""
echo "测试完成"
'