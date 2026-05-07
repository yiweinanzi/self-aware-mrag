#!/bin/bash

# 在salloc节点上运行的脚本

echo "=========================================="
echo "设置运行环境"
echo "=========================================="

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/data0/home/zqwang/ACL/FlashRAG:$PYTHONPATH"

# 检查GPU状态
nvidia-smi

# 运行单���数据集测试
echo "=========================================="
echo "测试运行：OK-VQA数据集"
echo "=========================================="

# 先激活环境（如果需要）
# source ~/.bashrc
# conda activate multirag

# 运行原始脚本（单数据集）
python experiments/run_all_baselines_100samples.py 2>&1 | tee logs/okvqa_test.log

# 检查是否成功
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "OK-VQA测试成功！"
else
    echo "OK-VQA测试失败，检查日志 logs/okvqa_test.log"
    exit 1
fi

echo "=========================================="
echo "多数据集对比实验"
echo "=========================================="

# 运行多数据集对比（使用2个数据集测试）
python experiments/run_all_baselines_multi_datasets.py \
    --datasets okvqa mrag-bench \
    --methods Self-Aware-MRAG ViDoRAG \
    --max-samples 10 \
    --gpu-id 0 2>&1 | tee logs/multi_datasets_test.log

echo "=========================================="
echo "实验完成！"
echo "=========================================="
echo "结果保存在："
echo "  - logs/okvqa_test.log"
echo "  - logs/multi_datasets_test.log"
echo "  - experiments/results_*/"