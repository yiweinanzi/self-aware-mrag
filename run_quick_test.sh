#!/bin/bash

# 简单测试脚本 - 测试修复后的baseline方法
# 只测试2个样本，快速验证修复效果

echo "🚀 测试修复后的OK-VQA Baseline方法"
echo "================================================"

# 设置环境变量
export PYTHONPATH=/data0/home/zqwang/ACL:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1

# 运行baseline实验，只测试2个样本
python test_baselines_debug.py \
    --dataset okvqa \
    --num_samples 2 \
    --baseline_methods all

echo ""
echo "✅ 测试完成！"