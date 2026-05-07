#!/bin/bash
# -*- coding: utf-8 -*-
# 最终消融实验运行脚本

set -e

echo "================================================================="
echo "开始消融实验"
echo "================================================================="
echo "开始时间: $(date)"

# 激活环境
echo "🔄 激活multirag环境..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multirag

if [ $? -ne 0 ]; then
    echo "❌ 无法激活multirag环境"
    exit 1
fi

echo "✅ 环境激活成功: $(which python)"

# 检查GPU
echo "🔍 检查GPU状态:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits

# 进入工作目录
cd /data0/home/zqwang/ACL/FlashRAG/experiments

# 运行环境检查
echo "🧪 运行环境检查..."
if [ ! -f "setup_environment.sh" ]; then
    echo "❌ 环境设置脚本不存在"
    exit 1
fi

# 设置环境（如果还没有运行过）
echo "📝 检查环境状态..."
if python -c "import torch; print('PyTorch:', torch.__version__)" 2>/dev/null; then
    echo "✅ 环境已设置"
else
    echo "🔄 运行环境设置..."
    ./setup_environment.sh
fi

# 运行快速测试
echo "🧪 运行快速验证..."
python quick_test_run.py

if [ $? -ne 0 ]; then
    echo "❌ 快速测试失败，请检查问题"
    exit 1
fi

echo "✅ 快速测试通过"

# 运行消融实验
echo "🚀 开始运行消融实验..."
echo "数据集: OK-VQA val2014 (全部样本)"
echo "GPU配置: $(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l) 个GPU"
echo "预计时间: 15-25小时"
echo

# 创建日志目录
LOG_DIR="/data0/home/zqwang/ACL/FlashRAG/experiments/logs/ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

echo "📝 日志目录: $LOG_DIR"

# 启动消融实验
python run_ablation_study_okvqa.py 2>&1 | tee $LOG_DIR/ablation_full.log

# 检查结果
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo
    echo "🎉 消融实验完成!"
    echo "📊 查看结果: ls -la /data0/home/zqwang/ACL/FlashRAG/experiments/results_ablation_okvqa/"
    echo "📋 查看日志: ls -la $LOG_DIR/"
else
    echo
    echo "❌ 消融实验失败"
    echo "📋 查看错误日志: $LOG_DIR/ablation_full.log"
    tail -50 $LOG_DIR/ablation_full.log
    exit 1
fi

echo
echo "================================================================="
echo "实验完成!"
echo "================================================================="
echo "结束时间: $(date)"
echo "================================================================="