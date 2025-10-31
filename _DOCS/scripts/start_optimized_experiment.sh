#!/bin/bash

# 优化版本实验启动脚本
# 修复: Context格式简化 + 参数优化 + 合理阈值

cd /root/autodl-tmp/FlashRAG/experiments

# 激活conda环境并运行
source /root/miniconda3/bin/activate multirag
python -u run_all_baselines_100samples.py 2>&1 | tee /root/autodl-tmp/optimized_100samples.log

