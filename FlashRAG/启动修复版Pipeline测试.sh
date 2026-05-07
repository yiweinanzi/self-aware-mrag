#!/bin/bash
# 启动修复版Pipeline测试
# 配置：500样本 + 300万Wikipedia

set -e

echo "=========================================="
echo "🔬 启动修复版Pipeline测试"
echo "=========================================="
echo "配置："
echo "  - 样本数: 500"
echo "  - Wikipedia: 300万条"
echo "  - 使用: 今晚修复的self_aware_pipeline_fixed.py"
echo ""
echo "新增功能："
echo "  1. ✅ Query Reformulation"
echo "  2. ✅ Attribution增强使用"
echo "  3. ✅ Modality Selection"
echo ""
echo "对比: 昨天68.90% vs 今晚预期70-72%"
echo ""

# 激活环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate multirag

# 进入目录
cd /root/autodl-tmp/FlashRAG

# 创建日志目录
mkdir -p logs
mkdir -p experiments/fixed_pipeline_test

# 运行实验（后台）
nohup python 实验-修复版Pipeline测试.py \
  --max_samples 500 \
  --max_wiki 3000000 \
  --topk 5 \
  --uncertainty_threshold 0.5 \
  --output_dir experiments/fixed_pipeline_test \
  > logs/fixed_pipeline_test_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 获取PID
PID=$!
echo "实验已启动（后台）"
echo "PID: $PID"
echo "日志: logs/fixed_pipeline_test_*.log"
echo ""
echo "预计时间: ~2小时"
echo "  - BGE编码: ~45分钟（300万条）"
echo "  - 实验运行: ~1小时（500样本）"
echo ""
echo "查看进度："
echo "  tail -f logs/fixed_pipeline_test_*.log"
echo ""

# 保存PID
echo $PID > fixed_pipeline_test.pid
echo "PID已保存到: fixed_pipeline_test.pid"
echo ""
echo "对比昨天:"
echo "  昨天: 68.90% (+21.50%)"
echo "  预期: 70-72% (+23-25%)"
echo "  新增贡献: +1-3%"


