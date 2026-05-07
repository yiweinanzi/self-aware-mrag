#!/bin/bash
# 完整样本消融实验脚本
# Full Dataset Ablation Study (5046 samples)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "=== 完整样本消融实验 ==="
echo "🎯 目标：使用全部5046个OK-VQA样本进行最终消融实验"
echo "📊 6个消融变体：Baseline, +Text Uncertainty, +Visual Uncertainty, +Position Fusion, +Full Self-Aware"
echo "⏱️ 预计时间：约8-10小时"
echo "🚀 开始时间: $(date)"
echo ""

# 使用修复后的配置运行完整实验
python run_real_model_ablation.py

echo "✅ 完整样本消融实验完成"
echo "📅 结束时间: $(date)"