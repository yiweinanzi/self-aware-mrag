#!/bin/bash
# 全样本OK-VQA消融实验脚本
# 基于54%准确率的成功配置

echo "🚀 启动全样本OK-VQA消融实验"
echo "基于54%准确率验证配置"
echo "========================================"

# 环境配置
export CUDA_VISIBLE_DEVICES=0,1
cd /data0/home/zqwang/ACL/FlashRAG/experiments

# 激活环境
source ~/.bashrc
source /data0/home/zqwang/miniconda3/bin/activate multirag

echo "📊 实验配置:"
echo "   数据集: OK-VQA (全样本 ~5046)"
echo "   GPU: 2x GPU"
echo "   基于配置: 54%准确率成功配置"
echo "   预计时间: ~13小时 (6个变体)"
echo ""

# 全样本实验配置
FULL_CONFIG="
--max-samples -1 \
--use-multi-gpu \
--num-gpus 2 \
--dataset okvqa \
--use-improved-estimator \
--text-retrieval-weight 0.6 \
--visual-retrieval-weight 0.4 \
--uncertainty-threshold 0.43 \
--text-weight 0.4 \
--visual-weight 0.3 \
--alignment-weight 0.3
"

echo "🔥 启动全样本消融实验..."
python run_unified_ablation.py $FULL_CONFIG

echo "✅ 全样本消融实验完成！"