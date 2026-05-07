#!/bin/bash
#SBATCH --job-name=full_okvqa_ablation_remaining5
#SBATCH --partition=5090
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --time=7-00:00:00  # 7天无时间限制
#SBATCH --output=full_ablation_%j.out
#SBATCH --error=full_ablation_%j.err
#SBATCH --mem=64G

echo "🚀 SLURM全样本OK-VQA消融实验"
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURM_NODELIST"
echo "开始时间: $(date)"
echo "========================================"

# 激活环境
source ~/.bashrc
source /data0/home/zqwang/miniconda3/bin/activate multirag

# 切换目录
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "📊 实验配置:"
echo "   数据集: OK-VQA (全样本 ~5046)"
echo "   GPU: 2x GPU (节点: $SLURM_NODELIST)"
echo "   基于配置: 54%准确率成功配置"
echo "   时间限制: 7天 (无限制)"
echo ""

# 全样本实验配置 (完全对标54%成功实验 + 完整不确定性 + 新增评估指标)
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
echo "配置: $FULL_CONFIG"
echo ""

# 跳过已完成的Baseline_MuRAG，运行剩余5个变体
python run_unified_ablation.py $FULL_CONFIG --variants Text_Uncertainty_Improved Visual_Uncertainty Cross_Modal_Alignment Position_Aware_Fusion Full_Self_Aware_RAG

echo "✅ 全样本消融实验完成！"
echo "结束时间: $(date)"