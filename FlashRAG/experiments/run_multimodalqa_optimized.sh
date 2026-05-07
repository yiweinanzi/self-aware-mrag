#!/bin/bash
#SBATCH --gres=gpu:2 --ntasks=1 --cpus-per-task=16 --mem=64G -p 5090
#SBATCH --job-name=multimodalqa_optimized
#SBATCH --output=/data0/home/zqwang/ACL/multimodalqa_optimized_%j.out
#SBATCH --error=/data0/home/zqwang/ACL/multimodalqa_optimized_%j.err

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "============================================"
echo "MultiModalQA 优化版实验（增强多模态检索）"
echo "开始时间: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "============================================"

echo -e "\n开始运行优化版实验..."
echo "  - 降低不确定性阈值 (0.35 -> 0.25)"
echo "  - 增强视觉检索权重 (40% -> 50%)"
echo "  - 增加检索文档数 (5 -> 10)"

# 修改运行脚本来支持参数
srun --gres=gpu:2 python run_all_baselines_MultimodalVQA_optimized.py \
    --max_samples 10 \
    --use_dataset_docs \
    --simple_table \
    --uncertainty_threshold 0.25 \
    --visual_weight 0.5 \
    --retrieval_topk 10 \
    --output_dir /data0/home/zqwang/ACL/results_multimodalqa_optimized

echo -e "\n实验完成时间: $(date)"
echo "结果保存在: /data0/home/zqwang/ACL/results_multimodalqa_optimized"
echo "============================================"