#!/bin/bash
echo "=========================================="
echo "OK-VQA Baselines 全样本对比实验"
echo "=========================================="
echo "实验开始时间: $(date)"
echo ""

cd /data0/home/zqwang/ACL/FlashRAG

# 激活环境并运行实验
source /data0/home/zqwang/miniconda3/bin/activate multirag && \
srun --gres=gpu:1 --ntasks=1 --cpus-per-task=8 --mem=32G \
python experiments/run_okvqa_baselines.py \
--max-samples 5046 \
--save-detailed-results \
--save-sample-results \
--enable-complete-metrics \
2>&1 | tee ../full_baselines_experiment_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=========================================="
echo "实验完成时间: $(date)"
echo "=========================================="