#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "=== 最终真实消融实验（��复版）==="
echo "使用真实的Qwen3-VL模型 + 3M FAISS索引 + wiki语料库"
echo "数据集：全部OK-VQA样本(5046个)"
echo "时间: $(date)"
echo ""

# 运行修复后的真实消融实验
python run_real_model_ablation.py