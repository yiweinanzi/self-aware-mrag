#!/bin/bash
source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

# 创建改进版MRAG运行脚本
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="slurm_mrag_improved_${TIMESTAMP}.out"

echo "开始运行改进版MRAG实验..." | tee $OUTPUT_FILE
echo "时间戳: $TIMESTAMP" | tee -a $OUTPUT_FILE

# 运行改进版MRAG实验
python3 run_all_baselines_MRAG_improved.py --max_samples 10 2>&1 | tee -a $OUTPUT_FILE