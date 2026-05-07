#!/bin/bash
source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments
# 使用时间戳命名输出文件
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
python3 run_all_baselines_MRAG.py --max_samples 10 > slurm_mrag_fixed_${TIMESTAMP}.out 2>&1