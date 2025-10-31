#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /root/autodl-tmp/FlashRAG/experiments

# 只运行Self-Aware-MRAG，100样本
python -u run_all_baselines_100samples.py 2>&1 | tee /root/autodl-tmp/test_selfaware_fixed_full.log
