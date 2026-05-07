#!/bin/bash
source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments
python3 run_all_baselines_MRAG.py --max_samples 10