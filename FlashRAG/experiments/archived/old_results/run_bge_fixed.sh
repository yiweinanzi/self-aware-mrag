#!/bin/bash
source ~/.bashrc
source /data0/home/zqwang/miniconda3/bin/activate multirag
python run_unified_ablation.py --max-samples 100 --use-multi-gpu --num-gpus 2 --dataset okvqa --use-improved-estimator --text-retrieval-weight 0.6 --visual-retrieval-weight 0.4 --uncertainty-threshold 0.43 --text-weight 0.4 --visual-weight 0.3 --alignment-weight 0.3