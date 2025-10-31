#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /root/autodl-tmp
python -u FlashRAG/experiments/run_all_baselines_100samples.py 2>&1 | tee run_100samples_$(date +%Y%m%d_%H%M%S).log
