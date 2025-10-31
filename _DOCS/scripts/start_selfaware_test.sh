#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /root/autodl-tmp
python -u run_selfaware_only_100samples.py 2>&1 | tee test_selfaware_fixed.log
