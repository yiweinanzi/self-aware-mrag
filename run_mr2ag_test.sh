#!/bin/bash
cd /data0/home/zqwang/ACL/FlashRAG

# 运行mR²AG的10样本测试
source /data0/home/zqwang/miniconda3/bin/activate multirag && \
srun --gres=gpu:1 --ntasks=1 --cpus-per-task=8 --mem=32G \
python experiments/run_okvqa_baselines.py \
--max-samples 10 \
--methods mR²AG \
2>&1 | tee mr2ag_test_10samples.log

echo "测试完成，请查看 mr2ag_test_10samples.log"