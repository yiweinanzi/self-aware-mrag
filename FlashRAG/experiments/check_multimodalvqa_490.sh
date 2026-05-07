#!/bin/bash
# Monitor MultiModalQA experiment job 491

echo "============================================"
echo "Monitoring MultiModalQA Experiment Job 491"
echo "============================================"
echo "Current time: $(date)"
echo

# Check job status
echo "Job Status:"
scontrol show job 491 | grep -E "JobState|RunTime|TimeLimit|NodeList"
echo

# Check output file
echo "Recent output (last 50 lines):"
if [ -f "/data0/home/zqwang/ACL/FlashRAG/experiments/multimodalvqa_fixed_491.out" ]; then
    tail -50 /data0/home/zqwang/ACL/FlashRAG/experiments/multimodalvqa_fixed_491.out
else
    echo "Output file not found yet"
fi

echo
echo "============================================"