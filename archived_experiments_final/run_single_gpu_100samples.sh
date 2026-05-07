#!/bin/bash
# 单GPU 100样本消融实验测试脚本
# Single GPU 100 Samples Ablation Test

echo "🚀 启动单GPU 100样本消融实验..."
echo "验证准确率提升效果：从1.3%到50%+"

srun --partition=5090 \
     --gres=gpu:1 \
     --ntasks=1 \
     --cpus-per-task=8 \
     --mem=64G \
     --time=02:00:00 \
     --job-name=single_gpu_100samples \
     bash -c "source /data0/home/zqwang/ACL/activate_env.sh && python /data0/home/zqwang/ACL/test_real_model_only_100samples.py"

echo "✅ 单GPU 100样本消融实验完成"