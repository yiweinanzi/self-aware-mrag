#!/bin/bash

#SBATCH --job-name=okvqa_fixed_test
#SBATCH --partition=5090
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=fixed_test_%j.out
#SBATCH --error=fixed_test_%j.err

echo "🚀 运行修复验证测试"
echo "================================================"
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "================================================"

# 设置环境变量
export PYTHONPATH=/data0/home/zqwang/ACL:$PYTHONPATH
export PYTHONPATH=/data0/home/zqwang/ACL/FlashRAG:$PYTHONPATH
export HF_HOME=/data0/home/zqwang/ACL/models/huggingface
export TRANSFORMERS_CACHE=/data0/home/zqwang/ACL/models/huggingface/transformers

# 激活conda环境
source /data0/home/zqwang/miniconda3/bin/activate multirag

# 检查GPU
echo "检查GPU状态:"
nvidia-smi
echo ""

# 运行测试
cd /data0/home/zqwang/ACL/FlashRAG

# 只测试3个方法，5个样本
python experiments/run_okvqa_baselines.py \
    --dataset okvqa \
    --max-samples 5 \
    --model-path /data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct \
    --torch-dtype bfloat16 \
    --max-new-tokens 20 \
    --retrieval-topk 5 \
    --faiss-index-path /data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index \
    --corpus-path /data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl \
    --retrieval-model-path /data0/home/zqwang/ACL/models/bge-large-en-v1.5 \
    --use-multimodal-retrieval \
    --clip-model-path /data0/home/zqwang/ACL/models/clip-vit-large-patch14-336 \
    --clip-index-path /data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/clip/clip_Flat.index \
    --text-retrieval-weight 0.6 \
    --visual-retrieval-weight 0.4 \
    --uncertainty-threshold 0.43 \
    --text-weight 0.4 \
    --visual-weight 0.3 \
    --alignment-weight 0.3 \
    --use-improved-estimator \
    --output-dir results_fixed_test \
    --save-detailed-results \
    --save-sample-results \
    --enable-complete-metrics \
    --methods Self-Aware-MRAG MuRAG VisRAG

echo ""
echo "✅ 测试完成！"