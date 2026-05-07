#!/bin/bash

# 运行OK-VQA Baseline实验 - 10个样本
echo "🚀 运行OK-VQA Baseline实验 (10个样本)"
echo "================================================"

# 设置环境变量
export PYTHONPATH=/data0/home/zqwang/ACL:$PYTHONPATH
export PYTHONPATH=/root/autodl-tmp/FlashRAG:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/data0/home/zqwang/ACL/models/huggingface
export TRANSFORMERS_CACHE=/data0/home/zqwang/ACL/models/huggingface/transformers

# 激活conda环境
source /data0/home/zqwang/miniconda3/bin/activate multirag

# 检查GPU可用性
echo "检查GPU状态:"
nvidia-smi

# 运行baseline实验
echo ""
echo "开始运行baseline实验..."
cd /data0/home/zqwang/ACL/FlashRAG

# 创建输出目录
mkdir -p results_okvqa_baselines_10samples

python experiments/run_okvqa_baselines.py \
    --dataset okvqa \
    --max-samples 10 \
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
    --use-multi-gpu \
    --num-gpus 2 \
    --uncertainty-threshold 0.43 \
    --text-weight 0.4 \
    --visual-weight 0.3 \
    --alignment-weight 0.3 \
    --use-improved-estimator \
    --output-dir results_okvqa_baselines_10samples \
    --save-detailed-results \
    --save-sample-results \
    --enable-complete-metrics \
    --methods Self-Aware-MRAG MuRAG VisRAG ViDoRAG RagVL SAM-RAG mR²AG

echo ""
echo "✅ 实验完成！"