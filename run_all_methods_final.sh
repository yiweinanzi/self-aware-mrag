#!/bin/bash

echo "🚀 Running All 7 Methods Baseline Test"
echo "==================================="

# Use srun with 2 GPUs
srun -p 5090 --gres=gpu:2 -t 02:00:00 bash << 'EOF'

# Set environment
export PYTHONPATH=/data0/home/zqwang/ACL:$PYTHONPATH
export PYTHONPATH=/data0/home/zqwang/ACL/FlashRAG:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME=/data0/home/zqwang/ACL/models/huggingface
export TRANSFORMERS_CACHE=/data0/home/zqwang/ACL/models/huggingface/transformers

# Activate environment
source /data0/home/zqwang/miniconda3/bin/activate multirag

# Create output directory
cd /data0/home/zqwang/ACL/FlashRAG
mkdir -p results_all_methods_final

# Run baseline experiment with all 7 methods
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
    --uncertainty-threshold 0.43 \
    --text-weight 0.4 \
    --visual-weight 0.3 \
    --alignment-weight 0.3 \
    --use-improved-estimator \
    --output-dir results_all_methods_final \
    --save-detailed-results \
    --save-sample-results \
    --enable-complete-metrics \
    --methods Self-Aware-MRAG MuRAG VisRAG ViDoRAG RagVL SAM-RAG mR²AG

EOF

echo ""
echo "✅ Complete experiment finished!"
echo "Results saved to: results_all_methods_final/"