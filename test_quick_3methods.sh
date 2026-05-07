#!/bin/bash

echo "🔧 Quick Test of 3 Fixed Methods"
echo "================================"

# Use srun with 2 GPUs
srun -p 5090 --gres=gpu:2 -t 00:20:00 bash << 'EOF'

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
mkdir -p results_quick_3methods

# Run baseline experiment with 3 fixed methods
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
    --output-dir results_quick_3methods \
    --save-detailed-results \
    --save-sample-results \
    --enable-complete-metrics \
    --methods MuRAG VisRAG ViDoRAG

EOF

echo ""
echo "✅ Quick test complete!"