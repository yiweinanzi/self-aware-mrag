#!/bin/bash

echo "🚀 运行最终修复验证测试"
echo "================================================"
echo "验证所有修复："
echo "1. max_new_tokens修复 (10->20)"
echo "2. top_p参数警告修复"
echo "3. 改进��答案提取器"
echo "4. ViDoRAG检索率修复"
echo "5. Correct字段计算修复"
echo "================================================"

# 设置环境
export PYTHONPATH=/data0/home/zqwang/ACL:$PYTHONPATH
export PYTHONPATH=/data0/home/zqwang/ACL/FlashRAG:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME=/data0/home/zqwang/ACL/models/huggingface
export TRANSFORMERS_CACHE=/data0/home/zqwang/ACL/models/huggingface/transformers

# 激活环境
source /data0/home/zqwang/miniconda3/bin/activate multirag

# 创建输出目录
cd /data0/home/zqwang/ACL/FlashRAG
mkdir -p results_final_fixed_test

echo ""
echo "开始运行修复验证实验（5个样本）..."
echo ""

# 运行baseline实验
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
    --output-dir results_final_fixed_test \
    --save-detailed-results \
    --save-sample-results \
    --enable-complete-metrics \
    --methods Self-Aware-MRAG MuRAG VisRAG ViDoRAG

echo ""
echo "✅ 实验完成！"
echo ""
echo "结果保存在: results_final_fixed_test/"
echo "请检查准确率是否正确计算。"