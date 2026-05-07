#!/bin/bash

echo "🚀 运行快速测试 - 验证max_new_tokens修复效果"
echo "================================================"

# 设置环境变量
export PYTHONPATH=/data0/home/zqwang/ACL:$PYTHONPATH
export PYTHONPATH=/data0/home/zqwang/ACL/FlashRAG:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME=/data0/home/zqwang/ACL/models/huggingface
export TRANSFORMERS_CACHE=/data0/home/zqwang/ACL/models/huggingface/transformers

# 激活conda环境
source /data0/home/zqwang/miniconda3/bin/activate multirag

# 检查GPU状态
echo "检查GPU状态:"
nvidia-smi

# 运行快速测试 - 只测试3个方法，5个样本
echo ""
echo "开始运行快速测试（5个样本）..."
cd /data0/home/zqwang/ACL/FlashRAG

# 创建输出目录
mkdir -p results_quick_test

# 直接运行，使用Python脚本
python -c "
import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 导入必要的模块
from experiments.run_okvqa_baselines import main

# 设置命令行参数
sys.argv = [
    'run_okvqa_baselines.py',
    '--dataset', 'okvqa',
    '--max-samples', '5',
    '--model-path', '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
    '--torch-dtype', 'bfloat16',
    '--max-new-tokens', '20',
    '--retrieval-topk', '5',
    '--faiss-index-path', '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
    '--corpus-path', '/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl',
    '--retrieval-model-path', '/data0/home/zqwang/ACL/models/bge-large-en-v1.5',
    '--use-multimodal-retrieval',
    '--clip-model-path', '/data0/home/zqwang/ACL/models/clip-vit-large-patch14-336',
    '--clip-index-path', '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/clip/clip_Flat.index',
    '--text-retrieval-weight', '0.6',
    '--visual-retrieval-weight', '0.4',
    '--uncertainty-threshold', '0.43',
    '--text-weight', '0.4',
    '--visual-weight', '0.3',
    '--alignment-weight', '0.3',
    '--use-improved-estimator',
    '--output-dir', 'results_quick_test',
    '--save-detailed-results',
    '--save-sample-results',
    '--enable-complete-metrics',
    '--methods', 'Self-Aware-MRAG', 'MuRAG', 'VisRAG'
]

# 运行主程序
main()
"

echo ""
echo "✅ 快速测试完成！"