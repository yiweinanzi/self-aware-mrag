#!/bin/bash
#SBATCH --gres=gpu:2 --ntasks=1 --cpus-per-task=16 --mem=64G -p 5090
#SBATCH --job-name=test_selfaware_mr_fixed
#SBATCH --output=test_selfaware_mr_fixed_%j.out
#SBATCH --error=test_selfaware_mr_fixed_%j.err

source /data0/home/zqwang/miniconda3/etc/profile.d/conda.sh
conda activate multirag
cd /data0/home/zqwang/ACL/FlashRAG/experiments

echo "============================================"
echo "测试修复后的Self-Aware-MRAG - 10样本"
echo "开始时间: $(date)"
echo "============================================"

mkdir -p results_selfaware_mr_fixed

# 只测试Self-Aware-MRAG
python -c "
import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from flashrag.pipeline.self_aware_pipeline_qwen3vl import SelfAwarePipelineQwen3VL
from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
from flashrag.retriever import DenseRetriever
import json
import gzip
from datetime import datetime

# 初始化模型
print('正在加载Qwen3-VL...')
qwen3_vl = create_qwen3_vl_wrapper(
    model_path='/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
    device='cuda',
    torch_dtype='bfloat16'
)

# 初始化检索器（使用BGE检索器，不使用多模态以避免问题）
print('正在初始化检索器...')
config = {
    'index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa/bge_Flat.index',
    'corpus_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/multimodalqa/corpus.jsonl',
    'retrieval_method': 'e5',
    'retrieval_model_path': '/data0/home/zqwang/ACL/models/bge-large-en-v1.5',
    'retrieval_query_max_length': 512,
    'retrieval_pooling_method': 'mean',
    'retrieval_use_fp16': True,
    'retrieval_batch_size': 128,
    'retrieval_topk': 5,
    'save_retrieval_cache': False,
    'use_retrieval_cache': False,
    'retrieval_cache_path': None,
    'faiss_gpu': False,
    'use_reranker': False,
    'instruction': '',
    'use_sentence_transformer': False,
}

retriever = DenseRetriever(config)

# 初始化Self-Aware-MRAG
print('正在初始化Self-Aware-MRAG...')
pipeline = SelfAwarePipelineQwen3VL(
    qwen3_vl_wrapper=qwen3_vl,
    retriever=retriever,
    config={
        'uncertainty_threshold': 0.35,
        'use_improved_estimator': True,
        'use_position_fusion': True,
        'use_attribution': True,
        'use_dataset_docs': True
    }
)

# 测试第一个样本
print('\\n开始测试第一个样本...')
with gzip.open('/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MultiModalQA/MMQA_dev.jsonl.gz', 'rt') as f:
    item = json.loads(f.readline())

print(f'问题: {item[\"question\"]}')
print(f'答案: {item[\"answer\"]}')

try:
    result = pipeline.process_sample(item)
    print(f'✅ Self-Aware-MRAG运行成功!')
    print(f'预测答案: {result[\"answer\"]}')
    print(f'是否正确: {result[\"answer\"].lower() == item[\"answer\"].lower()}')
except Exception as e:
    print(f'❌ 错误: {e}')
    import traceback
    traceback.print_exc()
"