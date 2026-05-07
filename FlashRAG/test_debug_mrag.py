#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
调试MRAG-Bench问题
"""

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
from flashrag.retriever import DenseRetriever
from flashrag.pipeline.self_aware_pipeline_qwen3vl import SelfAwarePipelineQwen3VL

# 1. 加载一个样本
from datasets import load_from_disk
dataset = load_from_disk('/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MRAG-Bench/raw')
test_data = dataset['test']
sample = test_data[0]

print("样本结构：")
print(f"- 问题: {sample['question']}")
print(f"- 选项: {sample['choices']}")
print(f"- 答案: {sample['answer']}")
print(f"- 图像路径: {sample.get('image', 'None')}")

# 2. 初始化模型
print("\n初始化模型...")
qwen3_vl = create_qwen3_vl_wrapper(
    model_path='/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
    device="cuda",
    torch_dtype="bfloat16"
)
print("✅ 模型加载成功")

# 3. 初始化检索器
print("\n初始化检索器...")
retriever_config = {
    'index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
    'corpus_path': '/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl',
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
    'use_reranker': False,
    'use_sentence_transformer': False,
    'faiss_gpu': False,
    'instruction': '',
}

retriever = DenseRetriever(retriever_config)
print("✅ 检索器加载成功")

# 4. 创建pipeline
print("\n创建pipeline...")
pipeline = SelfAwarePipelineQwen3VL(
    qwen3_vl_wrapper=qwen3_vl,
    retriever=retriever,
    config={
        'uncertainty_threshold': 0.43,
        'max_images': 20,
        'use_improved_estimator': True,
    }
)
print("✅ Pipeline创建成功")

# 5. 运行一个样本
print("\n运行测试...")
try:
    result = pipeline.run_single(sample)
    print(f"生成答案: {result.get('pred_answer', 'N/A')}")
    print(f"正确答案: {sample['answer']}")
    print(f"是否正确: {result.get('pred_answer', '') == sample['answer']}")
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()