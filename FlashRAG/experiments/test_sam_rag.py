#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试SAM-RAG实现 - 快速验证
"""

import sys
sys.path.insert(0, '/root/autodl-tmp/FlashRAG')

import datasets
from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
from flashrag.retriever import DenseRetriever

# 导入SAM-RAG Pipeline
import os
import importlib.util

# 动态导入run_all_baselines_100samples.py中的SAMRAGPipeline
spec = importlib.util.spec_from_file_location(
    "baselines",
    "/root/autodl-tmp/FlashRAG/experiments/run_all_baselines_100samples.py"
)
baselines = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baselines)

SAMRAGPipeline = baselines.SAMRAGPipeline


def test_sam_rag():
    """测试SAM-RAG"""
    print("="*80)
    print("测试SAM-RAG实现")
    print("="*80)
    
    # 1. 加载数据（只测试1个样本）
    print("\n1. 加载测试数据...")
    dataset_path = '/root/autodl-tmp/FlashRAG/flashrag/data/MRAG-Bench/raw'
    dataset_dict = datasets.load_from_disk(dataset_path)
    test_data = dataset_dict['test'].select(range(1))
    
    sample = {
        'question': test_data[0]['question'],
        'image': test_data[0]['image'],
        'answer': test_data[0]['answer'],
        'A': test_data[0]['A'],
        'B': test_data[0]['B'],
        'C': test_data[0]['C'],
        'D': test_data[0]['D'],
    }
    
    print(f"问题: {sample['question']}")
    print(f"正确答案: {sample['answer']}")
    
    # 2. 初始化模型
    print("\n2. 初始化Qwen3-VL...")
    qwen3_vl_path = '/root/autodl-tmp/models/Qwen3-VL-8B-Instruct'
    qwen3_vl = create_qwen3_vl_wrapper(qwen3_vl_path)
    print("✅ Qwen3-VL加载成功")
    
    # 3. 初始化检索器
    print("\n3. 初始化检索器...")
    retriever_config = {
        'index_path': '/root/autodl-tmp/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
        'corpus_path': '/root/autodl-tmp/FlashRAG/corpus/corpus_wiki_3m.jsonl',
        'retrieval_method': 'e5',
        'retrieval_model_path': '/root/autodl-tmp/models/bge-large-en-v1.5',
        'retrieval_query_max_length': 512,
        'retrieval_use_fp16': True,
        'retrieval_batch_size': 128,
        'retrieval_topk': 20,
        'save_retrieval_cache': False,
        'use_retrieval_cache': False,
    }
    
    retriever = DenseRetriever(retriever_config)
    print("✅ 检索器加载成功")
    
    # 4. 初始化SAM-RAG Pipeline
    print("\n4. 初始化SAM-RAG Pipeline...")
    config = {
        'sam_batch_size': 5,  # 每批5个文档
        'sam_max_batches': 4,  # 最多4批
        'retrieval_topk': 20,
        'temperature': 0.01,
        'max_new_tokens': 50,
    }
    
    pipeline = SAMRAGPipeline(qwen3_vl, retriever, config)
    print("✅ SAM-RAG Pipeline初始化成功")
    
    # 5. 运行测试
    print("\n5. 运行SAM-RAG...")
    print("-"*80)
    result = pipeline.run_single(sample)
    print("-"*80)
    
    # 6. 显示结果
    print("\n6. 结果:")
    print(f"预测答案: {result['answer']}")
    print(f"正确答案: {sample['answer']}")
    print(f"是否正确: {'✅' if result['answer'] == sample['answer'] else '❌'}")
    print(f"\n检索统计:")
    print(f"  - 使用检索: {result.get('used_retrieval', False)}")
    print(f"  - 检索批次: {result.get('num_batches_used', 0)}")
    print(f"  - 相关文档数: {result.get('relevant_docs_count', 0)}")
    print(f"  - 支持状态: {result.get('support_status', 'N/A')}")
    print(f"  - 有用性状态: {result.get('usefulness_status', 'N/A')}")
    
    if result.get('retrieved_docs'):
        print(f"\n检索到的文档 (前3个):")
        for i, doc in enumerate(result['retrieved_docs'][:3], 1):
            doc_preview = doc[:100] + "..." if len(doc) > 100 else doc
            print(f"  {i}. {doc_preview}")
    
    print("\n" + "="*80)
    print("✅ SAM-RAG测试完成！")
    print("="*80)


if __name__ == '__main__':
    test_sam_rag()

