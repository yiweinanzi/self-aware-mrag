#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断baseline方法的准确率和检索问题"""

import sys
import os
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

import json
import warnings
from typing import Dict, Any

# Suppress warnings
warnings.filterwarnings("ignore")

from flashrag.pipeline自我感知管道_qwen3vl import SelfAwarePipelineQwen3VL
from flashrag.retriever import initialize_retriever
from baselines.murag_enhanced import MuRAGEnhanced
from baselines.vidorag_pipeline import ViDoRAGPipeline
from utils.dataset_loader import UnifiedDatasetLoader
from models.model_loaders import Qwen3VLWrapper

def diagnose_baseline_methods():
    """诊断各个baseline方法的问题"""

    print("="*70)
    print("Baseline方法诊断")
    print("="*70)

    # 初始化
    config = {
        'retrieval_model_path': '/data0/home/zqwang/ACL/models/bge-m3',
        'faiss_index_path': '/data0/home/zqwang/ACL/FlashRAG/index/okvqa/bge_index.faiss',
        'corpus_path': '/data0/home/zqwang/ACL/FlashRAG/index/okvqa/corpus.json',
        'retrieval_topk': 5,
        'clip_model_path': '/data0/home/zqwang/ACL/models/clip-vit-large-patch14-336'
    }

    # 加载数据
    loader = UnifiedDatasetLoader()
    dataset = loader.load_dataset(
        dataset_name='okvqa',
        split='train',
        max_samples=3,  # 只用3个样本快速诊断
        image_dir=None
    )

    print(f"\n加载了{len(dataset)}个样本")

    # 初始化模型
    print("\n初始化模型...")
    qwen3vl_wrapper = Qwen3VLWrapper(
        model_path='/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
        device_map='auto'
    )

    retriever = initialize_retriever(
        retrieval_method="bm25",
        model_name=config['retrieval_model_path'],
        faiss_path=config['faiss_index_path'],
        corpus_path=config['corpus_path']
    )

    # 测试每个方法
    methods = {
        'MuRAG': MuRAGEnhanced(qwen3vl_wrapper, retriever, config),
        'ViDoRAG': ViDoRAGPipeline(qwen3vl_wrapper, retriever, config)
    }

    for method_name, pipeline in methods.items():
        print(f"\n{'='*50}")
        print(f"测试方法: {method_name}")
        print(f"{'='*50}")

        for i, sample in enumerate(dataset[:2]):  # 只测试前2个样本
            print(f"\n--- 样本 {i+1} ---")
            print(f"问题: {sample['question'][:50]}...")
            print(f"标准答案: {sample['golden_answers']}")

            try:
                # 运行方法
                result = pipeline.run_single(sample)

                # 检查结果
                answer = result.get('answer', '')
                retrieved_docs = result.get('retrieved_docs', [])

                print(f"生成答案: '{answer}'")
                print(f"检索到的文档数: {len(retrieved_docs)}")

                # 检查文档格式
                if retrieved_docs:
                    first_doc = retrieved_docs[0]
                    print(f"第一个文档类型: {type(first_doc)}")

                    if isinstance(first_doc, dict):
                        print(f"文档键: {list(first_doc.keys())}")
                        if 'contents' in first_doc:
                            print(f"内容预览: {str(first_doc['contents'])[:100]}...")
                    elif isinstance(first_doc, str):
                        print(f"文档是字符串: {first_doc[:100]}...")
                    else:
                        print(f"文档是其他类型: {first_doc}")

                # 检查答案匹配
                is_correct = any(
                    answer.lower().strip() == golden.lower().strip()
                    for golden in sample['golden_answers']
                )
                print(f"答案是否正确: {'✓' if is_correct else '✗'}")

            except Exception as e:
                print(f"❌ 错误: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    diagnose_baseline_methods()