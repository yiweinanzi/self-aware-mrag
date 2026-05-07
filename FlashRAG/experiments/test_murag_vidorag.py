#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from flashrag.dataset.unified_dataset_loader import UnifiedDatasetLoader
from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
from flashrag.retriever import DenseRetriever
from experiments.baselines.murag_enhanced import MuRAGEnhanced
from experiments.baselines.vidorag_pipeline import ViDoRAGPipeline

# 配置
CONFIG = {
    'dataset_name': 'okvqa',
    'max_samples': 3,
    'load_images': False,
    'okvqa_data_path': '/data1/userdata/zqwang/ACL_data/VQA',
    'qwen3_vl_path': '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
    'torch_dtype': 'bfloat16',
    'temperature': 0.01,
    'max_new_tokens': 30,
    'faiss_index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
    'corpus_path': '/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl',
    'retrieval_model_path': '/data0/home/zqwang/ACL/models/bge-large-en-v1.5',
    'retrieval_topk': 5,
    'bge_reranker_path': '/data0/home/zqwang/ACL/models/bge-reranker-v2-m3',
}

def main():
    print("="*80)
    print("测试 MuRAG 和 ViDoRAG 初始化")
    print("="*80)

    # 1. 加载数据
    print("\n1. 加载数据集")
    print("-"*40)
    try:
        loader = UnifiedDatasetLoader()
        dataset = loader.load_dataset(
            dataset_name=CONFIG['dataset_name'],
            split='val',
            max_samples=CONFIG['max_samples']
        )
        samples = [dataset[i] for i in range(len(dataset))]
        print(f"✅ 成功加载 {len(samples)} 个样本")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. 初始化模型
    print("\n2. 初始化模型")
    print("-"*40)
    try:
        qwen3_vl = create_qwen3_vl_wrapper(
            model_path=CONFIG['qwen3_vl_path'],
            device='cuda',
            torch_dtype=CONFIG['torch_dtype'],
            temperature=CONFIG['temperature'],
            max_new_tokens=CONFIG['max_new_tokens']
        )
        print("✅ Qwen3-VL加载成功")
    except Exception as e:
        print(f"❌ Qwen3-VL加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 初始化检索器
    print("\n3. 初始化检索器")
    print("-"*40)
    try:
        retriever_config = {
            'retrieval_model_path': CONFIG['retrieval_model_path'],
            'faiss_index_path': CONFIG['faiss_index_path'],
            'corpus_path': CONFIG['corpus_path'],
            'retrieval_cache_path': None,
            'use_reranker': False,
            'use_sentence_transformer': False,
            'faiss_gpu': False,
            'instruction': '',
        }
        retriever = DenseRetriever(retriever_config)
        print("✅ 检索器加载成功")
    except Exception as e:
        print(f"⚠️ 检索器加载失败: {e}")
        retriever = None

    # 4. 测试MuRAG
    print("\n4. 测试 MuRAG")
    print("-"*40)
    try:
        print("[INFO] 初始化 MuRAG...")
        murag = MuRAGEnhanced(qwen3_vl, retriever, CONFIG)
        print("✅ MuRAG 初始化成功")

        # 测试处理一个样本
        print("[INFO] 测试处理样本...")
        sample = samples[0]
        result = murag.run_single(sample)
        print(f"✅ MuRAG 测试成功")
        print(f"   输入: {sample.get('question', '')[:50]}...")
        print(f"   输出: {result.get('answer', '')[:50]}...")
    except Exception as e:
        print(f"❌ MuRAG 测试失败: {e}")
        import traceback
        traceback.print_exc()

    # 5. 测试ViDoRAG
    print("\n5. 测试 ViDoRAG")
    print("-"*40)
    try:
        print("[INFO] 初始化 ViDoRAG...")
        vidorag = ViDoRAGPipeline(qwen3_vl, retriever, CONFIG)
        print("✅ ViDoRAG 初始化成功")

        # 测试处理一个样本
        print("[INFO] 测试处理样本...")
        sample = samples[0]
        result = vidorag.run_single(sample)
        print(f"✅ ViDoRAG 测试成功")
        print(f"   输入: {sample.get('question', '')[:50]}...")
        print(f"   输出: {result.get('answer', '')[:50]}...")
    except Exception as e:
        print(f"❌ ViDoRAG 测试失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)

if __name__ == "__main__":
    import torch
    import gc
    main()