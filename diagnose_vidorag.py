#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断ViDoRAG的检索问题"""

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

import torch
from flashrag.retriever import BM25Retriever, DenseRetriever

def diagnose_vidorag_retrieval():
    """诊断ViDoRAG的检索问题"""

    print("="*70)
    print("ViDoRAG 检索诊断")
    print("="*70)

    # 配置（与run_okvqa_baselines.py相同）
    config = {
        'faiss_index_path': '/data0/home/zqwang/ACL/FlashRAG/index/okvqa/bge_index.faiss',
        'corpus_path': '/data0/home/zqwang/ACL/FlashRAG/index/okvqa/corpus.json',
        'retrieval_model_path': '/data0/home/zqwang/ACL/models/bge-m3',
        'retrieval_topk': 5
    }

    # 1. 初始化检索器
    print("\n1. 初始化检索器...")
    try:
        retriever = BM25Retriever(
            faiss_path=config['faiss_index_path'],
            corpus_path=config['corpus_path']
        )
        print(f"✅ 检索器类型: {type(retriever)}")
        print(f"   是否有search方法: {hasattr(retriever, 'search')}")
        print(f"   是否有retrieve方法: {hasattr(retriever, 'retrieve')}")
    except Exception as e:
        print(f"❌ 检索器初始化失败: {e}")
        return

    # 2. 测试检索
    print("\n2. 测试检索功能...")
    test_query = "What sport can you use this for?"

    try:
        # 测试search方法
        if hasattr(retriever, 'search'):
            print(f"\n测试search方法...")
            results = retriever.search(test_query, num=5)
            print(f"   结果类型: {type(results)}")
            print(f"   结果内容: {results[:200] if isinstance(results, str) else results}")

            if isinstance(results, tuple):
                docs, scores = results
                print(f"   文档数量: {len(docs)}")
                print(f"   第一个文档: {str(docs[0])[:100]}..." if docs else "   无文档")
                print(f"   第一个分数: {scores[0]}" if scores else "   无分数")

        # 测试retrieve方法
        if hasattr(retriever, 'retrieve'):
            print(f"\n测试retrieve方法...")
            results = retriever.retrieve(query_text=test_query, top_k=5)
            print(f"   结果类型: {type(results)}")
            print(f"   结果内容: {results[:200] if isinstance(results, str) else results}")

    except Exception as e:
        print(f"❌ 检索测试失败: {e}")
        import traceback
        traceback.print_exc()

    # 3. 模拟ViDoRAG的处理
    print("\n3. 模拟ViDoRAG的检索处理...")
    try:
        # 这是ViDoRAGPipeline.run_single中的逻辑
        top_k = 5

        if hasattr(retriever, 'search'):
            search_results = retriever.search(test_query, num=top_k)
            print(f"   search_results: {type(search_results)}")

            if isinstance(search_results, tuple):
                retrieved_docs, retrieval_scores = search_results
                print(f"   retrieved_docs数量: {len(retrieved_docs)}")
                print(f"   retrieved_docs[0]类型: {type(retrieved_docs[0]) if retrieved_docs else 'None'}")
                print(f"   retrieval_scores数量: {len(retrieval_scores)}")
            else:
                retrieved_docs = search_results if search_results else []
                retrieval_scores = [1.0] * len(retrieved_docs) if retrieved_docs else []
                print(f"   retrieved_docs数量: {len(retrieved_docs)}")

        print(f"   最终: 是否有文档？ {bool(retrieved_docs)}")
        print(f"   最终: 文档数量: {len(retrieved_docs)}")

    except Exception as e:
        print(f"❌ 模拟处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_vidorag_retrieval()