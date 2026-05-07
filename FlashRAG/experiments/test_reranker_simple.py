#!/usr/bin/env python3
"""
简单测试BGE Reranker功能
"""

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from flashrag.retriever import DenseRetriever

# BGE检索器配置（启用reranker）
retriever_config = {
    'index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
    'corpus_path': '/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl',
    'retrieval_method': 'e5',
    'retrieval_model_path': '/data0/home/zqwang/ACL/models/bge-large-en-v1.5',
    'retrieval_query_max_length': 512,
    'retrieval_pooling_method': 'cls',  # BGE-v1.5使用cls pooling
    'retrieval_use_fp16': True,
    'retrieval_batch_size': 128,
    'retrieval_topk': 10,  # 检索10个，然后rerank到5个
    'save_retrieval_cache': False,
    'use_retrieval_cache': False,
    'retrieval_cache_path': None,
    'use_reranker': True,  # 启用reranker
    'rerank_model_name': 'bge-reranker-v2-m3',
    'rerank_model_path': '/data0/home/zqwang/ACL/models/bge-reranker-v2-m3',
    'rerank_topk': 5,  # rerank后保留的文档数
    'rerank_max_length': 512,
    'rerank_batch_size': 32,
    'rerank_use_fp16': True,
    'device': 'cuda',
    'use_sentence_transformer': False,
    'faiss_gpu': False,
    'instruction': '',
}

print("="*80)
print("测试BGE Reranker功能")
print("="*80)

try:
    # 初始化检索器
    print("\n1. 初始化带Reranker的BGE检索器...")
    retriever = DenseRetriever(retriever_config)
    print("✅ 检索器初始化成功")

    # 测试检索
    print("\n2. 测试检索和Rerank...")
    query = "What is the capital of France?"
    print(f"查询: {query}")

    # 执行检索（会自动进行rerank）
    results = retriever.search(query, num=5, return_score=True)

    if isinstance(results, tuple):
        docs, scores = results
        print(f"\n✅ 检索成功，返回{len(docs)}个文档")
        print(f"最高分: {scores[0]:.4f}")
        print(f"最低分: {scores[-1]:.4f}")

        # 显示前3个结果
        print("\n前3个检索结果:")
        for i, (doc, score) in enumerate(zip(docs[:3], scores[:3])):
            print(f"\n{i+1}. [Score: {score:.4f}]")
            print(f"   {doc[:200]}...")
    else:
        print(f"✅ 检索成功，返回{len(results)}个文档")

except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("测试完成")
print("="*80)