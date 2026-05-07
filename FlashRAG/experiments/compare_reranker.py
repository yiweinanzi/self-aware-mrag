#!/usr/bin/env python3
"""
比较使用和不使用Reranker的检索效果
"""

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from flashrag.retriever import DenseRetriever

# 测试查询
queries = [
    "What is the capital of France?",
    "Who painted the Mona Lisa?",
    "What is the largest planet in our solar system?",
    "When did World War II end?",
    "What is the chemical symbol for gold?",
]

# BGE检索器配置（不使用reranker）
config_no_reranker = {
    'index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
    'corpus_path': '/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl',
    'retrieval_method': 'e5',
    'retrieval_model_path': '/data0/home/zqwang/ACL/models/bge-large-en-v1.5',
    'retrieval_query_max_length': 512,
    'retrieval_pooling_method': 'cls',
    'retrieval_use_fp16': True,
    'retrieval_batch_size': 128,
    'retrieval_topk': 5,
    'save_retrieval_cache': False,
    'use_retrieval_cache': False,
    'retrieval_cache_path': None,
    'use_reranker': False,  # 不使用reranker
    'device': 'cuda',
    'use_sentence_transformer': False,
    'faiss_gpu': False,
    'instruction': '',
}

# BGE检索器配置（使用reranker）
config_with_reranker = {
    'index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
    'corpus_path': '/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl',
    'retrieval_method': 'e5',
    'retrieval_model_path': '/data0/home/zqwang/ACL/models/bge-large-en-v1.5',
    'retrieval_query_max_length': 512,
    'retrieval_pooling_method': 'cls',
    'retrieval_use_fp16': True,
    'retrieval_batch_size': 128,
    'retrieval_topk': 5,
    'save_retrieval_cache': False,
    'use_retrieval_cache': False,
    'retrieval_cache_path': None,
    'use_reranker': True,  # 使用reranker
    'rerank_model_name': 'bge-reranker-v2-m3',
    'rerank_model_path': '/data0/home/zqwang/ACL/models/bge-reranker-v2-m3',
    'rerank_topk': 5,
    'rerank_max_length': 512,
    'rerank_batch_size': 32,
    'rerank_use_fp16': True,
    'device': 'cuda',
    'use_sentence_transformer': False,
    'faiss_gpu': False,
    'instruction': '',
}

def test_retriever(retriever, name, queries):
    """测试检索器"""
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"{'='*60}")

    for query in queries:
        print(f"\n查询: {query}")
        results = retriever.search(query, num=5, return_score=True)

        if isinstance(results, tuple):
            docs, scores = results
            print(f"返回 {len(docs)} 个文档:")
            for i, (doc, score) in enumerate(zip(docs[:3], scores[:3])):
                print(f"  {i+1}. [Score: {score:.4f}] {doc[:100]}...")
        else:
            print(f"返回 {len(results)} 个文档")

# 初始化检索器
print("初始化检索器...")

# 不使用reranker
print("\n1. 初始化BGE检索器（不使用Reranker）...")
retriever_no_reranker = DenseRetriever(config_no_reranker)
print("✅ 初始化完成")

# 使用reranker
print("\n2. 初始化BGE检索器（使用Reranker）...")
retriever_with_reranker = DenseRetriever(config_with_reranker)
print("✅ 初始化完成")

# 测试
test_retriever(retriever_no_reranker, "BGE (无Reranker)", queries)
test_retriever(retriever_with_reranker, "BGE + Reranker", queries)

print("\n" + "="*60)
print("测试完成")
print("="*60)