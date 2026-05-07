#!/usr/bin/env python3
"""
简单检索器测试脚本
确保检索器能够正常工作
"""
import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

def test_retriever():
    """测试检索器初始化和基本功能"""
    print("🔍 测试检索器功能...")

    try:
        from flashrag.retriever import DenseRetriever

        # 使用成功实验的配置
        config = {
            'retrieval_method': 'dense',
            'retrieval_model_path': '/data0/home/zqwang/ACL/models/bge-large-en-v1.5',
            'index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
            'corpus_path': '/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl',
            'retrieval_topk': 3,
            'save_retrieval_cache': False,
            'use_retrieval_cache': False,
            'retrieval_cache_path': None,
            'embedding_dim': 1024,  # BGE-large embedding dimension
            'use_reranker': False,  # 先不用reranker
            'retrieval_query_max_length': 512,
            'retrieval_pooling_method': 'mean',
            'retrieval_use_fp16': False,
            'retrieval_batch_size': 32,
            'instruction': '',
            'use_sentence_transformer': True,
            'faiss_gpu': False,  # 先用CPU测试
            'silent_retrieval': False,
            'device': 'cpu',
            'rerank_model_name': None,
            'rerank_topk': 5,
            'rerank_max_length': 512,
            'rerank_batch_size': 32,
        }

        print("📦 初始化检索器...")
        retriever = DenseRetriever(config)
        print("✅ 检索器初始化成功!")

        # 测试简单检索
        test_query = "What sport is motorcycle racing?"
        print(f"🔍 测试查询: {test_query}")

        print("📊 执行检索...")
        results = retriever.search(test_query, num=3)

        print(f"✅ 检索成功! 找到 {len(results)} 个结果:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result['title'][:50]}...")
            print(f"     内容: {result['contents'][:100]}...")
            score = result.get('score', result.get('similarity', 0))
            print(f"     分数: {score:.4f}")
            print()

        return True

    except Exception as e:
        print(f"❌ 检索器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_retriever()
    if success:
        print("🎉 检索器测试通过!")
    else:
        print("💥 检索器测试失败!")
    sys.exit(0 if success else 1)