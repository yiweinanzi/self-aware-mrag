#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速测试baseline方法的问题"""

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

# 测试各个baseline方法的导入
methods = [
    ("MuRAG", "experiments.baselines.murag_enhanced", "MuRAGEnhanced"),
    ("VisRAG", "experiments.baselines.visrag_enhanced", "VisRAGEnhanced"),
    ("ViDoRAG", "experiments.baselines.vidorag_pipeline", "ViDoRAGPipeline"),
    ("RagVL", "experiments.baselines.ragvl_enhanced", "RagVLEnhanced"),
    ("mR²AG", "experiments.baselines.mr2ag_enhanced", "MR2AGEnhanced"),
    ("SAM-RAG", "experiments.baselines.samrag_adapted", None)  # 这个可能不存在
]

print("检查baseline方法的可用性：")
print("=" * 60)

for name, module, class_name in methods:
    try:
        mod = __import__(module, fromlist=[class_name])
        if class_name:
            cls = getattr(mod, class_name)
            print(f"✅ {name}: {module}.{class_name}")
        else:
            print(f"✅ {name}: {module} (无特定类)")
    except ImportError as e:
        print(f"❌ {name}: 导入失败 - {e}")
    except AttributeError as e:
        print(f"❌ {name}: 类不存在 - {e}")

# 测试检索器
print("\n\n测试检索器初始化：")
print("=" * 60)

try:
    from flashrag.retriever import DenseRetriever

    config = {
        "retrieval_model": "BAAI/bge-m3",
        "corpus_path": "/data0/home/zqwang/ACL/FlashRAG/dataset/okvqa_wikiqa_musique_todo_corpus.jsonl",
        "embedding_path": None,
        "max_seq_length": 512,
        "batch_size": 32
    }

    retriever = DenseRetriever(config=config)
    print(f"✅ DenseRetriever初始化成功")
    print(f"   - 类型: {type(retriever)}")
    print(f"   - 有search方法: {hasattr(retriever, 'search')}")
    print(f"   - 有retrieve方法: {hasattr(retriever, 'retrieve')}")

    # 测试检索
    print("\n测试检索功能...")
    question = "What sport can you use this for?"
    results = retriever.search(question, num=5)
    print(f"   - 检索结果类型: {type(results)}")
    if isinstance(results, tuple):
        docs, scores = results
        print(f"   - 文档数量: {len(docs)}")
        print(f"   - 第一个文档类型: {type(docs[0]) if docs else 'None'}")
        if docs and isinstance(docs[0], dict):
            print(f"   - 第一个文档的键: {list(docs[0].keys())[:5]}")

except Exception as e:
    print(f"❌ 检索器测试失败: {e}")
    import traceback
    traceback.print_exc()