#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""简单测试ViDoRAG"""

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

import json
from experiments.baselines.vidorag_pipeline import create_vidorag_pipeline
from flashrag.model.model_qwen3vl import Qwen3VLWrapper
from flashrag.retriever import DenseRetriever

# 加载一个样本
with open('/data0/home/zqwang/ACL/FlashRAG/dataset/okvqa_test.json', 'r') as f:
    data = json.load(f)
sample = data[0]

print("Sample keys:", sample.keys())
print("Question:", sample['question'][:100] + "...")

# 加载Qwen3-VL (CPU)
print("\nLoading Qwen3-VL...")
qwen3vl = Qwen3VLWrapper(
    model_path="/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct",
    device="cpu"
)

# 加载检索器
print("\nLoading retriever...")
retriever_config = {
    "retrieval_model": "BAAI/bge-m3",
    "corpus_path": "/data0/home/zqwang/ACL/FlashRAG/dataset/okvqa_wikiqa_musique_todo_corpus.jsonl",
    "embedding_path": None,
    "max_seq_length": 512,
    "batch_size": 32
}
retriever = DenseRetriever(config=retriever_config)

print(f"Retriever: {type(retriever)}")
print(f"Has search: {hasattr(retriever, 'search')}")

# 测试检索
print("\nTesting retrieval...")
question = sample['question']
try:
    results = retriever.search(question, num=5)
    print(f"Search results type: {type(results)}")
    if isinstance(results, tuple):
        docs, scores = results
        print(f"Retrieved {len(docs)} documents")
        print(f"First doc type: {type(docs[0]) if docs else 'None'}")
        if docs and isinstance(docs[0], dict):
            print(f"First doc keys: {list(docs[0].keys())}")
    else:
        print(f"Results: {results}")
except Exception as e:
    print(f"Search failed: {e}")
    import traceback
    traceback.print_exc()

# 创建ViDoRAG pipeline
print("\nCreating ViDoRAG pipeline...")
config = {"retrieval_topk": 5}
try:
    pipeline = create_vidorag_pipeline(qwen3vl, retriever, config)
    print("✅ ViDoRAG pipeline created successfully")

    # 运行一个样本
    print("\nRunning ViDoRAG on sample...")
    result = pipeline.run_single(sample)

    print(f"\nResults:")
    print(f"  Used retrieval: {result.get('used_retrieval', False)}")
    print(f"  Retrieved docs: {len(result.get('retrieved_docs', []))}")
    print(f"  Answer: {result.get('answer', 'N/A')}")

except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()