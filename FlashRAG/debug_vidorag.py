#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""调试ViDoRAG检索问题"""

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from experiments.baselines.vidorag_pipeline import create_vidorag_pipeline
from flashrag.retriever import DenseRetriever
from flashrag.dataset import UnifiedDatasetLoader
import torch

# 加载数据
loader = UnifiedDatasetLoader(dataset_name="okvqa", split="train")
dataset = loader.load_dataset()
sample = dataset[0]

print("Sample info:")
print(f"  Question: {sample['question']}")
print(f"  Image path: {sample.get('image_path', 'No image')}")

# 加载Qwen3-VL
from flashrag.model.model_qwen3vl import Qwen3VLWrapper
qwen3vl = Qwen3VLWrapper(
    model_path="/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct",
    device="cpu"  # 使用CPU避免GPU问题
)

# 加载检索器
retriever_config = {
    "retrieval_model": "BAAI/bge-m3",
    "corpus_path": "/data0/home/zqwang/ACL/FlashRAG/dataset/okvqa_wikiqa_musique_todo_corpus.jsonl",
    "embedding_path": None,
    "max_seq_length": 512,
    "batch_size": 32
}
retriever = DenseRetriever(config=retriever_config)

print(f"\nRetriever type: {type(retriever)}")
print(f"Retriever has search: {hasattr(retriever, 'search')}")
print(f"Retriever has retrieve: {hasattr(retriever, 'retrieve')}")

# 测试检索
print("\n测试检索...")
try:
    if hasattr(retriever, 'search'):
        results = retriever.search(sample['question'], num=5)
        print(f"Search results type: {type(results)}")
        if isinstance(results, tuple):
            docs, scores = results
            print(f"Retrieved {len(docs)} documents")
            print(f"First doc: {docs[0] if docs else 'None'}")
        else:
            print(f"Results: {results}")
except Exception as e:
    print(f"Search failed: {e}")
    import traceback
    traceback.print_exc()

# 创建ViDoRAG pipeline
print("\n创建ViDoRAG pipeline...")
config = {"retrieval_topk": 5}
pipeline = create_vidorag_pipeline(qwen3vl, retriever, config)

# 运行一个样本
print("\n运行ViDoRAG...")
result = pipeline.run_single(sample)
print(f"\nResult keys: {result.keys()}")
print(f"Used retrieval: {result.get('used_retrieval', 'N/A')}")
print(f"Number of retrieved docs: {len(result.get('retrieved_docs', []))}")
if result.get('retrieved_docs'):
    print(f"First retrieved doc: {result['retrieved_docs'][0]}")
print(f"Answer: {result.get('answer', 'N/A')}")