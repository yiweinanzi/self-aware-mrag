#!/usr/bin/env python
"""Debug script to check what answers ViDoRAG, RagVL, and SAM-RAG are generating"""

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL')
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

import json
from datasets import load_dataset
from flashrag.utils.vqa_evaluator import extract_okvqa_answer

# Load model
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')
from flashrag.model.qwen3vl_wrapper import Qwen3VLWrapper

qwen3vl = Qwen3VLWrapper(
    model_path="/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct",
    device="cuda",
    dtype="bfloat16",
    thinking=False
)

# Load retriever
from flashrag.retriever.denseretriver import DenseRetriever
from flashrag.config import Config

config_obj = Config()
retriever = DenseRetriever(
    retrieval_config={
        "retrieval_model_path": "/data0/home/zqwang/ACL/models/bge-large-en-v1.5",
        "faiss_index_path": "/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index",
        "corpus_path": "/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl",
        "retrieval_topk": 5
    },
    config_obj=config_obj
)

# Load methods
from experiments.baselines.vidorag_pipeline import create_vidorag_pipeline
from experiments.baselines.ragvl_enhanced import create_ragvl_enhanced
from experiments.baselines.samrag_adapted import create_samrag_adapted

methods = {
    "ViDoRAG": create_vidorag_pipeline(qwen3vl, retriever, {"retrieval_topk": 5}),
    "RagVL": create_ragvl_enhanced(qwen3vl, retriever, {"retrieval_topk": 5}),
    "SAM-RAG": create_samrag_adapted(qwen3vl, retriever, {"retrieval_topk": 5})
}

# Load dataset
dataset = load_dataset("lmms-lab/OK-VQA", split="test", trust_remote_code=True)
sample = dataset[0]

print("=" * 60)
print("Sample 0")
print(f"Question: {sample['question']}")
print(f"Golden Answers: {sample['answers']}")
print(f"Image ID: {sample['image_id']}")
print("=" * 60)

# Test each method
for method_name, method in methods.items():
    print(f"\n{method_name}:")
    print("-" * 40)

    # Prepare sample dict
    test_sample = {
        "question": sample["question"],
        "image": f"/data0/home/zqwang/ACL/OK-VQA/images/{sample['image_id']}.jpg",
        "golden_answers": sample["answers"]
    }

    try:
        result = method.run_single(test_sample)
        print(f"Answer: '{result['answer']}'")
        print(f"Expected: {sample['answers'][:3]}")

        # Check if answer matches
        is_correct = any(result['answer'].lower() == ans.lower() for ans in sample['answers'])
        print(f"Correct: {is_correct}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()