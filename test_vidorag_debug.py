#!/usr/bin/env python3
"""
Test ViDoRAG with debug output
"""

import sys
import os
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')
sys.path.insert(0, '/data0/home/zqwang/ACL/ViDoRAG-main')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Import after setting path
from flashrag.utils.qwen3_vl import Qwen3VLWrapper
from experiments.baselines.vidorag_pipeline import create_vidorag_pipeline

# Initialize Qwen3-VL
print("Initializing Qwen3-VL...")
qwen3vl = Qwen3VLWrapper(
    model_path='/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
    torch_dtype='bfloat16'
)

# Create test sample
sample = {
    'question': 'What color is the sky?',
    'golden_answers': ['blue']
}

# Create mock retriever
class MockRetriever:
    def search(self, query, num=5):
        print(f"[MockRetriever] search called with query: {query}, num: {num}")
        docs = [
            {'contents': 'The sky is blue.', 'id': '1', 'title': '', 'source': 'test'},
            {'contents': 'Blue is the color of the sky.', 'id': '2', 'title': '', 'source': 'test'}
        ]
        return docs, [1.0, 0.9]

# Create pipeline
config = {'retrieval_topk': 3}
pipeline = create_vidorag_pipeline(qwen3vl, MockRetriever(), config)

print("\n" + "="*60)
print("Running ViDoRAG on test sample...")
print("="*60)

result = pipeline.run_single(sample)

print("\n" + "="*60)
print("Result:")
print("="*60)
print(f"  Question: {result['question']}")
print(f"  Answer: '{result['answer']}'")
print(f"  Retrieved: {result['retrieved']}")
print(f"  Correct: {result['correct']}")
print(f"  Num docs: {len(result.get('retrieved_docs', []))}")