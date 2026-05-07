#!/usr/bin/env python3
"""
Test ViDoRAG with single sample
"""

import sys
import os
sys.path.insert(0, '/data0/home/zqwang/ACL')
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')
sys.path.insert(0, '/data0/home/zqwang/ACL/ViDoRAG-main')

# Set environment
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
    'question': 'What color is the apple?',
    'image': None,  # No image for this test
    'golden_answers': ['red', 'green']
}

# Create pipeline
config = {
    'retrieval_topk': 3,
}

# Create mock retriever that returns some documents
class MockRetriever:
    def search(self, query, num=5):
        return [
            {'contents': 'The apple is red in color.', 'id': '1', 'title': '', 'source': 'test'},
            {'contents': 'Apples can be red or green.', 'id': '2', 'title': '', 'source': 'test'},
        ], [1.0, 0.9]

pipeline = create_vidorag_pipeline(qwen3vl, MockRetriever(), config)

print("\nRunning ViDoRAG on test sample...")
result = pipeline.run_single(sample)

print("\nResult:")
print(f"  Question: {result['question']}")
print(f"  Answer: '{result['answer']}'")
print(f"  Retrieved: {result['retrieved']}")
print(f"  Correct: {result['correct']}")
print(f"  Num docs: {len(result.get('retrieved_docs', []))}")