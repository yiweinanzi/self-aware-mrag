#!/usr/bin/env python
"""Debug RagVL to understand why it's still 0% accuracy"""

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from experiments.baselines.ragvl_enhanced import adapt_ragvl_for_okvqa
from experiments.baselines.answer_extractor import extract_answer_smart

# Test data
test_sample = {
    'question': 'What sport is being played in the image?',
    'image': None,  # We'll test without image first
    'golden_answers': ['race', 'racing', 'motocross']
}

# Create pipeline
print("Testing RagVL with debug...")
print("=" * 60)

pipeline = adapt_ragvl_for_okvqa()

# Test with a simple answer
retrieved_docs = [{'doc_content': 'The image shows a motorcycle race.'}]

print("Testing answer generation...")
result = pipeline.run(test_sample, retrieved_docs)

print(f"\nGenerated answer: '{result}'")
print(f"Extracted answer: '{extract_answer_smart(result)}'")
print(f"Expected answers: {test_sample['golden_answers']}")

# Test matching
from experiments.baselines.answer_matcher import smart_answer_match
extracted = extract_answer_smart(result)
is_match = smart_answer_match(extracted, test_sample['golden_answers'])
print(f"Smart match result: {is_match}")

print("\nDebugging pipeline internals...")
# Check what the pipeline is doing
print(f"Pipeline type: {type(pipeline)}")
print(f"Pipeline has run method: {hasattr(pipeline, 'run')}")