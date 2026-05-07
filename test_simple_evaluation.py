#!/usr/bin/env python
"""Simple test to verify smart answer matching works"""

import sys
import os
sys.path.insert(0, os.path.expanduser('~/ACL/FlashRAG'))

# Import just the matcher
from experiments.baselines.answer_matcher import smart_answer_match

# Test cases
test_cases = [
    {
        'answer': 'racing',
        'golden_answers': ['race', 'race', 'race'],
        'expected': True
    },
    {
        'answer': 'motorcycle racing',
        'golden_answers': ['race', 'racing', 'motocross'],
        'expected': True
    },
    {
        'answer': 'tennis',
        'golden_answers': ['basketball', 'football', 'baseball'],
        'expected': False
    },
    {
        'answer': 'red',
        'golden_answers': ['red color', 'reddish', 'red'],
        'expected': True
    },
    {
        'answer': 'The sport is racing',
        'golden_answers': ['race', 'racing', 'motocross'],
        'expected': True
    }
]

print("Testing smart answer matcher...")
print("=" * 60)

all_passed = True
for i, test in enumerate(test_cases, 1):
    result = smart_answer_match(test['answer'], test['golden_answers'])
    passed = result == test['expected']
    all_passed = all_passed and passed

    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"Test {i}: {status}")
    print(f"  Answer: '{test['answer']}'")
    print(f"  Golden: {test['golden_answers']}")
    print(f"  Expected: {test['expected']}, Got: {result}")
    print()

print(f"Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")

if all_passed:
    print("\n🎉 The smart answer matcher is working correctly!")
    print("This should fix the 0% accuracy issue for methods that generate")
    print("answers like 'racing' when the expected answer is 'race'")