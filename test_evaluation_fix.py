#!/usr/bin/env python
"""Quick test to verify evaluation fix works"""

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from experiments.baselines.evaluation_helper import evaluate_answer_correctness

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

print("Testing evaluation fix...")
print("=" * 60)

all_passed = True
for i, test in enumerate(test_cases, 1):
    result = evaluate_answer_correctness(test['answer'], test['golden_answers'])
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
    print("\n🎉 The evaluation fix is working correctly!")
    print("The smart answer matcher should now properly evaluate answers like 'racing' -> 'race'")