#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""调试准确率计算问题"""

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from flashrag.utils.vqa_evaluator import evaluate_vqa_accuracy, standardize_vqa_answer

def test_accuracy_calculation():
    """测试准确率计算"""

    # 模拟一些答案和标准答案
    test_cases = [
        # 情况1：完全匹配
        {
            'predicted': 'race',
            'golden_answers': ['race', 'racing', 'car race'],
            'description': '完全匹配'
        },
        # 情况2：部分匹配
        {
            'predicted': 'car racing',
            'golden_answers': ['race', 'racing', 'car race'],
            'description': '部分匹配（标准化后）'
        },
        # 情况3：不匹配
        {
            'predicted': 'baseball',
            'golden_answers': ['race', 'racing', 'car race'],
            'description': '完全不匹配'
        },
        # 情况4：长答案
        {
            'predicted': 'This is a car racing sport that involves competition',
            'golden_answers': ['race', 'racing', 'car race'],
            'description': '长答案（会被截取前3词）'
        }
    ]

    print("="*70)
    print("VQA准确率计算测试")
    print("="*70)

    for i, test in enumerate(test_cases):
        print(f"\n测试 {i+1}: {test['description']}")
        print(f"预测答案: '{test['predicted']}'")
        print(f"标准答案: {test['golden_answers']}")

        # 计算准确率
        result = evaluate_vqa_accuracy(test['predicted'], test['golden_answers'])

        print(f"\n标准化后:")
        print(f"  预测: '{result['processed_pred']}'")
        print(f"  标准答案: {result['processed_gts']}")
        print(f"  匹配数: {result['matches']}/3")
        print(f"  准确率: {result['accuracy']:.2f}%")
        print(f"  是否正确: {result['is_correct']}")

def test_answer_standardization():
    """测试答案标准化"""

    test_answers = [
        "Race car",
        "The answer is racing",
        "RACING!",
        "A car race",
        "This is a sport involving racing cars"
    ]

    print("\n" + "="*70)
    print("答案标准化测试")
    print("="*70)

    for ans in test_answers:
        standardized = standardize_vqa_answer(ans)
        print(f"原始: '{ans}' -> 标准化: '{standardized}'")

if __name__ == "__main__":
    test_accuracy_calculation()
    test_answer_standardization()