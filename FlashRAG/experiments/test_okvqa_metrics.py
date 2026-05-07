#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试OK-VQA评价指标是否正确
"""

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from run_okvqa_baselines_final import OKVQAEvaluator

def test_metrics():
    print("测试OK-VQA评价指标...")

    evaluator = OKVQAEvaluator()

    # 测试数据
    test_cases = [
        {
            'answer': 'red apple',
            'golden_answers': ['red apple', 'apple', 'red fruit'],
            'retrieved_docs': [
                {'contents': 'The apple is red and sweet.'},
                {'contents': 'A red apple grows on trees.'}
            ]
        },
        {
            'answer': 'yellow',
            'golden_answers': ['yellow', 'gold', 'amber'],
            'retrieved_docs': [
                {'contents': 'The banana is yellow.'},
                {'contents': 'Gold is a precious metal.'}
            ]
        },
        {
            'answer': 'cat',
            'golden_answers': ['dog', 'animal', 'pet'],
            'retrieved_docs': []
        }
    ]

    # 评估
    results = evaluator.evaluate_batch(test_cases)

    print("\n测试结果:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

    print("\n✅ 评价指标测试完成！")
    return results

if __name__ == "__main__":
    test_metrics()