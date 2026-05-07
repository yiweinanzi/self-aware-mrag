#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估指标测试脚本
Test Evaluation Metrics

验证7个核心指标是否能正常计算
"""

import sys
import warnings
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from flashrag.evaluator.complete_metrics import CompleteMetricsCalculator

def test_metrics():
    """测试评估指标计算"""
    print("="*80)
    print("测试评估指标计算")
    print("="*80)

    # 创建模拟数据
    predictions = [
        "cat",
        "dog",
        "red car",
        "mountain",
        "apple"
    ]

    golden_answers = [
        ["cat", "feline"],  # 正确
        ["dog", "canine"],  # 正确
        ["red"],           # 部分正确
        ["ocean"],         # 错误
        ["fruit", "apple"] # 正确
    ]

    retrieval_results = [
        [{"contents": "A cat is a small domesticated carnivorous mammal"}],  # 相关
        [{"contents": "Dogs are loyal pets"}],  # 相关
        [{"contents": "The car was painted red"}],  # 相关
        [{"contents": "Mountains are tall landforms"}],  # 相关
        [{"contents": "Apples are fruits that grow on trees"}]  # 相关
    ]

    # 创建Mock数据对象
    class MockData:
        def __init__(self, pred, golden_answers, retrieval_result):
            self.pred = pred
            self.golden_answers = golden_answers
            self.retrieval_result = retrieval_result
            self.items = [{'golden_answers': ga} for ga in golden_answers]
            self.choices = [[] for _ in pred]

    data = MockData(predictions, golden_answers, retrieval_results)

    # 计算指标
    try:
        config = {
            'use_llm_judge': False,  # 使用简化版
            'dataset_name': 'test',
            'metric_setting': {
                'retrieval_recall_topk': 5,
            }
        }

        calculator = CompleteMetricsCalculator(config)
        results = calculator.calculate_all_metrics(data)

        print("✅ 指标计算成功:")
        print(f"   EM: {results.get('em', 0):.4f}")
        print(f"   F1: {results.get('f1', 0):.4f}")
        print(f"   Recall@5: {results.get('retrieval_recall_top5', 0):.4f}")
        print(f"   VQA-Score: {results.get('vqa_score', 0):.4f}")
        print(f"   Faithfulness: {results.get('faithfulness', 0):.4f}")
        print(f"   Attribution Precision: {results.get('attribution_precision', 0):.4f}")
        print(f"   Position Bias Score: {results.get('position_bias_score', 0):.4f}")

        return True

    except Exception as e:
        print(f"❌ 指标计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vqa_specific():
    """测试VQA特定场景"""
    print("\n" + "="*80)
    print("测试VQA特定场景")
    print("="*80)

    # VQA类问题和答案
    vqa_predictions = [
        "cat",           # 正确
        "A dog",         # 正确（多词）
        "red",           # 部分正确
        "car",           # 错误
        "Yes, apple"     # 正确（带前缀）
    ]

    vqa_golden_answers = [
        ["cat"],
        ["dog"],
        ["red car"],
        ["blue car"],
        ["apple"]
    ]

    retrieval_results = [
        [{"contents": "Cat information"}] for _ in range(5)
    ]

    class MockData:
        def __init__(self, pred, golden_answers, retrieval_result):
            self.pred = pred
            self.golden_answers = [[ans] if isinstance(ans, str) else ans for ans in golden_answers]
            self.retrieval_result = retrieval_result
            self.items = [{'golden_answers': ga} for ga in self.golden_answers]
            self.choices = [[] for _ in pred]

    data = MockData(vqa_predictions, vqa_golden_answers, retrieval_results)

    try:
        config = {
            'use_llm_judge': False,
            'dataset_name': 'vqa_test',
            'metric_setting': {
                'retrieval_recall_topk': 5,
            }
        }

        calculator = CompleteMetricsCalculator(config)
        results = calculator.calculate_all_metrics(data)

        print("✅ VQA指标计算成功:")
        print(f"   EM: {results.get('em', 0):.4f}")
        print(f"   F1: {results.get('f1', 0):.4f}")
        print(f"   VQA-Score: {results.get('vqa_score', 0):.4f}")

        return True

    except Exception as e:
        print(f"❌ VQA指标计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success1 = test_metrics()
    success2 = test_vqa_specific()

    print("\n" + "="*80)
    if success1 and success2:
        print("🎉 所有测试通过！评估指标可以正常使用。")
    else:
        print("❌ 测试失败，需要修复评估指标。")
    print("="*80)