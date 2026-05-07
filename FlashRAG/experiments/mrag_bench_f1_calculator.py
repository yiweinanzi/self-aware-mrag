#!/usr/bin/env python3
"""
MRAG-Bench 专用F1计算器

专门为多选题设计的F1计算逻辑：
- 对于多选题，标准F1可能不适用
- 因为答案要么完全正确，要么完全错误
- 提供多种F1计算选项供选择
"""

from typing import List, Dict, Any
import numpy as np


class MRAGBenchF1Calculator:
    """MRAG-Bench专用F1计算器"""

    def __init__(self):
        self.name = "MRAG-Bench F1 Calculator"

    def calculate_standard_f1(self, results: List[Dict], samples: List[Dict]) -> Dict[str, float]:
        """
        计算标准F1（用于对比）

        Args:
            results: 预测结果列表
            samples: 数据样本列表

        Returns:
            F1相关指标
        """
        tp = fp = fn = 0

        for result, sample in zip(results, samples):
            # 获取标准答案
            gt = sample.get('answer_choice', '').upper()
            pred = result.get('answer', '').strip().upper()

            if gt and pred:
                if gt == pred:
                    tp += 1
                else:
                    fp += 1
                    fn += 1

        # 计算精确率、召回率、F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }

    def calculate_em_based_f1(self, results: List[Dict], samples: List[Dict]) -> float:
        """
        基于Exact Match的F1计算
        对于多选题，这是最合理的F1定义

        Args:
            results: 预测结果列表
            samples: 数据样本列表

        Returns:
            F1分数（等于EM）
        """
        correct = 0
        total = 0

        for result, sample in zip(results, samples):
            gt = sample.get('answer_choice', '').upper()
            pred = result.get('answer', '').strip().upper()

            if gt and pred:
                if gt == pred:
                    correct += 1
                total += 1

        # 对于多选题，F1 = Exact Match
        return correct / total if total > 0 else 0

    def calculate_partial_credit_f1(self, results: List[Dict], samples: List[Dict]) -> Dict[str, float]:
        """
        部分学分F1计算（如果需要给部分分数）

        考虑：
        1. 完全匹配：1.0分
        2. 错误匹配：0.0分
        3. 可以扩展为考虑语义相似度
        """
        scores = []

        for result, sample in zip(results, samples):
            gt = sample.get('answer_choice', '').upper()
            pred = result.get('answer', '').strip().upper()

            if gt and pred:
                # 简单的二值评分
                score = 1.0 if gt == pred else 0.0
                scores.append(score)

        avg_score = np.mean(scores) if scores else 0

        return {
            'avg_score': avg_score,
            'f1': avg_score,  # 平均分数作为F1
            'scores': scores
        }

    def calculate_multi_label_f1(self, results: List[Dict], samples: List[Dict]) -> Dict[str, float]:
        """
        多标签F1计算（如果问题可能有多个正确答案）

        对于MRAG-Bench通常不适用，因为每题只有一个正确答案
        """
        # 实际MRAG-Bench是单选，所以这个方法返回与EM相同的值
        return {
            'f1': self.calculate_em_based_f1(results, samples),
            'note': 'MRAG-Bench是单选题，多标签F1等于EM'
        }

    def explain_f1_approaches(self):
        """解释不同的F1计算方法"""
        print("="*60)
        print("MRAG-Bench F1计算方法说明")
        print("="*60)
        print("\n1. 标准F1计算:")
        print("   - TP: 预测正确")
        print("   - FP: 预测错误")
        print("   - FN: 未预测（多选题不适用）")
        print("   - 结果：对于多选题通常很低")

        print("\n2. EM-Based F1（推荐）:")
        print("   - F1 = Exact Match")
        print("   - 完全正确才给分")
        print("   - 结果：等于准确率")

        print("\n3. 部分学分F1:")
        print("   - 可以基于语义相似度")
        print("   - 需要额外相似度计算")
        print("   - 结果：0到1之间的连续值")

        print("\n建议：")
        print("- 在论文中使用EM-Based F1")
        print("- 说明为什么对于多选题F1=EM")
        print("- 可以同时展示准确率和EM/F1")


def test_f1_calculator():
    """测试F1计算器"""
    print("测试MRAG-Bench F1计算器")
    print("-"*40)

    # 模拟数据
    samples = [
        {'answer_choice': 'A'},
        {'answer_choice': 'B'},
        {'answer_choice': 'C'},
        {'answer_choice': 'D'},
        {'answer_choice': 'A'},
    ]

    results = [
        {'answer': 'A'},  # 正确
        {'answer': 'C'},  # 错误
        {'answer': 'C'},  # 正确
        {'answer': 'D'},  # 正确
        {'answer': 'B'},  # 错误
    ]

    calculator = MRAGBenchF1Calculator()

    # 计算不同类型的F1
    print("\n1. 标准F1:")
    standard_f1 = calculator.calculate_standard_f1(results, samples)
    print(f"   Precision: {standard_f1['precision']:.3f}")
    print(f"   Recall: {standard_f1['recall']:.3f}")
    print(f"   F1: {standard_f1['f1']:.3f}")

    print("\n2. EM-Based F1:")
    em_f1 = calculator.calculate_em_based_f1(results, samples)
    print(f"   F1: {em_f1:.3f}")
    print(f"   (等于准确率: {em_f1:.1%})")

    print("\n3. 部分学分F1:")
    partial_f1 = calculator.calculate_partial_credit_f1(results, samples)
    print(f"   F1: {partial_f1['f1']:.3f}")

    # 解释方法
    calculator.explain_f1_approaches()


if __name__ == "__main__":
    test_f1_calculator()