#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版统一评测系统测试
Simplified Test for Unified Evaluation System

不依赖torch等深度学习库，只测试基本功能
"""

import os
import sys
import json
from pathlib import Path

# 添加FlashRAG路径
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')


def test_dataset_paths():
    """测试数据集路径是否存在"""
    print("="*80)
    print("测试1: 数据集路径检查")
    print("="*80)

    dataset_paths = {
        'OK-VQA': '/data1/userdata/zqwang/ACL_data/OK-VQA',
        'A-OKVQA': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/A-OKVQA',
        'MultiModalQA': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MultiModalQA',
        'MRAG-Bench': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MRAG-Bench'
    }

    for dataset_name, path in dataset_paths.items():
        if os.path.exists(path):
            print(f"✅ {dataset_name}: {path}")
            # 列出文件
            files = os.listdir(path)[:5]  # 只显示前5个文件
            print(f"   文件: {files}")
        else:
            print(f"❌ {dataset_name}: {path} (不存在)")


def test_mragbench_evaluator():
    """测试MRAG-Bench评测器"""
    print("\n" + "="*80)
    print("测试2: MRAG-Bench评测器")
    print("="*80)

    # 模拟MRAG-Bench的评测逻辑
    # 参考 MRAG-Bench-main/eval/score.py

    # 创建测试数据
    test_data = [
        {
            'prompt': 'Question about image?',
            'output': 'The answer is A',
            'gt_choice': 'A',
            'scenario': 'Angle'
        },
        {
            'prompt': 'Another question?',
            'output': 'I think the answer is B',
            'gt_choice': 'B',
            'scenario': 'Partial'
        },
        {
            'prompt': 'Third question?',
            'output': 'The correct choice is C',
            'gt_choice': 'C',
            'scenario': 'Scope'
        },
        {
            'prompt': 'Fourth question?',
            'output': 'Answer: B',  # 错误答案
            'gt_choice': 'D',
            'scenario': 'Occlusion'
        }
    ]

    # 简化的答案提取
    def parse_multi_choice_response(response, choices, gt_idx):
        """简化版答案提取"""
        response_upper = response.upper()
        for choice in choices:
            if f"The answer is {choice}" in response_upper or f"Answer: {choice}" in response_upper:
                return choice
        # 如果没有找到，返回第一个选项
        return choices[0]

    # 计算准确率
    correct = 0
    total = len(test_data)
    scenario_stats = {}

    for item in test_data:
        gt = item['gt_choice']
        out = parse_multi_choice_response(item['output'], ['A', 'B', 'C', 'D'], 0)
        scenario = item['scenario']

        if scenario not in scenario_stats:
            scenario_stats[scenario] = {'correct': 0, 'total': 0}

        scenario_stats[scenario]['total'] += 1

        if out == gt:
            correct += 1
            scenario_stats[scenario]['correct'] += 1

    # 打印结果
    overall_accuracy = correct / total * 100
    print(f"总体准确率: {overall_accuracy:.2f}%")
    print("\n场景准确率:")
    for scenario, stats in scenario_stats.items():
        acc = stats['correct'] / stats['total'] * 100
        print(f"  {scenario}: {acc:.2f}% ({stats['correct']}/{stats['total']})")


def test_evaluation_metrics():
    """测试评测指标计算"""
    print("\n" + "="*80)
    print("测试3: 评测指标计算")
    print("="*80)

    # 7个核心指标的简化实现
    def normalize_answer(answer):
        """标准化答案"""
        import string
        answer = answer.lower().translate(str.maketrans('', '', string.punctuation))
        return ' '.join(answer.split())

    def calculate_f1(pred, gold):
        """计算F1分数"""
        pred_tokens = pred.split()
        gold_tokens = gold.split()

        if not pred_tokens and not gold_tokens:
            return 1.0
        if not pred_tokens or not gold_tokens:
            return 0.0

        common = set(pred_tokens) & set(gold_tokens)
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    # 测试数据
    test_cases = [
        {
            'pred': 'A cat sitting on a mat',
            'gold': ['cat on mat', 'feline on carpet'],
            'retrieved': ['cat', 'mat', 'animal']
        },
        {
            'pred': 'The answer is B',
            'gold': ['B'],
            'retrieved': []
        }
    ]

    print("\n测试案例:")
    for i, case in enumerate(test_cases):
        print(f"\n案例 {i+1}:")
        print(f"  预测: {case['pred']}")
        print(f"  标准答案: {case['gold']}")

        # EM (Exact Match)
        pred_norm = normalize_answer(case['pred'])
        gold_norms = [normalize_answer(g) for g in case['gold']]
        em = 1.0 if pred_norm in gold_norms else 0.0

        # F1 (取最佳匹配)
        max_f1 = 0.0
        for gold in gold_norms:
            f1 = calculate_f1(pred_norm, gold)
            max_f1 = max(max_f1, f1)

        # Retrieval Rate
        retrieval_rate = 1.0 if case['retrieved'] else 0.0

        print(f"  EM: {em:.2f}")
        print(f"  F1: {max_f1:.4f}")
        print(f"  Retrieval Rate: {retrieval_rate:.2f}")


def test_file_structure():
    """测试创建的文件结构"""
    print("\n" + "="*80)
    print("测试4: 创建的文件结构")
    print("="*80)

    created_files = [
        '/data0/home/zqwang/ACL/FlashRAG/flashrag/dataset/unified_dataset_loader.py',
        '/data0/home/zqwang/ACL/FlashRAG/flashrag/evaluator/unified_evaluator.py',
        '/data0/home/zqwang/ACL/FlashRAG/flashrag/evaluator/dataset_evaluation_manager.py',
        '/data0/home/zqwang/ACL/FlashRAG/experiments/run_all_datasets_comparison.py'
    ]

    for file_path in created_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
            # 获取文件大小
            size = os.path.getsize(file_path)
            print(f"   大小: {size} bytes")
        else:
            print(f"❌ {file_path} (不存在)")


def main():
    """主测试函数"""
    print("简化版统一评测系统测试")
    print("注意：此测试不需要torch等深度学习库依赖")
    print("="*80)

    # 测试1: 数据集路径
    test_dataset_paths()

    # 测试2: MRAG-Bench评测器
    test_mragbench_evaluator()

    # 测试3: 评测指标
    test_evaluation_metrics()

    # 测试4: 文件结构
    test_file_structure()

    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)
    print("\n下一步:")
    print("1. 确保数据文件已下载到正确位置")
    print("2. 安装必要的依赖: torch, transformers, datasets等")
    print("3. 运行完整评测: python experiments/run_all_datasets_comparison.py")


if __name__ == '__main__':
    main()