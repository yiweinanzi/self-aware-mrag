#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试统一评测系统
Test Unified Evaluation System

测试功能：
1. 数据集加载
2. 统一评测
3. 报告生成
"""

import sys
import json
from pathlib import Path

# 添加FlashRAG路径
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from flashrag.dataset.unified_dataset_loader import load_unified_dataset
from flashrag.evaluator.dataset_evaluation_manager import DatasetEvaluationManager


def test_dataset_loading():
    """测试数据集加载"""
    print("="*80)
    print("测试1: 数据集加载")
    print("="*80)

    datasets = ['okvqa', 'a-okvqa', 'multimodalqa', 'mrag-bench']

    for dataset_name in datasets:
        try:
            print(f"\n加载 {dataset_name.upper()} 数据集...")
            dataset = load_unified_dataset(
                dataset_name,
                split='val',
                max_samples=10  # 只加载10个样本用于测试
            )

            print(f"✅ {dataset_name.upper()} 加载成功:")
            stats = dataset.get_stats()
            for key, value in stats.items():
                print(f"  {key}: {value}")

            # 检查第一个样本
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"\n样本示例 (id: {sample.get('id', 'N/A')}):")
                print(f"  问题: {sample.get('question', '')[:100]}...")
                print(f"  答案: {sample.get('golden_answers', [])[:3]}")

        except Exception as e:
            print(f"❌ {dataset_name.upper()} 加载失败: {e}")


def test_unified_evaluation():
    """测试统一评测"""
    print("\n" + "="*80)
    print("测试2: 统一评测")
    print("="*80)

    # 创建评测管理器
    config = {
        'max_samples': 10,  # 只测试10个样本
        'output_dir': '/data0/home/zqwang/ACL/FlashRAG/experiments/test_unified_evaluation',
        'save_results': True,
        'generate_report': True,
        'run_inference': False  # 不运行真实推理
    }

    manager = DatasetEvaluationManager(config)

    # 运行评测
    results = manager.run_evaluation(
        datasets=['okvqa', 'mrag-bench'],  # 只测试两个数据集
        results_dir=None
    )

    # 打印结果摘要
    print("\n评测结果摘要:")
    for dataset_name, result in results.items():
        if 'metrics' in result:
            metrics = result['metrics']
            print(f"\n{dataset_name.upper()}:")
            print(f"  准确率: {metrics.get('accuracy', 0)*100:.2f}%")
            print(f"  F1: {metrics.get('avg_F1', 0):.4f}")
            print(f"  检索率: {metrics.get('retrieval_rate', 0)*100:.1f}%")


def test_mragbench_evaluation():
    """测试MRAG-Bench专用评测"""
    print("\n" + "="*80)
    print("测试3: MRAG-Bench专用评测")
    print("="*80)

    from flashrag.evaluator.unified_evaluator import evaluate_unified

    # 创建模拟数据
    predictions = []
    references = []

    scenarios = ['Angle', 'Partial', 'Scope', 'Occlusion', 'Temporal']
    choices = ['A', 'B', 'C', 'D']

    for i, scenario in enumerate(scenarios):
        # 正确答案
        pred = {
            'answer': choices[i % 4],
            'retrieved_docs': [f'Doc {j}' for j in range(5)],
            'retrieval_result': [{
                'retrieved_docs': [f'Doc {j}' for j in range(5)],
                'retrieved_scores': [0.9 - j*0.1 for j in range(5)],
                'retrieved_used': True
            }]
        }

        ref = {
            'golden_answers': [choices[i % 4]],
            'scenario': scenario,
            'dataset': 'mrag-bench'
        }

        predictions.append(pred)
        references.append(ref)

    # 评测
    metrics = evaluate_unified('mrag-bench', predictions, references)

    print("\nMRAG-Bench评测结果:")
    for key, value in metrics.items():
        if 'accuracy' in key:
            print(f"  {key}: {value:.2f}%")


def main():
    """主测试函数"""
    print("统一评测系统测试")
    print("="*80)

    # 测试1: 数据集加载
    test_dataset_loading()

    # 测试2: 统一评测
    test_unified_evaluation()

    # 测试3: MRAG-Bench评测
    test_mragbench_evaluation()

    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)


if __name__ == '__main__':
    main()