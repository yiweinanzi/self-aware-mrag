#!/usr/bin/env python3
"""
分析MultiModalQA实验结果，找出可以改进的地方
"""

import json
import sys
from pathlib import Path

def analyze_results(results_file):
    """分析实验结果"""
    with open(results_file, 'r') as f:
        data = json.load(f)

    # 分析每个方法的结果
    for method, results in data.items():
        print(f"\n{'='*80}")
        print(f"方法: {method}")
        print(f"{'='*80}")

        correct = 0
        total = len(results)

        for i, item in enumerate(results):
            question = item.get('question', '')
            answer = item.get('answer', '')
            ground_truth = item.get('ground_truth', '')

            # 处理可能有多个答案的情况
            if isinstance(ground_truth, list):
                ground_truth = ground_truth[0] if ground_truth else ''

            is_correct = answer.strip().lower() == ground_truth.strip().lower()
            if is_correct:
                correct += 1

            print(f"\n样本 {i+1}:")
            print(f"问题: {question}")
            print(f"预测: {answer}")
            print(f"真实: {ground_truth}")
            print(f"{'✓ 正确' if is_correct else '✗ 错误'}")

            # 如果有不确定性信息，打印出来
            if 'uncertainty' in item:
                unc = item['uncertainty']
                print(f"不确定性: total={unc.get('total', 0):.3f}, "
                      f"text={unc.get('text', 0):.3f}, "
                      f"visual={unc.get('visual', 0):.3f}")

        accuracy = correct / total
        print(f"\n{'-'*80}")
        print(f"准确率: {correct}/{total} = {accuracy:.2%}")
        print(f"{'-'*80}")

def compare_methods(results_file):
    """比较不同方法的性能"""
    with open(results_file, 'r') as f:
        data = json.load(f)

    print(f"\n{'='*80}")
    print("方法对比")
    print(f"{'='*80}")

    # 找出所有样本的问题和答案
    all_questions = []
    method_results = {}

    for method, results in data.items():
        method_results[method] = {}
        for i, item in enumerate(results):
            q = item.get('question', '')
            ans = item.get('answer', '')
            gt = item.get('ground_truth', '')
            if isinstance(gt, list):
                gt = gt[0] if gt else ''

            if q not in all_questions:
                all_questions.append(q)

            method_results[method][q] = {
                'answer': ans,
                'ground_truth': gt,
                'correct': ans.strip().lower() == gt.strip().lower()
            }

    # 对每个问题，找出哪些方法答对了
    for i, q in enumerate(all_questions):
        print(f"\n问题 {i+1}: {q}")
        print("-" * 80)

        correct_methods = []
        for method in method_results:
            if method_results[method][q]['correct']:
                correct_methods.append(method)

        if correct_methods:
            print(f"✓ 答对的方法: {', '.join(correct_methods)}")

        # 打印所有方法的答案
        for method in method_results:
            result = method_results[method][q]
            status = "✓" if result['correct'] else "✗"
            print(f"  {status:2} {method}: {result['answer']} (正确: {result['ground_truth']})")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_results.py <results_file.json>")
        sys.exit(1)

    results_file = sys.argv[1]

    # 分析每个方法
    analyze_results(results_file)

    # 比较方法
    compare_methods(results_file)