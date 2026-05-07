#!/usr/bin/env python3
"""
为MRAG-Bench重新计算F1分数
使用专门的多选题F1计算逻辑
"""

import json
from mrag_bench_f1_calculator import MRAGBenchF1Calculator
import pandas as pd
from datetime import datetime


def load_mrag_results():
    """加载MRAG实验结果"""
    with open('/data0/home/zqwang/ACL/FlashRAG/experiments/results_mragbench_baseline/all_results_20251219_012819.json', 'r') as f:
        return json.load(f)


def extract_method_results(method_name: str, all_results: dict):
    """从all_results中提取特定方法的结果"""
    results = []
    samples = []

    # 获取方法的结果
    if method_name in all_results:
        method_results = all_results[method_name]

        # 处理结果格式
        for result in method_results:
            results.append({
                'answer': result.get('answer', ''),
                'retrieved_docs': result.get('retrieved_docs', [])
            })

            # 假设原始数据格式，需要从answer中提取
            samples.append({
                'answer_choice': result.get('golden_answer', [None])[0]
                                if isinstance(result.get('golden_answer'), list)
                                else result.get('golden_answer', '')
            })

    return results, samples


def main():
    """主函数"""
    print("="*80)
    print("MRAG-Bench F1分数重新计算")
    print("="*80)

    # 加载原始结果
    all_results = load_mrag_results()
    calculator = MRAGBenchF1Calculator()

    # 需要计算F1的方法（排除失败的）
    methods = ['Self-Aware-MRAG', 'SAM-RAG', 'mR2AG', 'VisRAG', 'ViDoRAG', 'MuRAG']

    # 存储新的F1分数
    new_f1_scores = {}
    comparison_data = []

    print("\n重新计算F1分数...")
    print("-"*80)

    for method in methods:
        print(f"\n处理方法: {method}")

        # 提取结果
        results, samples = extract_method_results(method, all_results)

        if not results or not samples:
            print(f"  ⚠️ 无法加载 {method} 的结果")
            continue

        # 计算不同的F1
        standard_f1 = calculator.calculate_standard_f1(results, samples)
        em_f1 = calculator.calculate_em_based_f1(results, samples)

        print(f"  标准F1: {standard_f1['f1']:.3f}")
        print(f"  EM-Based F1: {em_f1:.3f} (推荐)")

        # 保存结果
        new_f1_scores[method] = {
            'standard_f1': standard_f1['f1'],
            'em_based_f1': em_f1,
            'precision': standard_f1['precision'],
            'recall': standard_f1['recall'],
            'tp': standard_f1['tp'],
            'fp': standard_f1['fp'],
            'fn': standard_f1['fn']
        }

        comparison_data.append({
            'Method': method,
            'Original F1': '0.000',
            'Standard F1': f"{standard_f1['f1']:.3f}",
            'EM-Based F1': f"{em_f1:.3f}",
            'Precision': f"{standard_f1['precision']:.3f}",
            'Recall': f"{standard_f1['recall']:.3f}",
            'TP': standard_f1['tp'],
            'FP': standard_f1['fp'],
            'FN': standard_f1['fn']
        })

    # 创建对比表
    df = pd.DataFrame(comparison_data)
    print("\n\nF1计算对比表：")
    print("-"*80)
    print(df.to_string(index=False))

    # 保存新计算的F1分数
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'/data0/home/zqwang/ACL/FlashRAG/experiments/results_mragbench_baseline/mrag_f1_calculated_{timestamp}.json'

    with open(output_file, 'w') as f:
        json.dump({
            'date': timestamp,
            'dataset': 'MRAG-Bench',
            'calculator': 'MRAGBenchF1Calculator',
            'methods': new_f1_scores,
            'recommendation': 'Use EM-Based F1 for multiple-choice questions',
            'original_data': '/data0/home/zqwang/ACL/FlashRAG/experiments/results_mragbench_baseline/metrics_comparison_20251219_012819.json'
        }, f, indent=2)

    print(f"\n✅ 新计算的F1分数已保存到: {output_file}")

    # 创建更新后的指标文件（仅用于展示）
    print("\n\n创建展示用的更新指标文件...")
    with open('/data0/home/zqwang/ACL/FlashRAG/experiments/results_mragbench_baseline/metrics_comparison_20251219_012819.json', 'r') as f:
        original_metrics = json.load(f)

    # 创建展示版本（不修改原始文件）
    display_metrics = original_metrics.copy()
    for method, f1_data in new_f1_scores.items():
        if method in display_metrics:
            # 添加新的F1字段，保留原始值
            display_metrics[method]['original_f1'] = display_metrics[method].get('f1', 0)
            display_metrics[method]['em_based_f1'] = f1_data['em_based_f1']

    # 保存展示版本
    display_file = f'/data0/home/zqwang/ACL/FlashRAG/experiments/results_mragbench_baseline/metrics_display_with_f1_{timestamp}.json'
    with open(display_file, 'w') as f:
        json.dump(display_metrics, f, indent=2)

    print(f"✅ 展示用的更新指标已保存到: {display_file}")

    print("\n" + "="*80)
    print("总结：")
    print("1. 原始指标文件保持不变，不影响其他实验")
    print("2. 创建了新的F1计算，适合多选题")
    print("3. 推荐在论文中使用EM-Based F1")
    print("4. 可以解释为什么标准F1对于多选题不适用")


if __name__ == "__main__":
    main()