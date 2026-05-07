#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强评估指标集成
为run_unified_ablation.py添加完整的评估指标
"""

import sys
import os
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from flashrag.utils.comprehensive_evaluator import evaluate_comprehensive_metrics


def enhance_evaluation_stats(base_stats: dict, results: list) -> dict:
    """为实验结果添加完整的评估指标"""

    print("📊 计算完整评估指标...")

    # 计算所有评估指标
    comprehensive_metrics = evaluate_comprehensive_metrics(results)

    # 转换为更友好的格式
    enhanced_stats = base_stats.copy()

    # 核心指标 (准确率和检索率已经在base_stats中)
    enhanced_stats['F1'] = comprehensive_metrics.get('avg_F1', 0.0)
    enhanced_stats['VQA_Score'] = comprehensive_metrics.get('avg_VQA_Score', 0.0)

    # RAG相关指标
    enhanced_stats['Recall@5'] = comprehensive_metrics.get('avg_Recall@5', 0.0)
    enhanced_stats['Faithfulness'] = comprehensive_metrics.get('avg_Faithfulness', 0.0)
    enhanced_stats['Attribution_Precision'] = comprehensive_metrics.get('avg_Attribution_Precision', 0.0)
    enhanced_stats['Position_Bias_Score'] = comprehensive_metrics.get('avg_Position_Bias_Score', 0.0)

    # 标准差信息
    enhanced_stats['F1_std'] = comprehensive_metrics.get('std_F1', 0.0)
    enhanced_stats['VQA_Score_std'] = comprehensive_metrics.get('std_VQA_Score', 0.0)
    enhanced_stats['Recall@5_std'] = comprehensive_metrics.get('std_Recall@5', 0.0)

    # 统计信息
    enhanced_stats['total_samples'] = comprehensive_metrics.get('total_samples', 0)
    enhanced_stats['retrieved_samples'] = comprehensive_metrics.get('retrieved_samples', 0)

    # 打印增强指标
    print(f"\n📈 完整评估指标:")
    print(f"   准确率: {enhanced_stats['accuracy']*100:.2f}%")
    print(f"   检索率: {enhanced_stats['retrieval_rate']*100:.2f}%")
    print(f"   F1: {enhanced_stats['F1']*100:.2f}% ± {enhanced_stats['F1_std']*100:.2f}%")
    print(f"   VQA-Score: {enhanced_stats['VQA_Score']*100:.2f}% ± {enhanced_stats['VQA_Score_std']*100:.2f}%")
    print(f"   Recall@5: {enhanced_stats['Recall@5']*100:.2f}% ± {enhanced_stats['Recall@5_std']*100:.2f}%")
    print(f"   Faithfulness: {enhanced_stats['Faithfulness']*100:.2f}%")
    print(f"   Attribution Precision: {enhanced_stats['Attribution_Precision']*100:.2f}%")
    print(f"   Position Bias Score: {enhanced_stats['Position_Bias_Score']*100:.2f}%")

    return enhanced_stats


def enhance_results_saving(save_data: dict, all_results: list) -> dict:
    """增强结果保存，包含完整评估指标"""

    print("💾 增强结果保存...")

    # 为每个变体添加完整指标
    for i, variant_result in enumerate(all_results):
        if variant_result is None:
            continue

        variant_name = variant_result['stats']['variant_name']
        results = variant_result['results']

        if results:
            comprehensive_metrics = evaluate_comprehensive_metrics(results)

            # 更新统计信息
            enhanced_stats = enhance_evaluation_stats(variant_result['stats'], results)
            variant_result['stats'] = enhanced_stats

            # 更新保存数据中的统计
            for summary_item in save_data['variants_summary']:
                if summary_item['variant_name'] == variant_name:
                    summary_item.update(enhanced_stats)
                    break

    return save_data


def generate_enhanced_report(all_results: list, save_data: dict) -> str:
    """生成包含完整评估指标的实验报告"""

    report_lines = [
        "# 综合评估指标消融实验报告",
        "",
        "## 实验概览",
        f"- **数据集**: {save_data['experiment_info']['dataset']}",
        f"- **样本数**: {save_data['experiment_info']['samples']}",
        f"- **时间戳**: {save_data['experiment_info']['timestamp']}",
        "",
        "## 评估指标说明",
        "- **EM (Exact Match)**: 精确匹配率",
        "- **F1 Score**: F1分数（基于token级别）",
        "- **VQA-Score**: VQA官方评测得分",
        "- **Recall@5**: 检索召回率@5",
        "- **Faithfulness**: 答案与检索文档一致性",
        "- **Attribution Precision**: 答案归因精确度",
        "- **Position Bias Score**: 位置偏差得分",
        "",
        "## 实验结果对比",
        "",
        "| 变体 | EM | F1 | VQA-Score | Recall@5 | Faithfulness | Attribution | Position Bias |",
        "|------|----|----|-----------|----------|-------------|--------------|---------------|",
    ]

    # 添加每个变体的结果
    for summary in save_data['variants_summary']:
        variant_name = summary['variant_name'].replace('_', ' ')
        em = summary.get('EM', 0.0) * 100
        f1 = summary.get('F1', 0.0) * 100
        vqa = summary.get('VQA_Score', 0.0) * 100
        recall5 = summary.get('Recall@5', 0.0) * 100
        faithful = summary.get('Faithfulness', 0.0) * 100
        attr_prec = summary.get('Attribution_Precision', 0.0) * 100
        pos_bias = summary.get('Position_Bias_Score', 0.0) * 100

        report_lines.append(
            f"| {variant_name} | {em:.1f}% | {f1:.1f}% | {vqa:.1f}% | "
            f"{recall5:.1f}% | {faithful:.1f}% | {attr_prec:.1f}% | {pos_bias:.3f} |"
        )

    report_lines.extend([
        "",
        "## 详细指标分析",
        ""
    ])

    # 为每个指标添加详细分析
    metrics_analysis = [
        ("EM", "Exact Match", "最严格的匹配标准，要求答案完全一致"),
        ("F1", "F1 Score", "平衡精确率和召回率的综合指标"),
        ("VQA-Score", "VQA官方评测", "基于VQA官方标准的评测得分"),
        ("Recall@5", "检索召回率", "前5个检索结果中包含正确答案的比例"),
        ("Faithfulness", "忠实度", "答案内容与检索文档的一致性"),
        ("Attribution Precision", "归因精确度", "答案内容可追溯到检索文档的比例"),
        ("Position Bias Score", "位置偏差", "检索结果位置分布的合理性")
    ]

    for metric_key, metric_name, description in metrics_analysis:
        values = []
        for summary in save_data['variants_summary']:
            if metric_key in summary:
                values.append(summary[metric_key])

        if values:
            avg_val = np.mean(values) * 100
            std_val = np.std(values) * 100
            report_lines.append(f"### {metric_name} ({metric_key})")
            report_lines.append(f"- **说明**: {description}")
            report_lines.append(f"- **平均值**: {avg_val:.2f}% ± {std_val:.2f}%")
            report_lines.append(f"- **最佳值**: {max(values)*100:.2f}%")
            report_lines.append(f"- **最低值**: {min(values)*100:.2f}%")
            report_lines.append("")

    return "\n".join(report_lines)


def test_comprehensive_evaluation():
    """测试综合评估指标功能"""
    print("🧪 测试综合评估指标...")

    # 创建模拟测试数据
    test_results = [
        {
            'answer': 'motorcycle racing',
            'golden_answers': ['motorcycle racing', 'racing', 'motorcycle sport'],
            'retrieved_docs': [
                {'contents': 'Motorcycle racing is a sport...', 'score': 0.9},
                {'contents': 'Racing motorcycles...', 'score': 0.8},
            ],
            'retrieved': True
        },
        {
            'answer': 'baseball',
            'golden_answers': ['baseball', 'sport', 'game'],
            'retrieved_docs': [
                {'contents': 'Baseball is a popular sport...', 'score': 0.7},
            ],
            'retrieved': True
        }
    ]

    # 计算指标
    metrics = evaluate_comprehensive_metrics(test_results)

    print("✅ 测试结果:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value*100:.2f}%")
        else:
            print(f"  {key}: {value}")

    return True


if __name__ == "__main__":
    import numpy as np
    test_comprehensive_evaluation()