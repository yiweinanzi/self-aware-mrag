#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估指标检查器
检查当前实验支持的所有评估指标
"""

import sys
import os
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

def check_evaluation_metrics():
    """检查当前实验支持的评估指标"""

    print("🔍 评估指标检查器")
    print("=" * 60)

    # 需要检查的指标列表
    required_metrics = [
        "EM (Exact Match)",
        "F1 Score",
        "VQA-Score",
        "Recall@5",
        "Faithfulness",
        "Attribution Precision",
        "Position Bias Score"
    ]

    # 检查VQA官方评测器
    print("📊 VQA官方评测器检查:")
    try:
        from flashrag.utils.vqa_evaluator import VQAEvaluator, evaluate_vqa_accuracy
        print("  ✅ VQA官方评测器: 可用")
        print("  ✅ VQA-Score: 支持 (基于官方标准)")
    except ImportError:
        print("  ❌ VQA官方评测器: 不可用")

    # 检查NLP基础指标
    print("\n📝 NLP基础指标检查:")
    try:
        from sklearn.metrics import f1_score, precision_score, recall_score
        print("  ✅ F1 Score: sklearn支持")
        print("  ✅ Precision: sklearn支持")
        print("  ✅ Recall: sklearn支持")
    except ImportError:
        print("  ❌ sklearn: 不可用")

    # 检查当前实验脚本
    print("\n🔧 当前实验脚本检查:")
    script_path = "/data0/home/zqwang/ACL/FlashRAG/experiments/run_unified_ablation.py"
    if os.path.exists(script_path):
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查每个指标
        implemented_metrics = []
        missing_metrics = []

        for metric in required_metrics:
            metric_key = metric.lower().replace(' ', '_').replace('-', '_')
            if metric_key in content.lower() or metric in content:
                implemented_metrics.append(metric)
            else:
                missing_metrics.append(metric)

        print(f"  ✅ 已实现指标: {len(implemented_metrics)}")
        for metric in implemented_metrics:
            print(f"    ✓ {metric}")

        print(f"\n  ❌ 缺失指标: {len(missing_metrics)}")
        for metric in missing_metrics:
            print(f"    × {metric}")

    # 检查Pipeline输出
    print("\n🚀 Pipeline输出检查:")
    try:
        from flashrag.pipeline.self_aware_pipeline_qwen3vl import SelfAwarePipelineQwen3VL
        print("  ✅ SelfAwarePipeline: 可用")

        # 检查是否支持详细评估
        pipeline_methods = dir(SelfAwarePipelineQwen3VL)
        eval_methods = [m for m in pipeline_methods if 'eval' in m.lower() or 'metric' in m.lower()]
        if eval_methods:
            print(f"  ✅ 评估方法: {eval_methods}")
        else:
            print("  ⚠️  未找到专用评估方法")

    except ImportError:
        print("  ❌ SelfAwarePipeline: 不可用")

    # 检查数据处理能力
    print("\n📈 数据处理能力检查:")
    data_capabilities = {
        "EM (Exact Match)": "字符串精确匹配",
        "F1 Score": "需要ground truth和预测文本的token化",
        "VQA-Score": "✅ 已集成VQA官方标准",
        "Recall@5": "需要检索结果排序",
        "Faithfulness": "需要答案与检索文档的对比",
        "Attribution Precision": "需要细粒度归因分析",
        "Position Bias Score": "需要位置统计信息"
    }

    for metric, description in data_capabilities.items():
        print(f"  {metric}: {description}")

    # 总结
    print("\n" + "=" * 60)
    print("📋 评估指标状态总结:")
    print(f"  总需求指标: {len(required_metrics)}")
    print(f"  当前支持: F1 Score, VQA-Score (部分)")
    print(f"  需要实现: EM, Recall@5, Faithfulness, Attribution Precision, Position Bias Score")
    print(f"  建议: 在全样本实验前实现完整的评估指标体系")

if __name__ == "__main__":
    check_evaluation_metrics()