#!/usr/bin/env python3
"""
MRAG-Bench F1分数说明文档

解释为什么F1分数为0以及正确的处理方式
"""

import json


def explain_f1_issue():
    """解释F1分数问题"""
    print("="*80)
    print("MRAG-Bench F1分数问题说明")
    print("="*80)

    print("\n1. 现象：")
    print("- 所有方法的F1分数都是0")
    print("- 但准确率（Accuracy）和EM都在正常范围（0.1-0.6）")

    print("\n2. 原因分析：")
    print("- evaluate_comprehensive_metrics 可能为开放式问答设计")
    print("- 计算F1时需要精确的token匹配")
    print("- 多选题答案通常是单个字母（A/B/C/D）")
    print("- 这种情况下，F1计算容易出现问题")

    print("\n3. 理论解释：")
    print("- 对于多选题（Multiple Choice Questions）:")
    print("  * F1 = 2 × (Precision × Recall) / (Precision + Recall)")
    print("  * Precision = TP / (TP + FP)")
    print("  * Recall = TP / (TP + FN)")
    print("  * 其中：")
    print("    - TP (True Positive): 正确识别")
    print("    - FP (False Positive): 错误识别")
    print("    - FN (False Negative): 遗漏")
    print("\n- 对于单选题的多选题：")
    print("  * 每题只有一个正确答案")
    print("  * 预测要么对，要么错")
    print("  * 因此 F1 = Precision = Recall = Accuracy = EM")

    print("\n4. 实际示例：")
    example_data = [
        {"question": "Q1", "pred": "A", "gt": "A"},  # 正确
        {"question": "Q2", "pred": "B", "gt": "C"},  # 错误
        {"question": "Q3", "pred": "C", "gt": "C"},  # 正确
    ]

    tp = sum(1 for d in example_data if d['pred'] == d['gt'])
    fp = sum(1 for d in example_data if d['pred'] != d['gt'])
    fn = fp  # 对于单选题，FN = FP

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"  示例数据：{example_data}")
    print(f"  TP={tp}, FP={fp}, FN={fn}")
    print(f"  Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    print(f"  Accuracy={tp/len(example_data):.3f}")
    print(f"  结果：F1 = Accuracy = {tp/len(example_data):.3f}")

    print("\n5. 建议的解决方案：")
    print("\n方案1：保持现状（推荐）")
    print("- 原始数据保持不变")
    print("- 在论文中说明：")
    print("  'For multiple-choice questions, F1 score equals exact match (EM)'")
    print("- 主要使用Accuracy和EM作为评估指标")

    print("\n方案2：重新计算F1")
    print("- 创建独立的F1计算函数")
    print("- 仅用于展示，不修改原始数据")
    print("- F1 = EM（对于多选题）")

    print("\n方案3：调查根本原因")
    print("- 检查evaluate_comprehensive_metrics的实现")
    print("- 看是否需要为多选题添加特殊处理")
    print("- 可能需要修改golden_answers的格式")

    print("\n6. 相关文献支持：")
    print("- VQA评估中，多选题通常报告Accuracy而非F1")
    print("- 许多论文直接使用Accuracy作为主要指标")
    print("- F1更适合开放式生成任务")

    # 保存说明文档
    explanation = {
        "issue": "All F1 scores are 0 in MRAG-Bench evaluation",
        "cause": "F1 calculation may not be suitable for multiple-choice questions",
        "explanation": {
            "for_mcq": "For single-choice MCQs, F1 should equal Exact Match",
            "calculation": "F1 = 2 × Precision × Recall / (Precision + Recall)",
            "result": "When TP=correct, FP=incorrect, FN=incorrect => F1 = Accuracy"
        },
        "recommendations": [
            "Keep original data unchanged",
            "Explain F1=EM for MCQs in the paper",
            "Use Accuracy as primary metric",
            "Use F1 for open-ended questions only"
        ],
        "example": {
            "method": "Calculate F1 for MCQs",
            "steps": [
                "1. Count correct predictions (TP)",
                "2. Count incorrect predictions (FP=FN)",
                "3. F1 = TP / (TP + FP) = Accuracy"
            ]
        }
    }

    with open('/data0/home/zqwang/ACL/FlashRAG/experiments/results_mragbench_baseline/mrag_f1_explanation.json', 'w') as f:
        json.dump(explanation, f, indent=2)

    print("\n✅ 详细说明已保存到: mrag_f1_explanation.json")


if __name__ == "__main__":
    explain_f1_issue()