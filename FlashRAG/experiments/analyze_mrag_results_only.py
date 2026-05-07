#!/usr/bin/env python3
"""
MRAG-Bench结果分析脚本（只用于分析，不修改原始数据）

注意：
- 原始F1分数保持不变
- 仅在展示时将F1显示为EM值（因为多选题F1应等于EM）
- 不影响其他数据集的对比实验
"""

import json
import pandas as pd

# 读取原始指标（保持不变）
with open('/data0/home/zqwang/ACL/FlashRAG/experiments/results_mragbench_baseline/metrics_comparison_20251219_012819.json', 'r') as f:
    original_metrics = json.load(f)

# 创建展示数据（仅用于可视化）
display_data = []
annotations = []

for method, m in original_metrics.items():
    # 原始数据
    display_data.append({
        'Method': method,
        'Original Accuracy': f"{m['accuracy']:.1%}",
        'Original EM': f"{m['em']:.3f}",
        'Original F1': f"{m['f1']:.3f}",  # 保持原始值
        'Retrieval Rate': f"{m['retrieval_rate']:.1%}",
        'Recall@5': f"{m['retrieval_recall_top5']:.1%}",
        'Faithfulness': f"{m['faithfulness']:.3f}",
        'Attribution Precision': f"{m['attribution_precision']:.3f}",
        'Time (sec/sample)': f"{m['seconds_per_sample']:.1f}"
    })

    # 说明
    if m['f1'] == 0:
        annotations.append({
            'Method': method,
            'Note': 'F1=0 (原始值) - 多选题任务中F1通常应等于EM'
        })

print("="*100)
print("MRAG-Bench 基线方法对比实验（展示用，原始数据未修改）")
print("="*100)
print("\n原始数据（保持不变）：")
print("-"*100)

df_original = pd.DataFrame(display_data)
df_original = df_original.sort_values('Original Accuracy', ascending=False)
print(df_original.to_string(index=False))

print("\n" + "="*100)
print("📝 说明：")
print("1. 原始F1分数保持为0（未修改）")
print("2. 对于多选题任务，理论上F1应该等于EM")
print("3. 这个F1=0的问题可能需要:")
print("   - 检查评估器的F1计算方式")
print("   - 或调整多选题的F1计算逻辑")
print("4. 不影响其他数据集（如OK-VQA）的F1计算")

# 保存分析报告
report = {
    'date': '2025-12-19',
    'dataset': 'MRAG-Bench',
    'samples': 10,
    'issue': 'F1 scores are all 0',
    'analysis': 'For multiple-choice questions, F1 should equal Exact Match',
    'recommendation': 'Investigate the F1 calculation in evaluate_comprehensive_metrics',
    'methods': original_metrics,
    'note': 'Original data preserved, no modifications made'
}

with open('/data0/home/zqwang/ACL/FlashRAG/experiments/results_mragbench_baseline/mrag_f1_analysis.json', 'w') as f:
    json.dump(report, f, indent=2)

print("\n✅ 分析报告已保存到: mrag_f1_analysis.json")
print("\n🚀 建议：")
print("- 保持原始数据不变，用于与其他数据集对比")
print("- 专门为MRAG-Bench开发合适的F1计算逻辑")
print("- 或者在论文中解释为什么多选题的F1等于EM")