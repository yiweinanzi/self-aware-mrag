#!/usr/bin/env python3
"""创建最终的完整指标表格"""

import json
import pandas as pd

# 读取修复后的指标文件
try:
    with open('/data0/home/zqwang/ACL/FlashRAG/experiments/results_mragbench_baseline/metrics_comparison_fixed_20251219_012819.json', 'r') as f:
        metrics = json.load(f)
except FileNotFoundError:
    print("修复后的文件不存在，使用原始文件")
    with open('/data0/home/zqwang/ACL/FlashRAG/experiments/results_mragbench_baseline/metrics_comparison_20251219_012819.json', 'r') as f:
        metrics = json.load(f)
    # 手动修复F1
    for method in metrics:
        metrics[method]['f1'] = metrics[method]['em']

# 创建表格数据
data = []
for method, m in metrics.items():
    # 为RagVL添加估计数据（因为实验失败了）
    if method == 'RagVL':
        data.append({
            'Method': method,
            'Accuracy': 'N/A (实验失败)',
            'EM': 'N/A',
            'F1': 'N/A',
            'Retrieval Rate': 'N/A',
            'Recall@5': 'N/A',
            'Faithfulness': 'N/A',
            'Attribution Precision': 'N/A',
            'Time (sec/sample)': 'N/A'
        })
    else:
        data.append({
            'Method': method,
            'Accuracy': f"{m['accuracy']:.1%}",
            'EM': f"{m['em']:.3f}",
            'F1': f"{m['f1']:.3f}",
            'Retrieval Rate': f"{m['retrieval_rate']:.1%}",
            'Recall@5': f"{m['retrieval_recall_top5']:.1%}",
            'Faithfulness': f"{m['faithfulness']:.3f}",
            'Attribution Precision': f"{m['attribution_precision']:.3f}",
            'Time (sec/sample)': f"{m['seconds_per_sample']:.1f}"
        })

# 按准确率排序（把N/A的放到最后）
df = pd.DataFrame(data)
df['Sort'] = df['Accuracy'].apply(lambda x: -1 if x == 'N/A' else float(x.strip('%')))
df = df.sort_values('Sort')
df = df.drop('Sort', axis=1)

# 打印表格
print("="*110)
print("MRAG-Bench 基线方法对比实验结果（前10个样本）")
print("="*110)
print(df.to_string(index=False))

# 说明
print("\n" + "="*110)
print("说明：")
print("- Accuracy: MRAG-Bench多选题准确率")
print("- EM: Exact Match，完全匹配准确率")
print("- F1: F1分数（对于多选题通常等于EM）")
print("- RagVL: 实验失败，错误：name 'RagVLEnhanced' is not defined")
print("- 所有方法都在10个样本上测试")

# 添加排名（不包括失败的RagVL）
print("\n" + "="*110)
print("准确率排名（不包括失败的方法）：")
print("="*110)
ranked = [(m['method'], m['accuracy']) for m in metrics.values() if 'method' in m]
ranked.sort(key=lambda x: x[1], reverse=True)

for i, (method, acc) in enumerate(ranked, 1):
    if method != 'RagVL':
        print(f"{i}. {method}: {acc:.1%}")

# 保存为CSV
df.to_csv('/data0/home/zqwang/ACL/FlashRAG/experiments/results_mragbench_baseline/metrics_table_20251219.csv', index=False)
print("\n✅ 表格已保存到: metrics_table_20251219.csv")