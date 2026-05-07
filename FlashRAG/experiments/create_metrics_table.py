#!/usr/bin/env python3
"""创建MRAG实验指标表格"""

import json
import pandas as pd

# 读取指标文件
with open('/data0/home/zqwang/ACL/FlashRAG/experiments/results_mragbench_baseline/metrics_comparison_20251219_012819.json', 'r') as f:
    metrics = json.load(f)

# 创建表格数据
data = []
for method, m in metrics.items():
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

# 按准确率排序
df = pd.DataFrame(data)
df = df.sort_values('Accuracy', ascending=False)

# 打印表格
print("="*100)
print("MRAG-Bench 基线方法对比实验结果（前10个样本）")
print("="*100)
print(df.to_string(index=False))
print("="*100)

# 分析低准确率方法的问题
print("\n\n低准确率方法指标分析：")
print("-"*80)
print("\n1. SAM-RAG (20%准确率):")
print(f"   - Retrieval Rate: 80% (检索率较高，但准确率低)")
print(f"   - Faithfulness: 0.789 (忠诚度最高，可能过于依赖检索文档)")
print(f"   - Attribution Precision: 0.725 (归因精度较高)")
print("   问题：虽然检索并利用了文档，但结果仍然错误，可能是检索质量不高")

print("\n2. ViDoRAG (30%准确率):")
print(f"   - Retrieval Rate: 100% (总是检索)")
print(f"   - Faithfulness: 0.400 (中等)")
print(f"   - Attribution Precision: 0.200 (较低)")
print("   问题：视频RAG用于静态图片，时序建模没有帮助")

print("\n3. VisRAG (10%准确率):")
print(f"   - Retrieval Rate: 100% (总是检索)")
print(f"   - Faithfulness: 0.039 (极低，几乎不利用检索内容)")
print(f"   - Attribution Precision: 0.000 (完全不归因)")
print("   问题：纯视觉RAG无法获取外部知识，但检索的内容又不被使用")

print("\n\n分场景准确率分析：")
print("-"*80)
for method in ['SAM-RAG', 'ViDoRAG', 'VisRAG']:
    if method in metrics:
        print(f"\n{method}:")
        scenarios = metrics[method]['scenario_accuracy']
        for scenario, acc in scenarios.items():
            if acc > 0:
                print(f"   - {scenario}: {acc:.1f}%")
            else:
                print(f"   - {scenario}: 0% (完全失败)")

print("\n\n关键发现：")
print("-"*80)
print("1. 所有方法的F1分数都是0，说明评估指标可能有问题")
print("2. VisRAG的忠诚度极低(0.039)，说明它检索但不使用文档")
print("3. SAM-RAG忠诚度最高(0.789)但准确率低，可能是检索质量差")
print("4. 大部分方法在Scope场景都失败(0%)，需要外部知识")