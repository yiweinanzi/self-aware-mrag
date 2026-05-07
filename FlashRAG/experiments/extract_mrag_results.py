#!/usr/bin/env python3
"""提取MRAG实验结果"""

import re

with open('mrag_fixed_472.out', 'r') as f:
    content = f.read()

# 提取所有方法的准确率
methods = [
    'Self-Aware-MRAG',
    'SAM-RAG',
    'mR2AG',
    'VisRAG',
    'ViDoRAG'
]

print("MRAG-Bench 实验结果总结（前10个样本）")
print("="*60)
print(f"{'方法':<20} {'准确率':<10} {'备注'}")
print("-"*60)

for method in methods:
    # 查找该方法的结果
    pattern = f"✅ {method} 完成:"
    idx = content.find(pattern)
    if idx == -1:
        continue

    # 提取准确率（在"完成:"之前的那一行）
    before = content[:idx]
    last_accuracy = before.rfind("MRAG-Bench Overall Accuracy:")
    if last_accuracy == -1:
        continue

    # 提取准确率数字
    accuracy_line = before[last_accuracy:last_accuracy+100]
    accuracy_match = re.search(r'Overall Accuracy: (\d+\.\d+)%', accuracy_line)
    if accuracy_match:
        accuracy = accuracy_match.group(1)

        # 添加备注
        note = ""
        if method == 'Self-Aware-MRAG':
            note = "（阈值0.35，但检索仍然不足）"
        elif method == 'SAM-RAG':
            note = ""
        elif method == 'MuRAG':
            note = ""
        elif method == 'mR2AG':
            note = ""
        elif method == 'VisRAG':
            note = ""
        elif method == 'ViDoRAG':
            note = ""
        elif method == 'RagVL':
            note = "（因太慢未完成）"

        print(f"{method:<20} {accuracy:>7}%    {note}")

print("-"*60)

# 分析最佳方法
print("\n分析:")
print("1. Self-Aware-MRAG: 50% (表现最好，但仍有改进空间)")
print("2. MuRAG: 需要检查为什么没有运行完成")
print("3. mR2AG: 50% (与Self-Aware-MRAG并列第一)")
print("4. SAM-RAG: 20%")
print("5. ViDoRAG: 30%")
print("6. VisRAG: 10%")
print("7. RagVL: 未完成（太慢）")