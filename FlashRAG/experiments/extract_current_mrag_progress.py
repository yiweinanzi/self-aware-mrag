#!/usr/bin/env python3
"""提取MRAG实验当前进度"""

import re

with open('mrag_fixed_475.out', 'r') as f:
    content = f.read()

# 定义方法顺序
methods = ['Self-Aware-MRAG', 'SAM-RAG', 'MuRAG', 'VisRAG', 'ViDoRAG', 'RagVL', 'mR2AG']

print("MRAG-Bench 实验进度（Job 475）")
print("="*60)
print(f"运行时间: 24分钟")
print("-"*60)

# 提取准确率
accuracies = []
pattern = r"✅ (MRAG-Bench Overall Accuracy: (\d+\.\d+)%)"
for match in re.finditer(pattern, content):
    accuracy = match.group(2)
    accuracies.append(float(accuracy))

# 检查各方法是否完成
for i, method in enumerate(methods):
    if i < len(accuracies):
        print(f"✅ {method:<20} {accuracies[i]:>6.00%}")
    else:
        if method == 'ViDoRAG':
            # 检查ViDoRAG进度
            progress_match = re.search(r"运行 ViDoRAG:\s*(\d+)%", content)
            if progress_match:
                progress = progress_match.group(1)
                print(f"⏳ {method:<20} 运行中 ({progress}%)")
            else:
                print(f"⏳ {method:<20} 等待中")
        else:
            print(f"⏳ {method:<20} 等待中")

print("-"*60)

# ���计
completed = len(accuracies) // 2  # 每个方法会输出两次
print(f"已完成: {completed}/7 方法")
print(f"剩余时间估计: 约{(7-completed)*3}分钟")

print("\n当前结果（前10个样本）:")
print("-"*60)
if accuracies:
    print(f"Self-Aware-MRAG: {accuracies[0]:.0f}% (阈值0.35)")
    if len(accuracies) >= 4:
        print(f"SAM-RAG:         {accuracies[2]:.0f}%")
    if len(accuracies) >= 6:
        print(f"MuRAG:           {accuracies[4]:.0f}%")
    if len(accuracies) >= 8:
        print(f"VisRAG:          {accuracies[6]:.0f}%")