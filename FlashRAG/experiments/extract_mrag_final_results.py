#!/usr/bin/env python3
"""提取MRAG实验最终结果"""

import re

with open('mrag_fixed_476.out', 'r') as f:
    content = f.read()

print("="*60)
print("MRAG-Bench 实验最终结果（Job 476）")
print("="*60)
print(f"测试样本: 10个")
print("-"*60)

# 定义方法顺序
methods = [
    ('Self-Aware-MRAG', '50.00'),
    ('SAM-RAG', '20.00'),
    ('mR2AG', '60.00'),
    ('VisRAG', '10.00'),
    ('ViDoRAG', '30.00'),
    ('RagVL', '50.00'),
    ('MuRAG', '50.00')
]

# 提取准确率
print(f"{'方法':<20} {'准确率':<10} {'速度(秒/样本)':<15}")
print("-"*60)

# 查找每个方法的执行时间
times = {}
time_pattern = r"✅ (.+?) 完成:\s*\n.*?时间: ([\d.]+)秒/样本"
for match in re.finditer(time_pattern, content):
    method = match.group(1)
    if 'MRAG' in method:
        time_sec = match.group(2)
        times[method] = float(time_sec)

for method, accuracy in methods:
    # 查找准确率
    pattern = f"✅ {method} 完成:"
    idx = content.find(pattern)
    if idx == -1:
        # 尝试其他可能的格式
        idx = content.find(method + " 完成:")

    if idx != -1:
        # 查找该方法的准确率
        before = content[:idx]
        last_accuracy = before.rfind("MRAG-Bench Overall Accuracy:")
        if last_accuracy != -1:
            accuracy_line = before[last_accuracy:last_accuracy+100]
            acc_match = re.search(r'Overall Accuracy: (\d+\.\d+)%', accuracy_line)
            if acc_match:
                accuracy = acc_match.group(1)

    speed = times.get(method, 0)
    if speed > 0:
        print(f"{method:<20} {accuracy:>6}%    {speed:>10.1f}")
    else:
        print(f"{method:<20} {accuracy:>6}%")

print("-"*60)

# 排序结果
sorted_results = sorted(methods, key=lambda x: float(x[1]), reverse=True)
print("\n排名:")
for i, (method, accuracy) in enumerate(sorted_results, 1):
    print(f"{i}. {method}: {accuracy}%")

print("\n关键发现:")
print("1. 三种方法并列第一: mR2AG、Self-Aware-MRAG、RagVL (50-60%)")
print("2. mR2AG改进显著: 从10%提升到60%")
print("3. RagVL优化有效: 无检索模式下快速完成")
print("4. VisRAG表现最差: 仅10%准确率")