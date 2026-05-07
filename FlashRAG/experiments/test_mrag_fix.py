#!/usr/bin/env python3
"""测试MRAG修复方案"""

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from datasets import load_from_disk

# 模拟当前代码的数据加载
dataset_path = "/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MRAG-Bench/raw"
dataset_dict = load_from_disk(dataset_path)
test_data = dataset_dict['test']

# 当前的错误加载方式（缺少answer_choice）
print("=== 当前的错误加载方式 ===")
samples_wrong = []
for i, item in enumerate(test_data.select(range(3))):
    sample = {
        'question': item['question'],
        'answer': item['answer'],  # 文本
        'A': item['A'],
        'B': item['B'],
        'C': item['C'],
        'D': item['D'],
        # 注意：缺少answer_choice！
    }
    samples_wrong.append(sample)

# 计算指标
correct = 0
total = 0
for i, (result, sample) in enumerate(zip([
    {'answer': 'Yorkshire_terrier'},
    {'answer': 'capuchin'},
    {'answer': 'Chicago'}
], samples_wrong)):
    gt = sample.get('answer_choice', sample['answer']).upper()  # answer_choice不存在，使用answer文本
    print(f"Sample {i}: GT = {gt}, Pred = {result['answer']}")
    if gt and result['answer'] and gt in result['answer'].upper():
        correct += 1
    total += 1
print(f"准确率: {correct}/{total} = {correct/total*100:.1f}%")

print("\n=== 修复后的加载方式 ===")
samples_correct = []
for i, item in enumerate(test_data.select(range(3))):
    sample = {
        'question': item['question'],
        'answer': item['answer'],  # 文本
        'answer_choice': item['answer_choice'],  # 添加answer_choice
        'A': item['A'],
        'B': item['B'],
        'C': item['C'],
        'D': item['D'],
    }
    samples_correct.append(sample)

# 计算指标
correct = 0
total = 0
for i, (result, sample) in enumerate(zip([
    {'answer': 'Yorkshire_terrier'},
    {'answer': 'capuchin'},
    {'answer': 'Chicago'}
], samples_correct)):
    gt = sample.get('answer_choice', sample['answer']).upper()  # 现在answer_choice存在
    print(f"Sample {i}: GT = {gt}, Pred = {result['answer']}")
    if gt and result['answer'] and gt in result['answer'].upper():
        correct += 1
    total += 1
print(f"准确率: {correct}/{total} = {correct/total*100:.1f}%")