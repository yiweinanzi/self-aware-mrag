#!/usr/bin/env python3
"""测试本地MRAG-Bench数据"""

import json
from datetime import datetime

print("="*80)
print("测试本地MRAG-Bench数据")
print("="*80)

# 读取本地数据
print("\n1. 加载本地数据...")
with open('/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MRAG-Bench/test.json', 'r') as f:
    data = json.load(f)
print(f"   加载了 {len(data)} 个样本")

# 检查数据结构
sample = data[0]
print("\n2. 检��数据结构...")
print(f"   样本字段: {list(sample.keys())}")
print(f"   是否有gt_images: {'gt_images' in sample}")
if 'gt_images' in sample:
    print(f"   gt_images数量: {len(sample['gt_images'])}")
else:
    print("   gt_images: None")

# 检查场景
scenarios = {}
for item in data:
    scenario = item.get('scenario', 'Unknown')
    scenarios[scenario] = scenarios.get(scenario, 0) + 1

print("\n3. 场景分布:")
for scenario, count in sorted(scenarios.items()):
    print(f"   {scenario}: {count} 样本")

# 打印第一个样本的详细信息
print("\n4. 第一个样本详情:")
print(f"   ID: {sample['id']}")
print(f"   Question: {sample['question'][:100]}...")
print(f"   Answer: {sample['answer']}")
print(f"   Answer Choice: {sample['answer_choice']}")
print(f"   Scenario: {sample.get('scenario', 'Unknown')}")

print("\n" + "="*80)
print("测试完成")
print("="*80)