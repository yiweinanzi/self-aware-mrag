#!/usr/bin/env python3
"""检查MRAG-Bench数据加载"""

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from datasets import load_from_disk

# 从实验使用的路径加载
dataset_path = "/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MRAG-Bench/raw"
dataset = load_from_disk(dataset_path)

print(f"Dataset info: {dataset}")
print(f"Splits: {list(dataset.keys())}")

test_data = dataset['test']
print(f"Test data length: {len(test_data)}")

# 检查前10个样本
for i in range(10):
    sample = test_data[i]
    print(f"\nSample {i}:")
    print(f"  Keys: {list(sample.keys())}")
    print(f"  answer_choice: {repr(sample.get('answer_choice', 'NOT_FOUND'))}")
    print(f"  answer: {repr(sample.get('answer', 'NOT_FOUND'))}")

    # 检查是否answer_choice字段包含的是文本而非字母
    ac = sample.get('answer_choice', '')
    if ac not in ['A', 'B', 'C', 'D']:
        print(f"  ERROR: answer_choice '{ac}' is not a single letter!")
        # 检查是否答案字段与某个选项匹配
        for choice in ['A', 'B', 'C', 'D']:
            if choice in sample and sample[choice] == ac:
                print(f"    Found match: answer_choice matches option {choice}")