#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
调试MRAG-Bench数据结构
"""

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from datasets import load_from_disk

# 1. 加载一个样本
dataset = load_from_disk('/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MRAG-Bench/raw')
test_data = dataset['test']
sample = test_data[0]

print("样本键：", list(sample.keys()))
print("\n样本详细信息：")
for key, value in sample.items():
    if isinstance(value, str) and len(value) > 100:
        print(f"- {key}: {value[:100]}...")
    else:
        print(f"- {key}: {value}")

# 查看第二个样本
print("\n\n第二个样本的选项：")
sample2 = test_data[1]
if 'options' in sample2:
    print(f"选项: {sample2['options']}")
    print(f"答案: {sample2['answer']}")