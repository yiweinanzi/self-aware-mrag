#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试图像加载功能
"""

import os
import sys

sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from flashrag.dataset.okvqa_dataset_simple import OKVQADatasetSimple

def main():
    print("="*80)
    print("测试OK-VQA图像加载")
    print("="*80)

    # 测试数据加载
    config = {
        'data_dir': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/VQA',
        'split': 'val',
        'load_images': True,
    }

    print("\n加载数据集...")
    dataset = OKVQADatasetSimple(config)

    print(f"\n数据集大小: {len(dataset)}")

    # 查看前3个样本
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\n样本 {i+1}:")
        print(f"  ID: {sample['id']}")
        print(f"  问题: {sample['question']}")
        print(f"  图像ID: {sample['image_id']}")
        print(f"  图像: {'已加载' if sample['image'] else '未加载'}")
        if sample['image']:
            print(f"  图像大小: {sample['image'].size}")
        print(f"  答案: {sample['golden_answers'][:3]}")

if __name__ == "__main__":
    main()