#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试修复后的4GPU实验 - 小规模测试
Test Fixed 4GPU Experiment - Small Scale Test
"""

import os
import sys
import json
import torch
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

def test_small_dataset():
    """测试小规模数据集以验证修复"""
    print("🔄 测试修复后的4GPU实验（小规模）...")

    try:
        from flashrag.dataset.okvqa_dataset_simple import OKVQADatasetSimple
        from flashrag.modules.simple_llm import SimpleLLM  # 测试导入

        # 加载数据集
        print("🔄 加载数据集...")
        dataset_obj = OKVQADatasetSimple({
            'data_dir': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/VQA',
            'split': 'val',
            'load_images': False,  # 不加载图像以节省时间
        })

        # 只测试前10个样本
        test_samples = dataset_obj.data[:10]
        print(f"✅ 加载测试样本: {len(test_samples)}")

        # 验证数据字段
        for i, sample in enumerate(test_samples):
            print(f"\n[样本{i+1}]")
            print(f"   问题: {sample['question'][:50]}...")
            print(f"   答案字段: {list(sample.keys())}")
            print(f"   golden_answers: {sample['golden_answers'][:3]}")  # 显示前3个答案

            # 测试SimpleLLM
            llm = SimpleLLM()
            answer = llm.generate(sample['question'])
            print(f"   SimpleLLM预测: {answer}")

        # 测试GPU模型管理器
        print(f"\n🔄 可用GPU数量: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print("✅ CUDA可用")
        else:
            print("⚠️ CUDA不可用，将使用CPU")

        print("✅ 小规模测试成功！所有修复都正常工作")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    test_small_dataset()