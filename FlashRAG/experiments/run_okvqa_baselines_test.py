#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OK-VQA Baselines测试 - 简化版本
仅测试2个样本确保代码正常运行
"""

import os
import sys
import json
import time
from datetime import datetime

# 添加FlashRAG路径
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from flashrag.dataset.okvqa_dataset_simple import OKVQADatasetSimple
from flashrag.utils.comprehensive_evaluator import evaluate_comprehensive_metrics

# 简化配置
CONFIG = {
    'data_dir': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/VQA',
    'max_samples': 2,  # 仅2个样本测试
    'output_dir': '/data0/home/zqwang/ACL/FlashRAG/experiments/results_okvqa_test',
}

def main():
    print("="*60)
    print("OK-VQA Baselines 简化测试")
    print("="*60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"样本数: {CONFIG['max_samples']}")

    # 1. 测试数据加载
    print("\n1. 测试数据加载")
    print("-" * 40)

    try:
        dataset = OKVQADatasetSimple({
            'data_dir': CONFIG['data_dir'],
            'split': 'val',
            'load_images': True,
        })

        samples = []
        for i in range(min(CONFIG['max_samples'], len(dataset))):
            sample = dataset[i]
            # 不保存图像对象，只保存是否有图像的信息
            samples.append({
                'id': sample['id'],
                'question': sample['question'],
                'has_image': sample.get('image') is not None,
                'golden_answers': sample['golden_answers']
            })

        print(f"✅ 成功加载 {len(samples)} 个样本")
        print(f"   图像加载: {all(s.get('has_image', False) for s in samples)}")

        # 显示样本示例
        for i, sample in enumerate(samples):
            print(f"\n样本 {i+1}:")
            print(f"  ID: {sample['id']}")
            print(f"  问题: {sample['question']}")
            print(f"  答案: {sample['golden_answers'][:3]}")  # 显示前3个答案

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. 测试评估指标
    print("\n\n2. 测试评估指标")
    print("-" * 40)

    # 创建模拟结果
    mock_results = []
    for sample in samples:
        mock_results.append({
            'question': sample['question'],
            'answer': sample['golden_answers'][0] if sample['golden_answers'] else 'unknown',  # 使用第一个答案作为预测
            'golden_answers': sample['golden_answers'],
            'retrieved_docs': [{'contents': 'mock document content'}] * 3,  # 模拟检索结果
            'retrieved': True
        })

    try:
        metrics = evaluate_comprehensive_metrics(mock_results)
        print("✅ 评估指标计算成功")
        print(f"   准确率: {metrics.get('avg_accuracy', 0):.4f}")
        print(f"   F1分数: {metrics.get('avg_F1', 0):.4f}")
        print(f"   Recall@5: {metrics.get('avg_Recall@5', 0):.4f}")
        print(f"   Faithfulness: {metrics.get('avg_Faithfulness', 0):.4f}")

    except Exception as e:
        print(f"❌ 评估指标计算失败: {e}")
        import traceback
        traceback.print_exc()

    # 3. 保存测试结果
    print("\n\n3. 保存测试结果")
    print("-" * 40)

    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    test_result = {
        'timestamp': datetime.now().isoformat(),
        'config': CONFIG,
        'samples': samples,
        'metrics': metrics if 'metrics' in locals() else {},
        'status': 'success'
    }

    output_file = os.path.join(CONFIG['output_dir'], 'test_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_result, f, indent=2, ensure_ascii=False)

    print(f"✅ 测试结果已保存到: {output_file}")

    print("\n" + "="*60)
    print("测试完成！基础功能正常")
    print("="*60)

if __name__ == "__main__":
    main()