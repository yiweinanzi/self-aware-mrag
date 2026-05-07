#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试四个数据集的数据加载 - 每个数据集10个样本
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# 添加FlashRAG路径
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

# 测试函数
def test_okvqa_loading():
    """测试OK-VQA数据加载"""
    print("\n" + "="*80)
    print("1. 测试OK-VQA数据加载")
    print("="*80)

    try:
        # 导入OK-VQA加载函数
        from flashrag.dataset.okvqa_dataset import load_okvqa_dataset

        # 加载10个样本
        dataset = load_okvqa_dataset(max_samples=10)
        print(f"✅ 成功加载 {len(dataset)} 样本")

        # 打印第一个样本
        if dataset:
            print(f"   示例样本: {dataset[0]}")

        return True, len(dataset)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False, 0

def test_mrag_loading():
    """测试MRAG-Bench数据加载"""
    print("\n" + "="*80)
    print("2. 测试MRAG-Bench数据加载")
    print("="*80)

    try:
        import datasets

        # 尝试加载MRAG-Bench
        dataset_path = '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MRAG-Bench/raw'
        if os.path.exists(dataset_path):
            dataset_dict = datasets.load_from_disk(dataset_path)
            test_data = dataset_dict['test']
            samples = list(test_data)[:10]

            print(f"✅ 成功加载 {len(samples)} 样本")

            # 打印第一个样本
            if samples:
                print(f"   示例样本 keys: {list(samples[0].keys())}")
                print(f"   问题: {samples[0].get('question', '')[:100]}...")

            return True, len(samples)
        else:
            print(f"⚠️ 数据路径不存在: {dataset_path}")
            return False, 0
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False, 0

def test_multimodalqa_loading():
    """测试MultiModalQA数据加载"""
    print("\n" + "="*80)
    print("3. 测试MultiModalQA数据加载")
    print("="*80)

    try:
        import gzip
        import json

        dataset_path = '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MultiModalQA'
        dev_file = os.path.join(dataset_path, 'MMQA_dev.jsonl.gz')

        if os.path.exists(dev_file):
            samples = []
            with gzip.open(dev_file, 'rt', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 10:
                        break
                    item = json.loads(line.strip())
                    samples.append(item)

            print(f"✅ 成功加载 {len(samples)} 样本")

            # 打印第一个样本
            if samples:
                print(f"   示例样本 keys: {list(samples[0].keys())}")
                print(f"   问题: {samples[0].get('question', '')[:100]}...")
                print(f"   问题类型: {samples[0].get('metadata', {}).get('type', 'Unknown')}")

            return True, len(samples)
        else:
            print(f"⚠️ 数据文件不存在: {dev_file}")
            return False, 0
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False, 0

def test_aokvqa_loading():
    """测试A-OKVQA数据加载"""
    print("\n" + "="*80)
    print("4. 测试A-OKVQA数据加载")
    print("="*80)

    try:
        # 尝试多个可能的文件
        possible_files = [
            '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/A-OKVQA/validation_sample.json',
            '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/A-OKVQA/train_sample.json',
            '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/A-OKVQA/test_sample.json'
        ]

        for file_path in possible_files:
            if os.path.exists(file_path):
                print(f"使用文件: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                samples = data[:10]
                print(f"✅ 成功加载 {len(samples)} 样本")

                # 打印第一个样本
                if samples:
                    print(f"   示例样本 keys: {list(samples[0].keys())}")
                    print(f"   问题: {samples[0].get('question', '')[:100]}...")
                    print(f"   选项数: {len(samples[0].get('choices', []))}")

                return True, len(samples)

        print("⚠️ 没有找到A-OKVQA数据文件")
        return False, 0
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False, 0

def main():
    """主函数"""
    print("="*80)
    print("测试四个数据集的数据加载（每个10个样本）")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    # 测试每个数据集
    tests = [
        ('OK-VQA', test_okvqa_loading),
        ('MRAG-Bench', test_mrag_loading),
        ('MultiModalQA', test_multimodalqa_loading),
        ('A-OKVQA', test_aokvqa_loading),
    ]

    for name, test_func in tests:
        start_time = time.time()
        success, count = test_func()
        elapsed = time.time() - start_time

        results[name] = {
            'success': success,
            'count': count,
            'time': elapsed
        }

    # 生成报告
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)

    for name, result in results.items():
        status = "✅ 成功" if result['success'] else "❌ 失败"
        print(f"{name}: {status} - {result['count']} 样本 (耗时: {result['time']:.2f}s)")

    # 保存详细报告
    output_dir = Path('/data0/home/zqwang/ACL/FlashRAG/test_results_4datasets')
    output_dir.mkdir(exist_ok=True)

    report_file = output_dir / f'loading_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 四个数据集加载测试报告\n\n")
        f.write(f"**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for name, result in results.items():
            f.write(f"## {name}\n")
            f.write(f"- 状态: {'✅ 成功' if result['success'] else '❌ 失败'}\n")
            f.write(f"- 样本数: {result['count']}\n")
            f.write(f"- 耗时: {result['time']:.2f}s\n\n")

    print(f"\n详细报告保存在: {report_file}")

if __name__ == '__main__':
    main()