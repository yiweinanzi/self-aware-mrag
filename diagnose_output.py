#!/usr/bin/env python3
"""
诊断准确率为0的方法的实际输出
"""

import sys
import os
sys.path.insert(0, '/data0/home/zqwang/ACL')
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

import json
from datetime import datetime

def main():
    print("🔍 诊断准确率为0的方法的实际输出")
    print("="*60)

    # 加载OK-VQA数据集样本
    dataset_path = '/root/autodl-fs/dataset/okvqa_2024'
    if os.path.exists(dataset_path):
        # 读取一些样本数据
        import json
        with open(os.path.join(dataset_path, 'annotations', 'okvqa_test.json'), 'r') as f:
            data = json.load(f)

        print("\n📊 OK-VQA数据集样本示例:")
        for i in range(min(3, len(data))):
            sample = data[i]
            print(f"\n样本 {i+1}:")
            print(f"  问题: {sample.get('question', 'N/A')}")
            print(f"  图像ID: {sample.get('image_id', 'N/A')}")
            print(f"  正确答案: {sample.get('answers', [])}")

    # 模拟测试MuRAG的输出
    print("\n" + "="*60)
    print("🧪 模拟测试答案提取和评估")

    from flashrag.utils.vqa_evaluator import extract_okvqa_answer, VQAEvaluator
    evaluator = VQAEvaluator()

    # 测试不同类型的答案
    test_cases = [
        # 预期的正确答案格式
        ("race", ["race", "race", "race"]),
        ("basketball", ["basketball", "basketball", "basketball"]),
        ("rose", ["rose", "rose", "rose"]),

        # 可能的错误答案格式
        ("The sport is basketball racing", ["race", "race", "race"]),
        ("This is a rose flower", ["rose", "rose", "rose"]),
        ("I think this is used for basketball", ["basketball", "basketball", "basketball"]),

        # 空答案或错误答案
        ("", ["race", "race", "race"]),
        ("I don't know", ["race", "race", "race"]),
        ("Unable to answer", ["race", "race", "race"]),
    ]

    print("\n测试答案提取:")
    for i, (answer, golden) in enumerate(test_cases):
        # 提取答案
        extracted = extract_okvqa_answer(answer)

        # 评估
        correct = evaluator.evaluate_answer(answer, golden) if hasattr(evaluator, 'evaluate_answer') else False

        # 标准化并检查
        standardized = evaluator.standardize_answer(extracted) if hasattr(evaluator, 'standardize_answer') else extracted
        matches = any(standardized == evaluator.standardize_answer(g) if hasattr(evaluator, 'standardize_answer') else standardized == g for g in golden)

        print(f"\n测试 {i+1}:")
        print(f"  原始答案: '{answer}'")
        print(f"  提取后: '{extracted}'")
        print(f"  标准化后: '{standardized}'")
        print(f"  正确答案: {golden}")
        print(f"  匹配: {'✅' if matches else '❌'}")

    # 检查实际输出文件
    print("\n" + "="*60)
    print("📁 查找实际输出文件")

    # 查找最近的结果文件
    import glob
    result_files = []

    # 搜索可能的结果文件
    search_patterns = [
        '/data0/home/zqwang/ACL/FlashRAG/results_*/*.json',
        '/data0/home/zqwang/ACL/FlashRAG/results_*/*_results.json',
        '/data0/home/zqwang/ACL/FlashRAG/*results*/**/*.json',
        '/data0/home/zqwang/ACL/**/results_*.json',
    ]

    for pattern in search_patterns:
        files = glob.glob(pattern, recursive=True)
        result_files.extend(files)

    if result_files:
        print(f"\n找到 {len(result_files)} 个结果文件:")
        for f in result_files[:10]:  # 只显示前10个
            print(f"  - {f}")

        # 尝试读取最新的结果文件
        latest_file = max(result_files, key=os.path.getctime) if result_files else None
        if latest_file:
            print(f"\n查看最新文件: {latest_file}")
            try:
                with open(latest_file, 'r') as f:
                    data = json.load(f)

                # 分析数据结构
                if isinstance(data, dict):
                    print(f"\n数据键: {list(data.keys())}")

                    # 查找样本结果
                    for key in ['sample_results', 'samples', 'predictions', 'results']:
                        if key in data:
                            samples = data[key]
                            if samples and len(samples) > 0:
                                print(f"\n{key} 包含 {len(samples)} 个样本")

                                # 显示前几个样本
                                for i, sample in enumerate(samples[:3]):
                                    print(f"\n样本 {i+1}:")
                                    for k, v in sample.items():
                                        if k in ['question', 'prediction', 'answer', 'golden_answers', 'is_correct']:
                                            print(f"  {k}: {v}")

            except Exception as e:
                print(f"读取失败: {e}")
    else:
        print("\n未找到结果文件")

    print("\n" + "="*60)
    print("🔍 问题诊断:")
    print("1. 检查extract_okvqa_answer是否正确工作")
    print("2. 检查实际生成的答案内容")
    print("3. 检查VQA评估器的评估逻辑")
    print("4. 确认答案格式是否符合要求")
    print("="*60)

if __name__ == "__main__":
    main()