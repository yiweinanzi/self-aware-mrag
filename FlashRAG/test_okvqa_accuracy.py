#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试OK-VQA准确率 - 简化版本
"""

import os
import sys
import json
from datetime import datetime

sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
from flashrag.dataset.okvqa_dataset_simple import OKVQADatasetSimple

def main():
    print("="*80)
    print("OK-VQA 准确率测试 - 带图像")
    print("="*80)

    # 1. 加载数据
    print("\n1. 加载数据集（带图像）")
    dataset = OKVQADatasetSimple({
        'data_dir': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/VQA',
        'split': 'val',
        'load_images': True,
    })
    print(f"✅ 加载了 {len(dataset)} 个样本，全部包含图像")

    # 2. 初始化模型
    print("\n2. 初始化Qwen3-VL模型")
    qwen3_vl = create_qwen3_vl_wrapper(
        model_path='/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
        device='cuda',
        torch_dtype='bfloat16'
    )
    print("✅ 模型加载成功")

    # 3. 测试前10个样本
    print("\n3. 测试前10个样本")
    print("-" * 40)

    correct = 0
    total = 10

    for i in range(min(total, len(dataset))):
        sample = dataset[i]
        question = sample['question']
        image = sample['image']
        golden_answers = sample['golden_answers']

        print(f"\n样本 {i+1}:")
        print(f"问题: {question}")
        print(f"图像: {'已加载' if image else '未加载'}")

        # 生成答案
        try:
            # 构建简单的prompt
            prompt = f"Question: {question}\n\nAnswer with one word:"

            # 生成
            answer = qwen3_vl.generate(
                text=prompt,
                image=image,
                max_new_tokens=5,
                temperature=0.0,
                do_sample=False
            ).strip().lower()

            # 提取第一个词
            answer_words = answer.split()
            if answer_words:
                answer = answer_words[0]

            # 检查是否正确
            is_correct = answer in golden_answers
            if is_correct:
                correct += 1

            print(f"生成答案: {answer!r}")
            print(f"标准答案: {golden_answers[:3]}")
            print(f"结果: {'✅ 正确' if is_correct else '❌ 错误'}")

        except Exception as e:
            print(f"❌ 生成失败: {e}")

    # 4. 统计结果
    accuracy = correct / total * 100
    print("\n" + "="*80)
    print("测试结果")
    print("="*80)
    print(f"准确率: {accuracy:.1f}% ({correct}/{total})")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 5. 分析
    print("\n分析:")
    if accuracy == 0:
        print("- 准确率为0%可能是因为:")
        print("  1. 模型需要更长的prompt来理解任务")
        print("  2. OK-VQA是一个具有挑战性的数据集")
        print("  3. 可能需要few-shot示例")
        print("  4. 图像内容可能与问题不匹配")
    elif accuracy < 30:
        print("- 准确率较低，建议:")
        print("  1. 优化prompt设计")
        print("  2. 使用few-shot提示")
        print("  3. 调整生成参数")
    else:
        print("- 准确率可以接受！")

if __name__ == "__main__":
    main()