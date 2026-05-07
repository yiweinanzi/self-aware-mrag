#!/usr/bin/env python3
"""
诊断MuRAG准确率为0%的原因
"""

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL')
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

import json
import os
from flashrag.dataset.dataset import OKVQADataset

def main():
    print("🔍 诊断MuRAG准确率为0%的原因")
    print("="*60)

    # 加载数据集
    dataset = OKVQADataset(
        dataset_path='/root/autodl-fs/dataset/okvqa_2024',
        max_samples=3,
        image_load=True
    )

    print(f"\n📊 加载了 {len(dataset)} 个样本")

    # 检查样本格式
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\n--- 样本 {i+1} ---")
        print(f"问题: {sample['question']}")
        print(f"正确答案: {sample.get('golden_answers', [])}")
        print(f"图像: {'有' if sample.get('image') else '无'}")

        # 检查是否有选择题选项
        has_options = any(k in sample for k in ['A', 'B', 'C', 'D'])
        print(f"选择题选项: {'有' if has_options else '无'}")

        if has_options:
            for opt in ['A', 'B', 'C', 'D']:
                if opt in sample:
                    print(f"  {opt}: {sample[opt]}")

    print("\n" + "="*60)
    print("📋 诊断结论:")
    print("1. OK-VQA是开放域问答，不是选择题")
    print("2. MuRAG可能错误地当作选择题处理")
    print("3. 需要检查MuRAG的答案生成逻辑")
    print("="*60)

    # 测试VQA答案提取
    from flashrag.utils.vqa_evaluator import extract_okvqa_answer

    test_answers = [
        "basketball",
        "basketball racing",
        "The answer is basketball",
        "basketball.",
        "This is a basketball"
    ]

    print("\n🧪 测试答案提取:")
    for ans in test_answers:
        extracted = extract_okvqa_answer(ans)
        print(f"  '{ans}' -> '{extracted}'")

if __name__ == "__main__":
    main()