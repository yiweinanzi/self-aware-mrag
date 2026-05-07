#!/usr/bin/env python3
"""
诊断准确率问题
"""

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL')
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from flashrag.utils.vqa_evaluator import VQAEvaluator

# 创建评估器
evaluator = VQAEvaluator()

# 测试答案提取和评估
test_cases = [
    # 长答案测试
    ("The sport you can use this for is basketball racing.", ["race", "race", "race"]),
    ("This is a rose flower plant.", ["rose", "rose", "rose"]),

    # 短答案测试
    ("race", ["race", "race", "race"]),
    ("rose", ["rose", "rose", "rose"]),

    # 带标点答案测试
    ("race.", ["race", "race", "race"]),
    ("basketball", ["basketball", "basketball", "basketball"]),
]

print("="*60)
print("测试VQA答案评估")
print("="*60)

for i, (answer, golden) in enumerate(test_cases):
    print(f"\n测试 {i+1}:")
    print(f"  生成答案: '{answer}'")
    print(f"  正确答案: {golden}")

    # 标准化答案
    standardized = evaluator.standardize_answer(answer)
    print(f"  标准化后: '{standardized}'")

    # 评估
    correct = evaluator.evaluate_okvqa(answer, golden)
    print(f"  评估结果: {'✓ 正确' if correct else '✗ 错误'}")

    # 截断到3个单词
    words = standardized.split()
    if len(words) > 3:
        truncated = ' '.join(words[:3])
        print(f"  截断到3词: '{truncated}'")
        correct_trunc = evaluator.evaluate_okvqa(truncated, golden)
        print(f"  截断后评估: {'✓ 正确' if correct_trunc else '✗ 错误'}")

print("\n" + "="*60)
print("问题诊断：")
print("1. 如果答案太长，需要截断到3个单词")
print("2. 答案需要标准化（小写、去除标点）")
print("3. 使用extract_okvqa_answer函数处理长答案")
print("="*60)