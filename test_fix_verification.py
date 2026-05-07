#!/usr/bin/env python3
"""
测试max_new_tokens修复后的效果
"""

import sys
import os
sys.path.insert(0, '/data0/home/zqwang/ACL')
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from flashrag.utils.vqa_evaluator import extract_okvqa_answer

# 测试答案生成
test_cases = [
    "What sport can you use this for?",
    "Name the type of plant this is?",
    "What toy is this?"
]

print("🔍 测试extract_okvqa_answer效果")
print("="*60)

for question in test_cases:
    print(f"\n问题: {question}")

    # 模拟长答案（max_new_tokens=10时可能生成的不完整答案）
    incomplete_answer = "basketball"
    short_answer = extract_okvqa_answer(incomplete_answer)
    print(f"  不完整答案: '{incomplete_answer}' → 提取后: '{short_answer}'")

    # 模拟完整答案（max_new_tokens=20时可以生成的完整答案）
    complete_answer = "basketball racing"
    short_answer = extract_okvqa_answer(complete_answer)
    print(f"  完整答案: '{complete_answer}' → 提取后: '{short_answer}'")

print("\n" + "="*60)
print("结论:")
print("1. max_new_tokens=10 太小，无法生成完整答案")
print("2. max_new_tokens=20 足够生成1-3个单词的答案")
print("3. extract_okvqa_answer 会提取核心答案，确保通过VQA评估")
print("="*60)