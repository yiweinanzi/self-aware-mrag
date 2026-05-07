#!/usr/bin/env python3
"""
直接测试修复的关键函数
"""

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL')
sys.path.insert(0, '/root/autodl-tmp/FlashRAG')

# 测试extract_okvqa_answer函数
from flashrag.utils.vqa_evaluator import extract_okvqa_answer

# 测试长答案提取
test_answers = [
    "The sport you can use this for is basketball.",
    "This is a type of flowering plant called rose.",
    "You can play tennis with this equipment.",
    "Red",
    "Race car",
    "The answer is: soccer"
]

print("测试extract_okvqa_answer函数:")
print("="*60)
for ans in test_answers:
    extracted = extract_okvqa_answer(ans)
    print(f"原始: {ans!r}")
    print(f"提取: {extracted!r}")
    print("-" * 40)

# 测试VQA评估
from flashrag.utils.evaluator import Evaluator

evaluator = Evaluator()

print("\n测试VQA评估:")
print("="*60)
tests = [
    ("basketball", ["basketball", "basketball", "basketball"]),
    ("rose", ["flower", "rose", "rose"]),
    ("tennis", ["tennis", "tennis", "tennis"]),
    ("red", ["red", "red", "red"]),
    ("race car", ["race", "race", "race"]),
    ("soccer", ["soccer", "soccer", "soccer"])
]

for pred, golden in tests:
    correct = evaluator.evaluate_okvqa(pred, golden)
    print(f"预测: {pred!r:15} | 正确答案: {golden} | 评估: {'✓' if correct else '✗'}")

print("\n✅ 测试完成！")