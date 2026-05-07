#!/usr/bin/env python
"""测试mR²AG修复效果"""

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

# 测试Retrieval-Reflection
from experiments.baselines.mr2ag_enhanced import MR2AGFixed

# 模拟wrapper
class MockQwen3VL:
    def generate(self, text, image=None, max_new_tokens=20, temperature=0.1, do_sample=False):
        # 简单模拟响应
        if "color" in text.lower():
            return "red"
        elif "year" in text.lower():
            return "2020"
        elif "What is this" in text:
            return "motorcycle"
        else:
            return "answer"

print("Testing mR²AG Fixed Retrieval-Reflection...")
print("=" * 70)

# 创建实例
mr2ag = MR2AGFixed(MockQwen3VL(), None)

# 测试问题
test_questions = [
    "What color is the car?",  # 常识 - 不需要检索
    "What year was this invented?",  # 知识 - 需要检索
    "What toy is this?",  # 知识 - 需要检索
    "How many cats are in the picture?",  # 常识 - 不需要检索
    "Who designed this statue?",  # 知识 - 需要检索
]

for q in test_questions:
    need_retrieval = mr2ag._retrieval_reflection(q)
    print(f"问题: {q}")
    print(f"需要检索: {'Yes' if need_retrieval else 'No'}")
    print()

print("\nTesting Relevance-Reflection...")
print("=" * 70)

# 测试相关性
test_cases = [
    ("What sport is this?", "Motorcycle racing is a dangerous sport."),
    ("What year was this built?", "The building was constructed in 1850."),
    ("Who painted this?", "The artist used oil paints for this portrait."),
]

for q, p in test_cases:
    relevant, score = mr2ag._relevance_reflection(q, p)
    print(f"问题: {q}")
    print(f"段落: {p}")
    print(f"相关: {'Yes' if relevant else 'No'} (分数: {score:.2f})")
    print()