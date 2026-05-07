#!/usr/bin/env python
"""测试RagVL的答案提取问题"""

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from flashrag.utils.vqa_evaluator import extract_okvqa_answer
from experiments.baselines.answer_extractor import extract_answer_smart

# 测试可能的答案
test_answers = [
    "motorcycle racing",
    "It's used for motorcycle racing",
    "racing",
    "The sport is racing",
    "race"
]

golden = "race"

print("测试答案提取函数：")
print("=" * 60)
print(f"期望答案: '{golden}'")
print("-" * 60)

for ans in test_answers:
    okvqa = extract_okvqa_answer(ans)
    smart = extract_answer_smart(ans)

    print(f"原始答案: '{ans}'")
    print(f"  extract_okvqa_answer: '{okvqa}' {'✅' if okvqa == golden else '❌'}")
    print(f"  extract_answer_smart: '{smart}' {'✅' if smart == golden else '❌'}")
    print()

print("结论：")
print("- extract_okvqa_answer只提取前几个词，可能包含无关内容")
print("- extract_answer_smart有更好的后处理")