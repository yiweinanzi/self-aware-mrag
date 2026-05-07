#!/usr/bin/env python
"""诊断ViDoRAG、RagVL、SAM-RAG生成的答案"""

import sys
import os
sys.path.insert(0, '/data0/home/zqwang/ACL')
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

import warnings
warnings.filterwarnings("ignore")

# 模拟测试数据
test_samples = [
    {
        'question': 'What sport can you use this for?',
        'golden_answers': ['race', 'race', 'race'],
        'image': 'test_image_1.jpg'
    },
    {
        'question': 'What color is the apple?',
        'golden_answers': ['red', 'red', 'red'],
        'image': 'test_image_2.jpg'
    }
]

# 简化的评估函数
def evaluate_simple(answer, golden_answers):
    """简单的答案评估"""
    if not answer:
        return False
    answer = str(answer).strip().lower()
    for ga in golden_answers[:3]:
        if answer == ga.lower():
            return True
    return False

print("=" * 80)
print("诊断：ViDoRAG、RagVL、SAM-RAG的答案生成问题")
print("=" * 80)

# 测试答案提取
from flashrag.utils.vqa_evaluator import extract_okvqa_answer
from experiments.baselines.answer_extractor import extract_answer_smart

test_answers = [
    "racing",
    "It is used for racing sports",
    "race car",
    "motorcycle racing",
    "The sport is racing",
    "bicycle",
    "",
    "I don't know"
]

print("\n1. 测试答案提取函数：")
print("-" * 50)
for ans in test_answers:
    extracted1 = extract_okvqa_answer(ans)
    extracted2 = extract_answer_smart(ans)
    match_race = evaluate_simple(extracted1, ['race', 'race', 'race'])
    print(f"原答案: '{ans}'")
    print(f"  extract_okvqa_answer: '{extracted1}' (匹配: {match_race})")
    print(f"  extract_answer_smart: '{extracted2}'")
    print()

print("\n2. 方法问题分析：")
print("-" * 50)
print("ViDoRAG问题：")
print("- 可能生成了过长或不匹配的答案")
print("- 需要检查prompt是否要求1-3词答案")

print("\nRagVL问题：")
print("- 可能生成了描述性答案而非关键词")
print("- MLLM reranker可能选择了错误的文档")

print("\nSAM-RAG问题：")
print("- 记忆机制可能干扰了答案生成")
print("- 需要检查记忆context的构建")

print("\n3. 建议的修复方案：")
print("-" * 50)
print("1. 所有方法都使用统一的prompt模板")
print("2. 强制要求1-3词答案")
print("3. 添加答案后处理")
print("4. 调试时打印实际生成的原始答案")