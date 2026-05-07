#!/usr/bin/env python3
"""
诊断MRAG实验准确率问题和性能问题
"""

import sys
import json
from datasets import load_dataset

# 添加路径
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

# 加载数据集
print("加载MRAG数据集...")
dataset = load_dataset('/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MRAG-Bench')
samples = dataset['test'].select(range(min(3, len(dataset['test']))))

# 检查数据格式
print("\n=== 数据格式检查 ===")
for i, sample in enumerate(samples):
    print(f"\n样本 {i}:")
    print(f"  问题: {sample['question']}")
    print(f"  选项 A: {sample['A']}")
    print(f"  选项 B: {sample['B']}")
    print(f"  选项 C: {sample.get('C', 'N/A')}")
    print(f"  选项 D: {sample.get('D', 'N/A')}")
    print(f"  正确答案(answer_choice): {sample['answer_choice']}")
    print(f"  答案文本(answer): {sample['answer']}")
    print(f"  场景: {sample['scenario']}")

# 检查解析逻辑
print("\n=== 答案解析测试 ===")

def test_parse_multi_choice_response(response, choices_list, sample):
    """测试答案解析逻辑"""
    import re

    print(f"\n原始响应: {response}")

    # 添加MRAG-Bench eval路径
    sys.path.insert(0, '/data0/home/zqwang/ACL/MRAG-Bench-main/eval/utils')

    # 构建index2ans映射
    index2ans = {
        'A': sample.get('A', '').lower(),
        'B': sample.get('B', '').lower(),
        'C': sample.get('C', '').lower(),
        'D': sample.get('D', '').lower()
    }
    print(f"选项映射: {index2ans}")

    # 尝试导入MRAG-Bench的官方解析函数
    try:
        from automatic_extract import parse_multi_choice_response as official_parse

        # 使用官方解析函数
        result = official_parse(response, choices_list, index2ans)
        print(f"官方解析结果: {result}")

        # 如果失败，返回原文
        if result not in choices_list:
            print("官方解析失败，使用改进逻辑")

    except Exception as e:
        print(f"官方解析失败: {e}")

    # 改进的解析逻辑
    response = response.strip().upper()

    # 清理响应文本
    for char in [',', '.', '!', '?', ';', ':', "'", '"']:
        response = response.strip(char)
    response = " " + response + " "

    # 1. 寻找括号中的选项
    candidates = []
    for choice in choices_list:
        if f'({choice})' in response:
            candidates.append(choice)

    # 2. 寻找独立的选项字母
    if len(candidates) == 0:
        for choice in choices_list:
            if f' {choice} ' in response:
                candidates.append(choice)

    # 3. 尝试匹配常见模式
    if len(candidates) == 0:
        patterns = [
            r'ANSWER IS ([ABCD])',
            r'CHOICE IS ([ABCD])',
            r'ANSWER:([ABCD])',
            r'([ABCD])\.',
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                candidates.append(match.group(1))
                break

    if len(candidates) > 0:
        print(f"解析的候选: {candidates}")
        return candidates[0] if len(candidates) == 1 else candidates[-1]

    print("无法解析答案")
    return response.strip()

# 测试一些可能的响应格式
test_responses = [
    "A",
    "The answer is A",
    "Answer: A",
    "(A)",
    "I think the answer is (B)",
    "silky_terrier",  # 选择了答案文本而不是选项
    "The correct choice is C",
    "Answer is (A)",
    "I choose B"
]

print("\n测试不同答案格式:")
for i, sample in enumerate(samples):
    gt = sample['answer_choice'].upper()
    print(f"\n样本 {i} - 正确答案: {gt}")

    for resp in test_responses[:3]:  # 只测试前3种格式
        parsed = test_parse_multi_choice_response(resp, ['A', 'B', 'C', 'D'], sample)
        correct = "✓" if parsed == gt else "✗"
        print(f"  响应: '{resp}' -> 解析: '{parsed}' {correct}")