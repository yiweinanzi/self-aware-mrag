#!/usr/bin/env python3
"""测试MRAG答案解析逻辑"""

from datasets import load_dataset

# 加载数据集
dataset = load_dataset('/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MRAG-Bench')
samples = dataset['test']

print("测试答案解析逻辑...")
print("="*80)

# 模拟模型的输出和解析结果
model_outputs = [
    "Yorkshire_terrier",  # Sample 0: 应该选A，但模型说了B的内容
    "capuchin",           # Sample 1: 正确！答案就是C
    "Chicago",            # Sample 2: 应该选C(New York)，但模型说了B
]

for i in range(3):
    sample = samples[i]
    print(f"\nSample {i}:")
    print(f"  Question: {sample['question'][:60]}...")
    print(f"  A: {sample['A']}")
    print(f"  B: {sample['B']}")
    print(f"  C: {sample['C']}")
    print(f"  D: {sample['D']}")
    print(f"  Ground Truth: {sample['answer_choice']} ({sample[sample['answer_choice']]})")

    # 模拟内容匹配逻辑
    response = model_outputs[i]
    predicted_choice = None

    for choice in ['A', 'B', 'C', 'D']:
        if sample[choice].lower() in response.lower():
            predicted_choice = choice
            break

    print(f"  Model Output: {response}")
    print(f"  Parsed Choice: {predicted_choice}")
    print(f"  Correct: {predicted_choice == sample['answer_choice']}")