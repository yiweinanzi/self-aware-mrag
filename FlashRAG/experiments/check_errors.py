#!/usr/bin/env python3
"""检查低准确率方法的具体错误"""

from datasets import load_from_disk

# 加载数据
dataset_path = "/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MRAG-Bench/raw"
dataset_dict = load_from_disk(dataset_path)
samples = dataset_dict['test'].select(range(10))

print("="*60)
print("低准确率方法错误分析")
print("="*60)

# 查看一些具体的错误案例
print("\n1. 典型错误案��（所有方法都容易出错的）")
print("-"*40)

# Sample 0: 动物识别
sample = samples[0]
print(f"\n样本0 ({sample['scenario']}):")
print(f"问题: {sample['question']}")
print(f"A: {sample['A']} (正确答案)")
print(f"B: {sample['B']} (模型常选)")
print(f"C: {sample['C']}")
print(f"D: {sample['D']}")
print("分析: Yorkshire_terrier和silky_terrier非常相似，容易���淆")

# Sample 2: 建筑识别
sample = samples[2]
print(f"\n样本2 ({sample['scenario']}):")
print(f"问题: {sample['question']}")
print(f"A: {sample['A']}")
print(f"B: {sample['B']} (模型常选)")
print(f"C: {sample['C']} (正确答案)")
print(f"D: {sample['D']}")
print("分析: 纽约和芝加哥都是美国大城市，需要具体知识")

# Sample 4: 建筑识别
sample = samples[4]
print(f"\n样本4 ({sample['scenario']}):")
print(f"问题: {sample['question']}")
print(f"A: {sample['A']}")
print(f"B: {sample['B']}")
print(f"C: {sample['C']}")
print(f"D: {sample['D']} (正确答案)")
print("分析: 凡尔赛宫在法国，需要外部知识")

print("\n\n2. 方法特定问题分析")
print("-"*40)

print("\nSAM-RAG (20%):")
print("- 问题: 使用SAM(Speculative Augmentation Manifold)模式")
print("- 可能原因: 4批次的处理方式不适合多选题")
print("- SAM可能更适合开放式问答")

print("\nViDoRAG (30%):")
print("- 问题: 视频RAG用于静态图片")
print("- 可能原因: 时序建模对单张图片没有帮助")
print("- 视频特有的处理逻辑可能干扰了图片理解")

print("\nVisRAG (10%):")
print("- 问题: 纯视觉RAG，不使用文本检索")
print("- 可能原因:")
print("  1. 需要外部知识的问题无法回答（如城市名、品种名）")
print("  2. 缺乏文本检索导致无法获取必要背景信息")
print("  3. 纯视觉模型难以处理抽象概念")

print("\n\n3. 改进建议")
print("-"*40)
print("1. SAM-RAG: 调整为适合多选题的提示词和策略")
print("2. ViDoRAG: 增加对静态图片的特殊处理逻辑")
print("3. VisRAG: 结合文本检索，而不是纯视觉")