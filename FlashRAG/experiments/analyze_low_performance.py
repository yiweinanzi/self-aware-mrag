#!/usr/bin/env python3
"""分析低准确率方法的问题"""

import re
import json
from datasets import load_from_disk

# 加载数据集
dataset_path = "/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MRAG-Bench/raw"
dataset_dict = load_from_disk(dataset_path)
samples = dataset_dict['test'].select(range(10))

# 读取输出文件
with open('mrag_fixed_476.out', 'r') as f:
    content = f.read()

print("="*80)
print("低准确率方法分析")
print("="*80)

# 分析SAM-RAG (20%)
print("\n1. SAM-RAG 分析 (准确率: 20%)")
print("-"*40)
samrag_pattern = r"计算 SAM-RAG 的指标.*?(?=计算|✅)"
samrag_section = re.search(samrag_pattern, content, re.DOTALL)
if samrag_section:
    section = samrag_section.group()
    # 提取错误样本
    errors = re.findall(r"Sample (\d+):.*?GT choice: ([ABCD]).*?Pred parsed: ([ABCD])", section, re.DOTALL)
    print(f"发现 {len(errors)} 个样本")
    for i, (sample_id, gt, pred) in enumerate(errors[:5]):
        if int(sample_id) < len(samples):
            s = samples[int(sample_id)]
            print(f"\n样本 {sample_id}:")
            print(f"  场景: {s['scenario']}")
            print(f"  问题: {s['question'][:50]}...")
            print(f"  正确答案: {gt} ({s[gt]})")
            print(f"  模型输出: {pred} ({s[pred] if pred in s else 'N/A'})")
            print(f"  错误: 模型选了{pred}而不是{gt}")

# 分析ViDoRAG (30%)
print("\n\n2. ViDoRAG 分析 (准确率: 30%)")
print("-"*40")
vidorag_pattern = r"评测方法: ViDoRAG.*?(?=评测方法|✅)"
vidorag_section = re.search(vidorag_pattern, content, re.DOTALL)
if vidorag_section:
    section = vidorag_section.group()
    errors = re.findall(r"Sample (\d+):.*?GT choice: ([ABCD]).*?Pred parsed: ([ABCD])", section, re.DOTALL)
    print(f"发现 {len(errors)} 个样本")
    for i, (sample_id, gt, pred) in enumerate(errors[:5]):
        if int(sample_id) < len(samples):
            s = samples[int(sample_id)]
            print(f"\n样本 {sample_id}:")
            print(f"  场景: {s['scenario']}")
            print(f"  问题: {s['question'][:50]}...")
            print(f"  正确答案: {gt} ({s[gt]})")
            print(f"  模型输出: {pred}")

# 分析VisRAG (10%)
print("\n\n3. VisRAG 分析 (准确率: 10%)")
print("-"*40)
visrag_pattern = r"评测方法: VisRAG.*?(?=评测方法|✅)"
visrag_section = re.search(visrag_pattern, content, re.DOTALL)
if visrag_section:
    section = visrag_section.group()
    errors = re.findall(r"Sample (\d+):.*?GT choice: ([ABCD]).*?Pred parsed: ([ABCD])", section, re.DOTALL)
    print(f"发现 {len(errors)} 个样本")
    for i, (sample_id, gt, pred) in enumerate(errors[:5]):
        if int(sample_id) < len(samples):
            s = samples[int(sample_id)]
            print(f"\n样本 {sample_id}:")
            print(f"  场景: {s['scenario']}")
            print(f"  问题: {s['question'][:50]}...")
            print(f"  正确答案: {gt} ({s[gt]})")
            print(f"  模型输出: {pred}")

print("\n\n4. 可能的原因分析")
print("-"*40)
print("a) SAM-RAG (20%):")
print("   - 使用4批次的SAM模式可能不适合MRAG-Bench")
print("   - 可能在处理多选题时策略不当")
print("   - 检索-生成-检索-生成的循环可能没有利用好")

print("\nb) ViDoRAG (30%):")
print("   - 视频导向的RAG可能不适用于静态图片")
print("   - 时序建模对MRAG-Bench可能没有帮助")
print("   - 可能需要更多文本信息支持")

print("\nc) VisRAG (10%):")
print("   - 纯视觉RAG，完全依赖图像信息")
print("   - MRAG-Bench需要外部知识（如城市名、品种名等）")
print("   - 没有文本检索导致无法获取必要知识")

print("\n5. 改进建议")
print("-"*40)
print("- 这些方法需要更好地结合文本检索")
print("- 考虑降低对纯视觉信息的依赖")
print("- 改进提示词，更好地引导模型利用检索到的文档")