#!/usr/bin/env python3
"""测试MRAG-Bench使用gt_images的效果"""

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
from datasets import load_dataset
import json
from datetime import datetime

print("="*80)
print("测试MRAG-Bench使用gt_images")
print("="*80)

# 加载数据集
print("\n1. 加载MRAG-Bench数据集...")
dataset = load_dataset("uclanlp/MRAG-Bench", split="test")
print(f"   加载了 {len(dataset)} 个样本")

# 初始化模型
print("\n2. 初始化Qwen3-VL...")
qwen3_vl = create_qwen3_vl_wrapper(
    model_path='/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
    device='cuda',
    torch_dtype='bfloat16'
)

# 测试第一个样本
print("\n3. 测试第一个样本...")
sample = dataset[0]
print(f"   问题: {sample['question']}")
print(f"   正确答案: {sample['answer_choice']}")
print(f"   场景: {sample.get('scenario', 'Unknown')}")
print(f"   gt_images数量: {len(sample.get('gt_images', []))}")

# 构建prompt
prompt = f"""You will be given one question concerning several images. The first image is the input image, others are retrieved examples to help you. Answer with the option's letter from the given choices directly.

Choices:
A: {sample['A']}
B: {sample['B']}
C: {sample['C']}
D: {sample['D']}

Question: {sample['question']}

Answer with the letter only (A/B/C/D):"""

print("\n4. 生成答案...")

# 准备图像
images = [sample['image']]
if 'gt_images' in sample and sample['gt_images']:
    images.extend(sample['gt_images'])
    print(f"   使用了 {len(images)} 张图像（1张输入 + {len(sample['gt_images'])}张检索）")
else:
    print(f"   仅使用1张输入图像（无检索图像）")

try:
    # 生成答案
    response = qwen3_vl.generate(
        text=prompt,
        image=images[:10],  # 限制最多10张图像
        max_new_tokens=5,
        temperature=0.01
    )

    answer = response.strip().upper()
    print(f"\n✅ 生成成功!")
    print(f"   预测答案: {answer}")
    print(f"   正确答案: {sample['answer_choice']}")
    print(f"   是否正确: {answer == sample['answer_choice']}")

except Exception as e:
    print(f"\n❌ 生成失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("测试完成")
print("="*80)