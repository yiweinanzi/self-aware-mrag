#!/usr/bin/env python3
"""诊断Self-Aware-MRAG的问题"""

# 从数据集加载前10个样本
from datasets import load_from_disk
import re

dataset_path = "/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MRAG-Bench/raw"
dataset_dict = load_from_disk(dataset_path)
test_data = dataset_dict['test'].select(range(10))

# 读取输出
with open('mrag_fixed_472.out', 'r') as f:
    content = f.read()

# 提取不确定性信息
uncertainty_pattern = r'\[DEBUG\] uncertainty=(\d\.\d+) \[text=(\d\.\d+), visual=(\d\.\d+), align=(\d\.\d+)\], threshold=\d\.\d+, should_retrieve=(True|False)'
uncertainties = re.findall(uncertainty_pattern, content)

print("Self-Aware-MRAG 诊断报告")
print("="*80)

print("\n1. 不确定性分析:")
print("-"*40)
for i, (unc, text, visual, align, retrieve) in enumerate(uncertainties[:10]):
    sample = test_data[i]
    print(f"\nSample {i}:")
    print(f"  场景: {sample['scenario']}")
    print(f"  不确定性: {float(unc):.4f} (text={text}, visual={visual}, align={align})")
    print(f"  是否检索: {retrieve}")
    print(f"  问题: {sample['question'][:60]}...")

# 分析阈值问题
print("\n\n2. 阈值分析:")
print("-"*40)
threshold = 0.43
if uncertainties:
    avg_uncertainty = sum(float(u[0]) for u in uncertainties[:10]) / len(uncertainties[:10])
    print(f"当前阈值: {threshold}")
    print(f"平均不确定性: {avg_uncertainty:.4f}")
else:
    avg_uncertainty = 0
    print("未找到不确定性数据")

if avg_uncertainty < threshold:
    print(f"⚠️  平均不确定性({avg_uncertainty:.4f}) < 阈值({threshold})")
    print(f"   建议：降低阈值以鼓励更多检索")

# 提高准确率的建议
print("\n\n3. 提高准确率的建议:")
print("-"*40)
print("a) 调整不确定性阈值:")
print("   - 当前阈值0.43可能过高，导致大部分样本不进行检索")
print("   - 建议尝试0.3-0.35之间的阈值")
print("\nb) 改进不确定性估计:")
print("   - text不确定性普遍较高(0.45-0.65)")
print("   - 可以考虑重新校准text分量的权重")
print("\nc) 增强检索质量:")
print("   - 即使检索了文档，模型可能还是答错了")
print("   - 需要检查检索文档的相关性")

# 查看错误案例
print("\n\n4. 错误案例分析:")
print("-"*40)

# 提取错误样本
error_samples = [
    {
        'id': 0,
        'gt': 'A',
        'gt_text': 'silky_terrier',
        'pred': 'Yorkshire_terrier',
        'unc': uncertainties[0][0]
    },
    {
        'id': 2,
        'gt': 'C',
        'gt_text': 'New York City',
        'pred': 'Chicago',
        'unc': uncertainties[2][0]
    }
]

for err in error_samples:
    sample = test_data[err['id']]
    print(f"\nSample {err['id']} (不确定性={err['unc']}):")
    print(f"  问题: {sample['question']}")
    print(f"  正确答案: {err['gt']} ({err['gt_text']})")
    print(f"  模型输出: {err['pred']}")
    print(f"  选项: A({sample['A']}), B({sample['B']}), C({sample['C']}), D({sample['D']})")
    print(f"  分析: 模型选择了{err['pred']}，它接近{err['gt_text']}但不是正确答案")
    if err['pred'].lower() in [sample['A'].lower(), sample['B'].lower(), sample['C'].lower(), sample['D'].lower()]:
        for choice in ['A', 'B', 'C', 'D']:
            if err['pred'].lower() in sample[choice].lower():
                print(f"  -> 实际上选择了选项 {choice}")

print("\n\n5. 核心问题:")
print("-"*40)
print("Self-Aware-MRAG的一个核心问题是：当模型不确定时，它应该检索")
print("并利用检索到的信息来纠正自己的错误，而不是依赖自己的先验知识。")