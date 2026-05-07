#!/usr/bin/env python3
"""分析Self-Aware-MRAG的详细结果"""

with open('mrag_fixed_472.out', 'r') as f:
    content = f.read()

# 找到Self-Aware-MRAG部分
start_idx = content.find("计算 Self-Aware-MRAG 的指标")
if start_idx == -1:
    print("未找到Self-Aware-MRAG部分")
    exit()

# 找到下一个方法开始的位置
end_idx = content.find("评测方法: SAM-RAG", start_idx)
self_aware_content = content[start_idx:end_idx]

print("Self-Aware-MRAG 详细分析")
print("="*80)

# 提取准确率
import re
accuracy_match = re.search(r'MRAG-Bench Overall Accuracy: (\d+\.\d+)%', self_aware_content)
if accuracy_match:
    print(f"整体准确率: {accuracy_match.group(1)}%")

# 提取样本信息
samples = []
# 查找所有样本块
sample_blocks = re.split(r'\[DEBUG\] Sample (\d+):', self_aware_content)[1:]

for i, block in enumerate(sample_blocks):
    lines = block.strip().split('\n')
    if not lines or 'Question:' not in lines[0]:
        continue

    sample_info = {
        'id': i,
        'question': lines[0].replace('Question: ', '').strip(),
        'gt_choice': None,
        'gt_text': None,
        'pred_raw': None,
        'pred_parsed': None
    }

    for line in lines[1:]:
        if 'GT choice:' in line:
            sample_info['gt_choice'] = line.split('GT choice:')[1].strip()
        elif 'GT text:' in line:
            sample_info['gt_text'] = line.split('GT text:')[1].strip()
        elif 'Pred raw:' in line:
            sample_info['pred_raw'] = line.split('Pred raw:')[1].strip()
        elif 'Pred parsed:' in line:
            sample_info['pred_parsed'] = line.split('Pred parsed:')[1].strip()

    samples.append(sample_info)

# 分析错误案例
print("\n详细结果分析:")
print("-"*80)
correct = 0
total = len(samples)

for sample in samples:
    if sample['gt_choice'] and sample['pred_parsed']:
        is_correct = sample['gt_choice'] == sample['pred_parsed']
        if is_correct:
            correct += 1
            status = "✅ 正确"
        else:
            status = "❌ 错误"

        print(f"\nSample {sample['id']}: {status}")
        print(f"  问题: {sample['question'][:60]}...")
        print(f"  正确答案: {sample['gt_choice']} ({sample['gt_text']})")
        print(f"  模型输出: {sample['pred_raw'][:50]}...")
        print(f"  解析结果: {sample['pred_parsed']}")

print(f"\n{'='*80}")
print(f"总结: {correct}/{total} = {correct/total*100:.1f}% 准确率")

# 分析错误类型
print(f"\n错误分析:")
error_samples = [s for s in samples if s['gt_choice'] != s['pred_parsed']]
print(f"错误样本数: {len(error_samples)}")

for sample in error_samples[:3]:  # 只显示前3个错误
    print(f"\nSample {sample['id']}:")
    print(f"  - 模型说了 '{sample['pred_raw']}' 但正确答案是 {sample['gt_choice']} ({sample['gt_text']})")
    print(f"  - 可能原因: 模型可能被检索到的误导性文档影响了")