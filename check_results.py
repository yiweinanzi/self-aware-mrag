#!/usr/bin/env python3
"""
检查实验结果
"""

import json

# 读取结果文件
with open('/data0/home/zqwang/ACL/FlashRAG/results_final_validation/unified_ablation_results_20251216_185845.json', 'r') as f:
    data = json.load(f)

# 检查样本结果
for method in ['MuRAG', 'VisRAG']:
    print(f"\n=== {method} 样本结果 ===")
    samples = data['detailed_results'][method]['sample_results']

    for i, sample in enumerate(samples):
        print(f"\n样本 {i+1}:")
        print(f"  问题: {sample['question']}")
        print(f"  生成答案: '{sample['answer']}'")
        print(f"  正确答案: {sample['golden_answers']}")

        # 检查答案是否匹配
        golden = sample['golden_answers']
        is_correct = sample['answer'] in golden
        print(f"  是否正确: {'✅' if is_correct else '❌'}")