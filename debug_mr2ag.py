#!/usr/bin/env python
"""Debug mR²AG performance issues"""

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')
import json

# 读取结果文件
result_file = "/data0/home/zqwang/ACL/FlashRAG/experiments/results_okvqa_baselines/unified_ablation_results_20251217_123144.json"
with open(result_file, 'r') as f:
    data = json.load(f)

# 找到mR²AG的结果
mr2ag_results = None
for variant in data['variants_summary']:
    if 'mR²AG' in variant['variant_name']:
        mr2ag_results = variant
        break

print("=== mR²AG Performance Analysis ===")
print(f"准确率: {mr2ag_results['accuracy']*100:.1f}% ({mr2ag_results['correct_samples']}/{mr2ag_results['total_samples']})")
print(f"检索率: {mr2ag_results['retrieval_rate']*100:.1f}%")
print()

# 分析详细结果
print("\n=== Detailed Analysis ===")
mr2ag_detailed = data['detailed_results']['mR²AG']
samples = mr2ag_detailed['sample_results']

# 统计
retrieval_count = 0
no_retrieval_count = 0
relevant_paragraphs_count = 0
empty_answers = 0

# 检查每个样本
for i, result in enumerate(samples[:20]):  # 只看前20个
    print(f"\n样本 {i+1}:")
    print(f"  问题: {result['question'][:50]}...")
    print(f"  答案: '{result['answer']}'")
    print(f"  Golden: {result['golden_answers'][:3]}")
    print(f"  是否检索: {result['retrieved']}")
    print(f"  总段落数: {result.get('total_paragraphs', 0)}")
    print(f"  相关段落: {result.get('relevant_paragraphs', 0)}")
    print(f"  检索决策: {result.get('retrieval_decision', 'N/A')}")

    if result['retrieved']:
        retrieval_count += 1
        relevant_paragraphs_count += result.get('relevant_paragraphs', 0)
    else:
        no_retrieval_count += 1

    if not result['answer'] or result['answer'].strip() == "":
        empty_answers += 1
        print("  ⚠️ 空答案!")
    elif result['correct']:
        print("  ✅ 正确!")
    else:
        print("  ❌ 错误")

print("\n=== Statistics (first 20 samples) ===")
print(f"检索样本: {retrieval_count}/20 ({retrieval_count*5:.0f}%)")
print(f"无检索样本: {no_retrieval_count}/20 ({no_retrieval_count*5:.0f}%)")
print(f"总相关段落: {relevant_paragraphs_count}")
print(f"空答案数: {empty_answers}")
print(f"平均相关段落/检索样本: {relevant_paragraphs_count/max(retrieval_count, 1):.1f}")

# 检查一些具体的失败案例
print("\n=== Failure Analysis ===")
failures = [r for r in samples if not r['correct']][:5]
for i, result in enumerate(failures):
    print(f"\n失败案例 {i+1}:")
    print(f"  问题: {result['question']}")
    print(f"  生成答案: '{result['answer']}'")
    print(f"  期望答案: {result['golden_answers']}")
    print(f"  检索: {result['retrieved']}")
    if result.get('retrieval_decision') == 'No Retrieval':
        print("  ⚠️ 可能问题：Retrieval-Reflection判断错误，应该检索但没检索")
    elif result.get('relevant_paragraphs', 0) == 0 and result['retrieved']:
        print("  ⚠️ 可能问题：Relevance-Reflection太严格，没有相关段落")

# 检查成功案例
print("\n=== Success Analysis ===")
successes = [r for r in samples if r['correct']][:5]
print(f"成功案例数量: {len([s for s in samples if s['correct']])}")
for i, result in enumerate(successes):
    print(f"\n成功案例 {i+1}:")
    print(f"  问题: {result['question']}")
    print(f"  答案: '{result['answer']}'")
    print(f"  期望: {result['golden_answers']}")
    print(f"  检索: {result['retrieved']}, 相关段落: {result.get('relevant_paragraphs', 0)}")