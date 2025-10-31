#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试Self-RAG实现 - 10样本快速验证
"""
import sys
sys.path.insert(0, '/root/autodl-tmp/FlashRAG')

import json
from datetime import datetime
from FlashRAG.experiments.run_all_baselines_100samples import *

print("=" * 70)
print("测试Self-RAG实现 - 10样本验证")
print("=" * 70)

# 配置
config = CONFIG.copy()
config['max_samples'] = 10

# 加载数据
print("\n加载数据...")
samples = load_dataset(config['dataset_path'], max_samples=10)
print(f"✅ 加载了 {len(samples)} 个样本")

# 初始化
print("\n初始化模型...")
qwen3_vl = initialize_qwen3_vl(config)
retriever = initialize_retriever(config, use_multimodal=False)
print("✅ 模型初始化完成")

# 创建Self-RAG pipeline
print("\n创建Self-RAG pipeline...")
selfrag = SelfRAGPipeline(qwen3_vl, retriever, config)

# 运行测试
print("\n" + "=" * 70)
print("开始测试...")
print("=" * 70)

retrieval_decisions = {
    'No Retrieval': 0,
    'Retrieval': 0,
    'Retrieval (no docs)': 0,
    'Retrieval (no relevant)': 0
}

relevant_counts = []
support_status = {'Supported': 0, 'Not Supported': 0, 'N/A': 0}

for i, sample in enumerate(samples):
    print(f"\n样本 {i+1}/10:")
    print(f"  问题: {sample['question'][:60]}...")
    
    result = selfrag.run_single(sample)
    
    # 统计
    decision = result.get('retrieval_decision', 'Unknown')
    retrieval_decisions[decision] = retrieval_decisions.get(decision, 0) + 1
    
    relevant_count = result.get('relevant_docs_count', 0)
    relevant_counts.append(relevant_count)
    
    support = result.get('support_status', 'N/A')
    support_status[support] = support_status.get(support, 0) + 1
    
    print(f"  决策: {decision}")
    print(f"  相关文档数: {relevant_count}")
    print(f"  支持度: {support}")
    print(f"  答案: {result['answer'][:50]}...")

# 统计结果
print("\n" + "=" * 70)
print("测试结果统计")
print("=" * 70)

print("\n1. 检索决策分布:")
total = sum(retrieval_decisions.values())
for decision, count in retrieval_decisions.items():
    pct = count / total * 100 if total > 0 else 0
    print(f"  {decision:30s}: {count:2d} ({pct:5.1f}%)")

retrieval_rate = (retrieval_decisions.get('Retrieval', 0) / total * 100) if total > 0 else 0
print(f"\n  实际检索率: {retrieval_rate:.1f}%")

print("\n2. 相关文档数分布:")
if relevant_counts:
    print(f"  平均相关文档: {sum(relevant_counts)/len(relevant_counts):.2f}")
    print(f"  最多: {max(relevant_counts)}, 最少: {min(relevant_counts)}")

print("\n3. 支持度分布:")
for status, count in support_status.items():
    pct = count / total * 100 if total > 0 else 0
    print(f"  {status:20s}: {count:2d} ({pct:5.1f}%)")

# 验证
print("\n" + "=" * 70)
print("验证结果")
print("=" * 70)

checks = {
    "检索率在合理范围": 20 <= retrieval_rate <= 80,
    "有自适应检索": retrieval_decisions.get('No Retrieval', 0) > 0,
    "有相关性过滤": any(c < 5 for c in relevant_counts if c > 0),
    "有支持度判断": support_status.get('Supported', 0) + support_status.get('Not Supported', 0) > 0
}

all_passed = True
for check, passed in checks.items():
    status = "✅" if passed else "❌"
    print(f"{status} {check}: {passed}")
    if not passed:
        all_passed = False

print("\n" + "=" * 70)
if all_passed:
    print("✅ 所有验证通过！Self-RAG实现正确！")
else:
    print("⚠️ 部分验证未通过，可能需要调整")
print("=" * 70)

# 保存结果
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_file = f"/root/autodl-tmp/selfrag_test_result_{timestamp}.json"
with open(result_file, 'w') as f:
    json.dump({
        'retrieval_decisions': retrieval_decisions,
        'relevant_counts': relevant_counts,
        'support_status': support_status,
        'checks': checks,
        'all_passed': all_passed
    }, f, indent=2)

print(f"\n结果已保存到: {result_file}")

