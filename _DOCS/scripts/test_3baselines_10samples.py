#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试3个完整实现的baseline - 10样本验证

验证：
1. Self-RAG实现正确性
2. mR²AG实现正确性
3. VisRAG实现正确性
4. 3个指标不再为0
"""

import sys
sys.path.insert(0, '/root/autodl-tmp/FlashRAG')

import os
import json
import datasets
from datetime import datetime

# 设置CUDA设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 从主脚本导入
from FlashRAG.experiments.run_all_baselines_100samples import *

def main():
    print("=" * 80)
    print("3个Baseline 10样本验证测试")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # === 加载数据 ===
    print("1. 加载MRAG-Bench数据集...")
    dataset = datasets.load_from_disk(CONFIG['dataset_path'])
    samples = dataset.select(range(10))  # 只取10个样本
    print(f"✅ 加载完成: {len(samples)}个样本")
    print()
    
    # === 初始化模型 ===
    print("2. 初始化模型和检索器...")
    qwen3_vl = create_qwen3_vl_wrapper(CONFIG)
    retriever = create_retriever(CONFIG)
    print("✅ 初始化完成")
    print()
    
    # === 测试3个baseline ===
    methods = [
        ('Self-RAG', SelfRAGPipeline),
        ('mR²AG', MR2AGPipeline),
        ('VisRAG', VisRAGPipeline),
    ]
    
    all_results = {}
    
    for method_name, pipeline_class in methods:
        print(f"\n{'='*80}")
        print(f"测试 {method_name}")
        print(f"{'='*80}\n")
        
        # 创建pipeline
        pipeline = pipeline_class(qwen3_vl, retriever, CONFIG)
        
        results = []
        for i, sample in enumerate(samples):
            print(f"[{method_name}] 处理样本 {i+1}/10...", flush=True)
            result = pipeline.run_single(sample)
            results.append(result)
        
        all_results[method_name] = results
        print(f"\n✅ {method_name} 完成: {len(results)}个结果")
        
        # 检查关键字段
        sample_result = results[0]
        print(f"\n字段检查:")
        print(f"  • retrieval_result: {'✅' if 'retrieval_result' in sample_result else '❌'}")
        print(f"  • attributions: {'✅' if 'attributions' in sample_result else '❌'}")
        print(f"  • position_bias_results: {'✅' if 'position_bias_results' in sample_result else '❌'}")
    
    # === 保存结果 ===
    print(f"\n{'='*80}")
    print("保存测试结果...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'test_3baselines_results_{timestamp}.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 结果已保存: {output_file}")
    
    # === 统计摘要 ===
    print(f"\n{'='*80}")
    print("测试摘要:")
    print(f"{'='*80}")
    for method_name in all_results:
        results = all_results[method_name]
        print(f"\n{method_name}:")
        print(f"  • 完成样本数: {len(results)}")
        print(f"  • 平均检索文档数: {sum(len(r.get('retrieved_docs', [])) for r in results) / len(results):.1f}")
        
        # 检查特殊字段
        if method_name == 'Self-RAG':
            retrieval_decisions = [r.get('retrieval_decision', 'Unknown') for r in results]
            print(f"  • 检索决策: {retrieval_decisions.count('Retrieval')}/10 需要检索")
        elif method_name == 'mR²AG':
            avg_paras = sum(r.get('total_paragraphs', 0) for r in results) / len(results)
            avg_rel = sum(r.get('relevant_paragraphs', 0) for r in results) / len(results)
            print(f"  • 平均段落数: {avg_paras:.1f}")
            print(f"  • 平均相关段落数: {avg_rel:.1f}")
        elif method_name == 'VisRAG':
            reranker_count = sum(1 for r in results if r.get('reranker_used', False))
            print(f"  • 使用Reranker: {reranker_count}/10")
    
    print(f"\n{'='*80}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    print("\n✅ 测试完成！")
    print("\n下一步:")
    print("  1. 检查结果文件，确认所有字段正确")
    print("  2. 如果测试通过，运行100样本对比实验")
    print("  3. 对比新旧实验结果，确认baseline修复成功")

if __name__ == '__main__':
    main()
