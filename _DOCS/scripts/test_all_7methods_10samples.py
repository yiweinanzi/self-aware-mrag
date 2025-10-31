#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试7个方法 - 10样本验证

方法：
1. Self-Aware-MRAG (我们的方法)
2. Self-RAG
3. mR²AG
4. VisRAG
5. REVEAL
6. MuRAG
7. RagVL

验证：
- 所有方法实现正确性
- 7个指标都能正确计算
- 没有报错
"""

import sys
sys.path.insert(0, '/root/autodl-tmp/FlashRAG')

import os
import json
import datasets
from datetime import datetime

# 设置CUDA设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 从主脚本导入baseline
from FlashRAG.experiments.run_all_baselines_100samples import (
    SelfRAGPipeline,
    MR2AGPipeline,
    VisRAGPipeline,
    REVEALPipeline,
    MuRAGPipeline,
    RagVLPipeline,
)

# 导入我们的方法
from flashrag.pipeline.self_aware_pipeline_qwen3vl import SelfAwarePipelineQwen3VL

# 定义辅助函数
from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
from flashrag.retriever.retriever import Retriever

# 配置
CONFIG = {
    'dataset_path': '/root/autodl-tmp/FlashRAG/flashrag/data/MRAG-Bench/raw',
    'model_path': '/root/autodl-tmp/models/Qwen3-VL-8B-Instruct',
    'corpus_path': '/root/autodl-tmp/FlashRAG/corpus/corpus_wiki_3m.jsonl',
    'index_path': '/root/autodl-tmp/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
    'retrieval_method': 'e5',
    'retrieval_topk': 5,
    'uncertainty_threshold': 0.30,
}

def create_retriever(config):
    """创建检索器"""
    retriever = Retriever(config)
    return retriever

def main():
    print("=" * 80)
    print("7个方法 10样本验证测试")
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
    
    # === 定义7个方法 ===
    methods = [
        ('Self-Aware-MRAG', SelfAwarePipelineQwen3VL, True),  # 我们的方法，特殊处理
        ('Self-RAG', SelfRAGPipeline, False),
        ('mR²AG', MR2AGPipeline, False),
        ('VisRAG', VisRAGPipeline, False),
        ('REVEAL', REVEALPipeline, False),
        ('MuRAG', MuRAGPipeline, False),
        ('RagVL', RagVLPipeline, False),
    ]
    
    all_results = {}
    
    for method_name, pipeline_class, is_our_method in methods:
        print(f"\n{'='*80}")
        print(f"测试 {method_name}")
        print(f"{'='*80}\n")
        
        # 创建pipeline
        if is_our_method:
            # 我们的方法
            pipeline = pipeline_class(qwen3_vl, retriever, CONFIG)
        else:
            # Baseline方法
            pipeline = pipeline_class(qwen3_vl, retriever, CONFIG)
        
        results = []
        for i, sample in enumerate(samples):
            print(f"[{method_name}] 处理样本 {i+1}/10...", end='', flush=True)
            try:
                result = pipeline.run_single(sample)
                results.append(result)
                print(" ✅")
            except Exception as e:
                print(f" ❌ 错误: {e}")
                import traceback
                traceback.print_exc()
        
        all_results[method_name] = results
        print(f"\n✅ {method_name} 完成: {len(results)}个结果")
        
        # 检查关键字段
        if results:
            sample_result = results[0]
            print(f"\n字段检查:")
            print(f"  • answer: {'✅' if 'answer' in sample_result else '❌'}")
            print(f"  • retrieval_result: {'✅' if 'retrieval_result' in sample_result else '❌'}")
            print(f"  • attributions: {'✅' if 'attributions' in sample_result else '❌'}")
            print(f"  • position_bias_results: {'✅' if 'position_bias_results' in sample_result else '❌'}")
            
            # 显示特殊信息
            if method_name == 'Self-RAG' and 'retrieval_decision' in sample_result:
                decisions = [r.get('retrieval_decision', 'Unknown') for r in results]
                print(f"  • 检索决策: {decisions.count('Retrieval')}/10 需要检索")
            elif method_name == 'mR²AG' and 'total_paragraphs' in sample_result:
                avg_paras = sum(r.get('total_paragraphs', 0) for r in results) / len(results)
                print(f"  • 平均段落数: {avg_paras:.1f}")
            elif method_name == 'VisRAG' and 'reranker_used' in sample_result:
                reranker_count = sum(1 for r in results if r.get('reranker_used', False))
                print(f"  • 使用Reranker: {reranker_count}/10")
            elif method_name == 'MuRAG' and 'ensemble_size' in sample_result:
                avg_ensemble = sum(r.get('ensemble_size', 0) for r in results) / len(results)
                print(f"  • 平均ensemble大小: {avg_ensemble:.1f}")
            elif method_name == 'RagVL' and 'reranked_count' in sample_result:
                avg_reranked = sum(r.get('reranked_count', 0) for r in results) / len(results)
                print(f"  • 平均rerank后文档数: {avg_reranked:.1f}")
            elif method_name == 'Self-Aware-MRAG':
                retrieval_count = sum(1 for r in results if r.get('retrieval_result', {}).get('used_retrieval', False))
                print(f"  • 触发检索: {retrieval_count}/10")
    
    # === 保存结果 ===
    print(f"\n{'='*80}")
    print("保存测试结果...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'test_all_7methods_10samples_{timestamp}.json'
    
    # 简化结果以便保存（移除不可序列化的对象）
    simplified_results = {}
    for method_name, results in all_results.items():
        simplified_results[method_name] = []
        for r in results:
            simplified = {}
            for k, v in r.items():
                # 只保留基本类型
                if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                    simplified[k] = v
                elif k == 'answer':
                    simplified[k] = str(v)
            simplified_results[method_name].append(simplified)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(simplified_results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 结果已保存: {output_file}")
    
    # === 统计摘要 ===
    print(f"\n{'='*80}")
    print("测试摘要:")
    print(f"{'='*80}")
    
    for method_name in all_results:
        results = all_results[method_name]
        if not results:
            continue
        
        print(f"\n{method_name}:")
        print(f"  • 完成样本数: {len(results)}/10")
        
        # 统计检索使用情况
        retrieved_count = 0
        for r in results:
            if 'used_retrieval' in r and r['used_retrieval']:
                retrieved_count += 1
            elif 'retrieval_result' in r and isinstance(r['retrieval_result'], dict):
                if r['retrieval_result'].get('used_retrieval', False):
                    retrieved_count += 1
        
        print(f"  • 使用检索: {retrieved_count}/10")
        
        # 统计评估字段完整性
        has_retrieval_result = sum(1 for r in results if 'retrieval_result' in r)
        has_attributions = sum(1 for r in results if 'attributions' in r)
        has_position_bias = sum(1 for r in results if 'position_bias_results' in r)
        
        print(f"  • Evaluator字段:")
        print(f"    - retrieval_result: {has_retrieval_result}/10")
        print(f"    - attributions: {has_attributions}/10")
        print(f"    - position_bias_results: {has_position_bias}/10")
    
    print(f"\n{'='*80}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    print("\n✅ 测试完成！")
    
    # 检查是否所有方法都成功
    all_success = all(len(results) == 10 for results in all_results.values())
    
    if all_success:
        print("\n🎉 所有7个方法都成功完成10样本测试！")
        print("\n下一步:")
        print("  1. 检查结果文件，确认所有字段正确")
        print("  2. 运行100样本完整对比实验")
        print("  3. 对比新旧实验结果")
    else:
        print("\n⚠️ 部分方法测试失败，请检查日志")
        for method_name, results in all_results.items():
            if len(results) < 10:
                print(f"  - {method_name}: {len(results)}/10 样本")

if __name__ == '__main__':
    main()

