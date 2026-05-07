#!/usr/bin/env python3
"""
测试所有修复后的baseline方法
"""

import os
import sys
sys.path.insert(0, '/data0/home/zqwang/ACL')
sys.path.insert(0, '/root/autodl-tmp/FlashRAG')

from flashrag.pipeline import Pipeline
from flashrag.utils.evaluator import Evaluator
import json

def test_all_baselines():
    """测试所有baseline方法"""

    print("="*60)
    print("🚀 测试修复后的OK-VQA Baseline方法")
    print("="*60)

    # 初始化组件
    config = {
        'retrieval_method': 'bge',
        'model_path': '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
        'device': 'cuda',
        'max_new_tokens': 10,
        'temperature': 0.1,
        'top_k': 5
    }

    pipeline = Pipeline(config)

    # 加载少量测试数据
    dataset_dir = '/root/autodl-fs/benchmark/OK-VQA'
    dataset = pipeline.load_dataset(dataset_dir, num_samples=5)

    print(f"\n✅ 加载了 {len(dataset)} 个测试样本")

    # 测试方法列表
    methods_to_test = [
        'Self-Aware-MRAG',
        'MuRAG',
        'VisRAG',
        'ViDoRAG',
        'RagVL',
        'SAM-RAG',
        'mR²AG'
    ]

    results = {}

    for method_name in methods_to_test:
        print(f"\n{'='*60}")
        print(f"🔍 测试方法: {method_name}")
        print(f"{'='*60}")

        try:
            # 加载方法
            if method_name == 'Self-Aware-MRAG':
                from flashrag.experiments.ablations import SelfAwareMRAGQwen3VL
                method = SelfAwareMRAGQwen3VL(**config)
            elif method_name == 'MuRAG':
                from flashrag.experiments.baselines.murag_enhanced import MuRAGEnhanced
                method = MuRAGEnhanced(pipeline.model, pipeline.retriever, config)
            elif method_name == 'VisRAG':
                from flashrag.experiments.baselines.visrag_enhanced import VisRAGEnhanced
                method = VisRAGEnhanced(pipeline.model, pipeline.retriever, config)
            elif method_name == 'ViDoRAG':
                from flashrag.experiments.baselines.vidorag_pipeline import ViDoRAGPipeline
                method = ViDoRAGPipeline(pipeline.model, pipeline.retriever, config)
            elif method_name == 'RagVL':
                from flashrag.experiments.baselines.ragvl_enhanced import RagVLEnhanced
                method = RagVLEnhanced(pipeline.model, pipeline.retriever, config)
            elif method_name == 'SAM-RAG':
                from flashrag.experiments.baselines.samrag_adapted import SAMRAGAdapted
                method = SAMRAGAdapted(pipeline.model, pipeline.retriever, config)
            elif method_name == 'mR²AG':
                from flashrag.experiments.baselines.mr2ag_enhanced import MR2AGEnhanced
                method = MR2AGEnhanced(pipeline.model, pipeline.retriever, config)

            # 测试第一个样本
            sample = dataset[0]
            print(f"\n问题: {sample['question']}")
            print(f"正确答案: {sample.get('golden_answers', [])}")

            # 运行方法
            result = method.run_single(sample)

            # 检查结果
            print(f"\n生成答案: \"{result.get('answer', '')}\"")
            print(f"检索状态: {result.get('retrieved', False)}")
            print(f"检索文档数: {len(result.get('retrieved_docs', []))}")

            # 评估答案
            evaluator = Evaluator()
            correct = evaluator.evaluate_okvqa(result.get('answer', ''), sample.get('golden_answers', []))
            print(f"评估结果: {'✓' if correct else '✗'}")

            # 保存结果
            results[method_name] = {
                'answer': result.get('answer', ''),
                'retrieved': result.get('retrieved', False),
                'num_docs': len(result.get('retrieved_docs', [])),
                'correct': correct
            }

        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            results[method_name] = {
                'error': str(e),
                'retrieved': False,
                'correct': False
            }

    # 打印总结
    print(f"\n{'='*60}")
    print("📊 测试总结")
    print(f"{'='*60}")

    for method, result in results.items():
        if 'error' in result:
            print(f"{method:15} ❌ 错误: {result['error']}")
        else:
            status = "✓" if result['correct'] else "✗"
            retrieval = "✓" if result['retrieved'] else "✗"
            print(f"{method:15} {status} 答案 {retrieval} 检索({result['num_docs']} docs)")

    # 保存结果
    with open('/data0/home/zqwang/ACL/test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ 测试完成！结果已保存到 test_results.json")

if __name__ == "__main__":
    test_all_baselines()