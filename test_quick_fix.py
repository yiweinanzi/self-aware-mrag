#!/usr/bin/env python3
"""
快速测试修复后的方法 - 3个样本
"""

import sys
import os
sys.path.insert(0, '/data0/home/zqwang/ACL')
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from FlashRAG.experiments.run_okvqa_baselines import run_baseline_experiment
from datetime import datetime

def main():
    print("🔧 快速测试修复效果")
    print("="*60)
    print("测试方法: Self-Aware-MRAG, MuRAG, VisRAG")
    print("样本数: 3")
    print("="*60)

    # 实验配置
    config = {
        'dataset': 'okvqa',
        'max_samples': 3,
        'model_path': '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
        'torch_dtype': 'bfloat16',
        'max_new_tokens': 20,
        'retrieval_topk': 5,
        'faiss_index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
        'corpus_path': '/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl',
        'retrieval_model_path': '/data0/home/zqwang/ACL/models/bge-large-en-v1.5',
        'use_multimodal_retrieval': True,
        'clip_model_path': '/data0/home/zqwang/ACL/models/clip-vit-large-patch14-336',
        'clip_index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/clip/clip_Flat.index',
        'text_retrieval_weight': 0.6,
        'visual_retrieval_weight': 0.4,
        'use_multi_gpu': False,
        'num_gpus': 1,
        'uncertainty_threshold': 0.43,
        'text_weight': 0.4,
        'visual_weight': 0.3,
        'alignment_weight': 0.3,
        'use_improved_estimator': True,
        'output_dir': f'results_quick_fix_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'save_detailed_results': True,
        'save_sample_results': True,
        'enable_complete_metrics': True,
    }

    # 只测试3个方法
    methods = ['Self-Aware-MRAG', 'MuRAG', 'VisRAG']

    all_results = []
    for method in methods:
        print(f"\n{'='*60}")
        print(f"测试方法: {method}")
        print(f"{'='*60}")

        try:
            result = run_baseline_experiment(config, method)
            all_results.append(result)

            print(f"\n✅ {method} 完成:")
            print(f"   准确率: {result['accuracy']:.2%}")
            print(f"   检索率: {result['retrieval_rate']:.1%}")

            # 显示一些样本结果
            if 'sample_results' in result and len(result['sample_results']) > 0:
                print("\n   样本结果示例:")
                for i, sample in enumerate(result['sample_results'][:2]):
                    print(f"   样本{i+1}:")
                    print(f"     问题: {sample.get('question', '')[:50]}...")
                    print(f"     生成答案: '{sample.get('prediction', '')}'")
                    print(f"     正确答案: {sample.get('golden_answers', [])}")
                    print(f"     是否正确: {sample.get('is_correct', False)}")

        except Exception as e:
            print(f"\n❌ {method} 失败: {e}")
            import traceback
            traceback.print_exc()

    # 总结
    print(f"\n{'='*60}")
    print("📊 测试总结")
    print(f"{'='*60}")

    for result in all_results:
        method = result.get('method', 'Unknown')
        accuracy = result.get('accuracy', 0)
        retrieval_rate = result.get('retrieval_rate', 0)

        print(f"{method:15}: 准确率={accuracy:6.2%}, 检索率={retrieval_rate:5.1%}")

    print("\n✨ 测试完成！")

if __name__ == "__main__":
    main()