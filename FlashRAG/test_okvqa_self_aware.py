#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OK-VQA Self-Aware-MRAG测试 - 完整版本
"""

import os
import sys
import json
import time
from datetime import datetime

# 添加FlashRAG路径
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
from flashrag.dataset.okvqa_dataset_simple import OKVQADatasetSimple
from flashrag.retriever import DenseRetriever
from flashrag.pipeline.self_aware_pipeline_qwen3vl import SelfAwarePipelineQwen3VL
from flashrag.utils.comprehensive_evaluator import evaluate_comprehensive_metrics

# ============================================================================
# 配置
# ============================================================================

CONFIG = {
    # 数据集配置
    'dataset_path': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/VQA',
    'max_samples': 10,
    'split': 'val',  # 使用val split
    'load_images': True,  # 加载图像以获得更好的��确率

    # 模型配置
    'qwen3_vl_path': '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
    'torch_dtype': 'bfloat16',
    'temperature': 0.01,

    # 检索器配置
    'faiss_index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
    'corpus_path': '/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl',
    'retrieval_model_path': '/data0/home/zqwang/ACL/models/bge-large-en-v1.5',
    'retrieval_topk': 5,

    # Self-Aware-MRAG配置
    'uncertainty_threshold': 0.43,
    'use_improved_estimator': True,
    'use_position_fusion': True,
    'use_attribution': True,

    # 输出配置
    'output_dir': '/data0/home/zqwang/ACL/FlashRAG/experiments/results_okvqa_self_aware',
}

# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行OK-VQA Self-Aware-MRAG测试"""
    print("="*80)
    print("OK-VQA Self-Aware-MRAG测试")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"样本数: {CONFIG['max_samples']}")

    # 1. 加载数据
    print("\n1. 加载数据集")
    print("-" * 40)
    try:
        dataset = OKVQADatasetSimple({
            'data_dir': CONFIG['dataset_path'],
            'split': CONFIG['split'],
            'load_images': CONFIG['load_images'],
        })

        # 获取前N个样本
        samples = []
        for i in range(min(CONFIG['max_samples'], len(dataset))):
            item = dataset[i]
            sample = {
                'id': item.get('id', f'okvqa_{i}'),
                'question': item['question'],
                'image': item.get('image'),
                'answer': item.get('golden_answers', [''])[0] if item.get('golden_answers') else '',
                'golden_answers': item.get('golden_answers', [''])
            }
            samples.append(sample)

        print(f"✅ 成功加载 {len(samples)} 样本")
        if samples:
            print(f"\n第一个样本:")
            print(f"  问题: {samples[0]['question']}")
            print(f"  答案: {samples[0]['answer']}")
            print(f"  标注: {samples[0]['golden_answers'][:3]}")

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. 初始化模型
    print("\n2. 初始化模型")
    print("-" * 40)
    try:
        qwen3_vl = create_qwen3_vl_wrapper(
            model_path=CONFIG['qwen3_vl_path'],
            device="cuda",
            torch_dtype=CONFIG['torch_dtype']
        )
        print("✅ Qwen3-VL模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 3. 初始化检索器
    print("\n3. 初始化检索器")
    print("-" * 40)
    try:
        retriever_config = {
            'index_path': CONFIG['faiss_index_path'],
            'corpus_path': CONFIG['corpus_path'],
            'retrieval_method': 'e5',
            'retrieval_model_path': CONFIG['retrieval_model_path'],
            'retrieval_query_max_length': 512,
            'retrieval_pooling_method': 'mean',
            'retrieval_use_fp16': True,
            'retrieval_batch_size': 128,
            'retrieval_topk': CONFIG['retrieval_topk'],
            'save_retrieval_cache': False,
            'use_retrieval_cache': False,
            'retrieval_cache_path': None,
            'use_reranker': False,
            'use_sentence_transformer': False,
            'faiss_gpu': False,
            'instruction': '',
        }

        retriever = DenseRetriever(retriever_config)
        print("✅ 检索器加载成功")
    except Exception as e:
        print(f"❌ 检索器加载失败: {e}")
        print("继续使用无检索模式...")
        retriever = None

    # 4. 创建Self-Aware-MRAG pipeline
    print("\n4. 创建Self-Aware-MRAG Pipeline")
    print("-" * 40)
    try:
        pipeline = SelfAwarePipelineQwen3VL(
            qwen3_vl_wrapper=qwen3_vl,
            retriever=retriever,
            config={
                'uncertainty_threshold': CONFIG['uncertainty_threshold'],
                'max_images': 20,
                'use_improved_estimator': CONFIG['use_improved_estimator'],
                'use_position_fusion': CONFIG['use_position_fusion'],
                'use_attribution': CONFIG['use_attribution'],
                'enable_multimodal_output': False,
            }
        )
        print("✅ Self-Aware-MRAG Pipeline创建成功")
        print(f"  - 不确定性阈值: {CONFIG['uncertainty_threshold']}")
        print(f"  - 改进版估计器: {CONFIG['use_improved_estimator']}")
        print(f"  - 位置融合: {CONFIG['use_position_fusion']}")
    except Exception as e:
        print(f"❌ Pipeline创建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. 运行测试
    print("\n5. 运行Self-Aware-MRAG测试")
    print("-" * 40)

    results = []
    start_time = time.time()
    correct = 0

    for i, sample in enumerate(samples):
        print(f"\r进度: {i+1}/{len(samples)}", end='', flush=True)
        try:
            # 运行pipeline
            result = pipeline.run_single(sample)
            results.append(result)

            # 检查答案（改进的匹配）
            answer = result.get('answer', '').strip().lower()
            golden_answers = sample['golden_answers']

            # 提取第一个词作为答案
            answer_words = answer.split()
            if answer_words:
                answer = answer_words[0]

            # 改进的匹配逻辑
            is_correct = False
            for golden in golden_answers:
                golden = golden.lower().strip()
                golden_words = golden.split()

                # 1. 精确匹配
                if answer == golden:
                    is_correct = True
                    break

                # 2. 第一个词匹配
                if golden_words and answer == golden_words[0]:
                    is_correct = True
                    break

                # 3. 包含匹配（只对较长的词）
                if len(answer) > 2 and answer in golden:
                    is_correct = True
                    break

            if is_correct:
                correct += 1

            # 打印第一个样本的详细信息
            if i == 0:
                print(f"\n\n第一个样本详细结果:")
                print(f"  问题: {sample['question']}")
                print(f"  标准答案: {golden_answers[:3]}")
                print(f"  原始生成答案: {result.get('answer', '').strip()!r}")
                print(f"  提取答案: {answer!r}")
                print(f"  是否正确: {is_correct}")
                if result.get('uncertainty') is not None:
                    uncertainty = result['uncertainty']
                    if isinstance(uncertainty, dict):
                        total_unc = uncertainty.get('total', uncertainty.get('uncertainty', 0))
                        print(f"  不确定性: {total_unc:.4f}")
                    else:
                        print(f"  不确定性: {uncertainty:.4f}")
                if result.get('retrieved_docs'):
                    print(f"  检索文档数: {len(result['retrieved_docs'])}")

        except Exception as e:
            print(f"\n样本 {i} 处理失败: {e}")
            import traceback
            traceback.print_exc()
            results.append({'answer': '', 'retrieved_docs': []})

    elapsed_time = time.time() - start_time
    accuracy = correct / len(results) * 100 if results else 0
    print(f"\n\n完成! 耗时: {elapsed_time:.2f}s")
    print(f"准确率: {accuracy:.1f}% ({correct}/{len(results)})")

    # 6. 评估指标
    print("\n6. 评估指标")
    print("-" * 40)

    # 准备评估数据
    formatted_results = []
    for i, r in enumerate(results):
        if i < len(samples):  # 确保索引不越界
            # 提取第一个词
            answer = r.get('answer', '').strip().lower()
            answer_words = answer.split()
            if answer_words:
                answer = answer_words[0]

            formatted_results.append({
                'answer': answer,
                'golden_answers': samples[i]['golden_answers'],
                'retrieved_docs': r.get('retrieved_docs', [])
            })

    try:
        metrics = evaluate_comprehensive_metrics(formatted_results)
        print(f"  EM: {metrics.get('em', 0):.4f}")
        print(f"  F1: {metrics.get('avg_F1', 0):.4f}")
        print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"  Retrieval Rate: {metrics.get('retrieval_rate', 0):.4f}")
        print(f"  Recall@5: {metrics.get('retrieval_recall_top5', 0):.4f}")
        print(f"  Faithfulness: {metrics.get('avg_Faithfulness', 0):.4f}")
        print(f"  Attribution Precision: {metrics.get('avg_Attribution_Precision', 0):.4f}")
    except Exception as e:
        print(f"  评估失败: {e}")

    # 7. 保存结果
    print("\n7. 保存结果")
    print("-" * 40)
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # 保存详细结果
    results_file = os.path.join(CONFIG['output_dir'], 'self_aware_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'config': CONFIG,
            'metrics': metrics if 'metrics' in locals() else {},
            'results': results,
            'samples': samples,
            'accuracy': accuracy,
            'correct_count': correct,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)

    print(f"✅ 结果已保存到: {results_file}")

    print("\n" + "="*80)
    print("Self-Aware-MRAG测试完成!")
    print("="*80)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"准确率: {accuracy:.1f}%")

if __name__ == '__main__':
    main()