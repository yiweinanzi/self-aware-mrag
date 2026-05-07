#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行所有Baseline对比实验 - 使用项目中已有完整实现
基于用户反馈，使用项目中的完整baseline实现，不做简化

七个方法：
1. Self-Aware-MRAG (Ours) - 有不确定性估计
2. MuRAG - 使用已有实现
3. VisRAG - 使用已有实现
4. ViDoRAG - 使用已有实现
5. RagVL - 使用已有实现
6. SAM-RAG - 使用已有实现
7. mR²AG - 使用已有实现

七个核心指标：
1. EM (Exact Match)
2. F1 Score
3. Recall@5
4. VQA-Score
5. Faithfulness
6. Attribution Precision
7. Position Bias Score
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime
from typing import List, Dict, Any, Optional

# 添加FlashRAG路径
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

# 导入必要的模块
import numpy as np
from flashrag.dataset.okvqa_dataset_simple import OKVQADatasetSimple
from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
from flashrag.retriever import DenseRetriever
from flashrag.pipeline.self_aware_pipeline_qwen3vl import SelfAwarePipelineQwen3VL
from flashrag.utils.comprehensive_evaluator import evaluate_comprehensive_metrics

# 导入已有的baseline实现
from experiments.baselines.murag_enhanced import MuRAGEnhanced
from experiments.baselines.visrag_enhanced import VisRAGEnhanced
from experiments.baselines.vidorag_pipeline import ViDoRAGAdapter
from experiments.baselines.ragvl_enhanced import RagVLEnhanced
from experiments.baselines.sam_rag_enhanced import SAMRAGEnhanced
from experiments.baselines.mr2ag_enhanced import MR2AGEnhanced

# ============================================================================
# 配置参数
# ============================================================================

CONFIG = {
    # 数据集配置
    'dataset_name': 'okvqa',
    'data_dir': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/VQA',
    'split': 'val',
    'max_samples': 10,  # 10个样本测试
    'load_images': True,  # 加载图像

    # 模型配置
    'qwen3_vl_path': '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
    'torch_dtype': 'bfloat16',
    'temperature': 0.1,  # 修复temperature=0.0的问题

    # 检索器配置
    'faiss_index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
    'corpus_path': '/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl',
    'retrieval_model_path': '/data0/home/zqwang/ACL/models/bge-large-en-v1.5',
    'retrieval_topk': 5,

    # CLIP模型配置 - 修复CLIP路径问题
    'clip_model_path': '/data0/home/zqwang/ACL/models/clip-vit-large-patch14-336',

    # 不确定性配置（仅用于Self-Aware-MRAG）
    'uncertainty_threshold': 0.43,
    'use_improved_estimator': True,
    'text_weight': 0.4,
    'visual_weight': 0.3,
    'alignment_weight': 0.3,

    # 输出配置
    'output_dir': '/data0/home/zqwang/ACL/FlashRAG/experiments/results_okvqa_baselines_proper',
}

# ============================================================================
# 数据加载
# ============================================================================

def load_dataset(data_dir, max_samples=None):
    """加载OK-VQA数据集"""
    print(f"加载数据集: OK-VQA")
    print(f"数据路径: {data_dir}")
    print(f"最大样本数: {max_samples if max_samples else '全部'}")

    try:
        dataset = OKVQADatasetSimple({
            'data_dir': data_dir,
            'split': 'val',
            'load_images': True,
        })

        # 转换为样本列表
        samples = []
        for i in range(min(max_samples if max_samples else len(dataset), len(dataset))):
            item = dataset[i]
            sample = {
                'id': item['id'],
                'question': item['question'],
                'image': item.get('image'),
                'answer': item.get('answer', ''),
                'golden_answers': item['golden_answers']
            }
            samples.append(sample)

        print(f"✅ 成功加载 {len(samples)} 个样本")
        print(f"   图像加载: {all(s.get('image') is not None for s in samples)}")
        return samples

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return []

# ============================================================================
# 模型初始化
# ============================================================================

def init_qwen3_vl(model_path):
    """初始化Qwen3-VL"""
    print(f"初始化Qwen3-VL: {model_path}")
    wrapper = create_qwen3_vl_wrapper(model_path=model_path, device="cuda")
    print("✅ Qwen3-VL加载成功")
    return wrapper

def init_retriever(config):
    """初始化检索器"""
    print("初始化检索器...")

    retriever_config = {
        'index_path': config['faiss_index_path'],
        'corpus_path': config['corpus_path'],
        'retrieval_method': 'e5',
        'retrieval_model_path': config['retrieval_model_path'],
        'retrieval_query_max_length': 512,
        'retrieval_pooling_method': 'mean',
        'retrieval_use_fp16': True,
        'retrieval_batch_size': 128,
        'retrieval_topk': config['retrieval_topk'],
        'save_retrieval_cache': False,
        'use_retrieval_cache': False,
        'retrieval_cache_path': None,
        'use_reranker': False,
        'use_sentence_transformer': False,
        'faiss_gpu': False,
        'instruction': '',
    }

    try:
        retriever = DenseRetriever(retriever_config)
        print("✅ 检索器加载成功")
        return retriever
    except Exception as e:
        print(f"❌ 检索器加载失败: {e}")
        return None

# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 80)
    print("OK-VQA 七个Baseline方法对比测试 (使用完整实现)")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"样本数: {CONFIG['max_samples']}")
    print(f"数据集: {CONFIG['dataset_name'].upper()}")
    print()

    # 创建输出目录
    output_dir = os.path.join(CONFIG['output_dir'])
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载数据
    print("=" * 80)
    print("1. 加载数据集")
    print("-" * 40)
    samples = load_dataset(CONFIG['data_dir'], CONFIG['max_samples'])

    if not samples:
        print("❌ 数据加载失败，退出")
        return

    # 2. 初始化模型和检索器
    print("\n" + "=" * 80)
    print("2. 初始化模型和检索器")
    print("-" * 40)
    qwen3_vl = init_qwen3_vl(CONFIG['qwen3_vl_path'])
    retriever = init_retriever(CONFIG)

    # 3. 初始化baseline方法
    print("\n" + "=" * 80)
    print("3. 初始化七个Baseline方法")
    print("-" * 40)

    methods = {}

    # 1. Self-Aware-MRAG (我们的方法，有不确定性估计)
    try:
        methods['Self-Aware-MRAG (Ours)'] = SelfAwarePipelineQwen3VL(
            qwen3_vl_wrapper=qwen3_vl,
            retriever=retriever,
            config={
                'uncertainty_threshold': CONFIG['uncertainty_threshold'],
                'use_improved_estimator': CONFIG['use_improved_estimator'],
                'use_position_fusion': True,
                'use_attribution': True,
                'enable_multimodal_output': False,
                'clip_model_path': CONFIG['clip_model_path'],
                'retrieval_topk': CONFIG['retrieval_topk'],
                'thinking': False,
                'max_images': 20,
            }
        )
        print("✅ Self-Aware-MRAG (Ours) 初始化成功")
    except Exception as e:
        print(f"❌ Self-Aware-MRAG 初始化失败: {e}")

    # 2. MuRAG - 使用已有实现
    try:
        methods['MuRAG'] = MuRAGEnhanced(
            qwen3_vl=qwen3_vl,
            retriever=retriever,
            config={
                **CONFIG,
                'max_evidence': 5,
                'fusion_strategy': 'vote',
                'parallel_encoding': True,
            }
        )
        print("✅ MuRAG 初始化成功")
    except Exception as e:
        print(f"❌ MuRAG 初始化失败: {e}")

    # 3. VisRAG - 使用已有实现
    try:
        methods['VisRAG'] = VisRAGEnhanced(
            qwen3_vl=qwen3_vl,
            retriever=retriever,
            config={
                **CONFIG,
                'use_reranker': True,
                'rerank_topk': 3,
                'vision_first': True,
                'max_dpr': 5,
            }
        )
        print("✅ VisRAG 初始化成功")
    except Exception as e:
        print(f"❌ VisRAG 初始化失败: {e}")

    # 4. ViDoRAG - 使用已有实现
    try:
        methods['ViDoRAG'] = ViDoRAGAdapter(
            qwen3_vl=qwen3_vl,
            retriever=retriever,
            config={
                **CONFIG,
                'num_agents': 2,
                'max_iterations': 3,
                'document_selector': True,
                'visual_reasoning': True,
            }
        )
        print("✅ ViDoRAG 初始化成功")
    except Exception as e:
        print(f"❌ ViDoRAG 初始化失败: {e}")

    # 5. RagVL - 使用已有实现
    try:
        methods['RagVL'] = RagVLEnhanced(
            qwen3_vl=qwen3_vl,
            retriever=retriever,
            config={
                **CONFIG,
                'use_mllm_reranker': True,
                'rerank_topk': 3,
                'coarse_topk': 20,
                'retrieval_strategy': 'two-stage',
            }
        )
        print("✅ RagVL 初始化成功")
    except Exception as e:
        print(f"❌ RagVL 初始化失败: {e}")

    # 6. SAM-RAG - 使用已有实现
    try:
        methods['SAM-RAG'] = SAMRAGEnhanced(
            qwen3_vl=qwen3_vl,
            retriever=retriever,
            config={
                **CONFIG,
                'batch_size': 5,
                'max_batches': 4,
                'relevance_threshold': 0.5,
                'adaptive_retrieval': True,
                'quality_assessment': True,
            }
        )
        print("✅ SAM-RAG 初始化成功")
    except Exception as e:
        print(f"❌ SAM-RAG 初始化失败: {e}")

    # 7. mR²AG - 使用已有实现
    try:
        methods['mR²AG'] = MR2AGEnhanced(
            qwen3_vl=qwen3_vl,
            retriever=retriever,
            config={
                **CONFIG,
                'enable_reflection': True,
                'reflection_strategy': 'two-stage',
                'paragraph_level': True,
                'hierarchical_scoring': True,
                'confidence_threshold': 0.3,
            }
        )
        print("✅ mR²AG 初始化成功")
    except Exception as e:
        print(f"❌ mR²AG 初始化失败: {e}")

    print(f"\n✅ 初始化了 {len(methods)} 个Baseline方法")

    # 4. 运行所有方法
    print("\n" + "=" * 80)
    print("4. 运行所有Baseline方法")
    print("-" * 40)

    all_results = {}

    for method_name, pipeline in methods.items():
        print(f"\n{'='*60}")
        print(f"运行方法: {method_name}")
        print(f"{'='*60}")

        start_time = time.time()

        try:
            # 运行pipeline
            if hasattr(pipeline, 'run_batch'):
                # 使用run_batch方法
                results = pipeline.run_batch(samples)
            elif hasattr(pipeline, 'run'):
                # 使用run方法
                results = pipeline.run(samples)
            else:
                print(f"❌ {method_name} 没有可用的运行方法")
                continue

            elapsed_time = time.time() - start_time

            # 评估
            print(f"\n评估 {method_name}...")
            metrics = evaluate_comprehensive_metrics(results)

            # 保存结果
            method_result = {
                'method': method_name,
                'config': CONFIG,
                'metrics': metrics,
                'results': results,
                'elapsed_time': elapsed_time,
                'samples_per_second': len(results) / elapsed_time if elapsed_time > 0 else 0,
                'timestamp': datetime.now().isoformat()
            }

            all_results[method_name] = method_result

            print(f"\n✅ {method_name} 完成:")
            print(f"   耗时: {elapsed_time:.1f}秒")
            print(f"   速度: {method_result['samples_per_second']:.2f} 样本/秒")

        except Exception as e:
            print(f"\n❌ {method_name} 运行失败: {e}")
            import traceback
            traceback.print_exc()

    # 5. 保存汇总结果
    print("\n\n5. 保存汇总结果")
    print("-" * 40)

    # 创建汇总报告
    summary = {
        'config': CONFIG,
        'timestamp': datetime.now().isoformat(),
        'total_samples': len(samples),
        'methods': {}
    }

    # 提取关键指标
    print("\n关键指标汇总:")
    print("-" * 80)
    print(f"{'方法':<20} {'EM':<8} {'F1':<8} {'Recall@5':<10} {'VQA-Score':<11} {'Faithfulness':<12} {'Attribution':<12} {'PositionBias':<12}")
    print("-" * 80)

    for method_name, result in all_results.items():
        metrics = result.get('metrics', {})

        # 保存到summary
        summary['methods'][method_name] = {
            'EM': metrics.get('avg_accuracy', 0),
            'F1': metrics.get('avg_F1', 0),
            'Recall@5': metrics.get('avg_Recall@5', 0),
            'VQA-Score': metrics.get('avg_VQA_Score', 0) * 100,  # 转换为百分制
            'Faithfulness': metrics.get('avg_Faithfulness', 0),
            'Attribution Precision': metrics.get('avg_Attribution_Precision', 0),
            'Position Bias Score': metrics.get('avg_Position_Bias_Score', 0),
        }

        # 打印表格
        print(f"{method_name:<20} "
              f"{metrics.get('avg_accuracy', 0):<8.4f} "
              f"{metrics.get('avg_F1', 0):<8.4f} "
              f"{metrics.get('avg_Recall@5', 0):<10.4f} "
              f"{metrics.get('avg_VQA_Score', 0)*100:<11.4f} "
              f"{metrics.get('avg_Faithfulness', 0):<12.4f} "
              f"{metrics.get('avg_Attribution_Precision', 0):<12.4f} "
              f"{metrics.get('avg_Position_Bias_Score', 0):<12.4f}")

    # 保存汇总文件
    summary_file = os.path.join(output_dir, 'all_baselines_summary_proper.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 汇总结果已保存到: {summary_file}")

    # 6. 完成提示
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
    print(f"总样本数: {len(samples)}")
    print(f"完成方法数: {len(all_results)}")
    print(f"输出目录: {output_dir}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()