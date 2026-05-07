#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OK-VQA Baselines对比实验 - 基于消融实验框架
OK-VQA Baselines Comparison - Based on Ablation Study Framework

七个方法对比:
1. Self-Aware-MRAG (Ours)
2. MuRAG
3. VisRAG
4. ViDoRAG
5. RagVL
6. SAM-RAG
7. mR²AG

评价指标:
- 准确率 (Accuracy)
- 检索率 (Retrieval Rate)
- F1
- Recall@5
- Faithfulness
- Attribution Precision
- Position Bias Score
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import torch
import gc
import argparse
import numpy as np

# 添加FlashRAG路径
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

# 导入增强评估指标
from flashrag.utils.comprehensive_evaluator import evaluate_comprehensive_metrics
from experiments.enhanced_evaluation import enhance_evaluation_stats, enhance_results_saving

# ============================================================================
# GPU和内存管理 (基于多个成功版本)
# ============================================================================

def clear_gpu_memory():
    """清理GPU内存 (来自run_real_model_ablation.py)"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"🧹 GPU内存已清理: 分配={memory_allocated:.2f}GB, 保留={memory_reserved:.2f}GB")
    except Exception as e:
        print(f"⚠️ GPU内存清理失败: {e}")

def setup_device(num_gpus=None):
    """设置GPU设备 (基于多个成功版本)"""
    if not torch.cuda.is_available():
        print("⚠️ CUDA不可用，使用CPU")
        return ['cpu']

    device_count = torch.cuda.device_count()
    print(f"✅ 检测到 {device_count} 个GPU")

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}, {props.total_memory/1024**3:.1f}GB")

    if num_gpus is None:
        num_gpus = min(2, device_count)  # 推荐使用2GPU以获得最佳性能

    num_gpus = min(num_gpus, device_count)

    if num_gpus == 0:
        return ['cpu']

    devices = [f'cuda:{i}' for i in range(num_gpus)]
    print(f"🚀 使用GPU: {devices}")

    return devices

# ============================================================================
# 配置参数 - 基于59%准确率成功实验
# ============================================================================

def create_config(args):
    """根据命令行参数创建配置"""
    config = {
        # 数据集配置 (基于run_ablation_study_okvqa.py)
        'dataset_name': args.dataset,
        'data_dir': args.data_dir,
        'split': args.split,
        'max_samples': args.max_samples,
        'load_images': True,

        # 实验配置
        'output_dir': args.output_dir,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),

        # 模型配置
        'model_path': args.model_path,
        'torch_dtype': args.torch_dtype,
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,

        # GPU配置 (基于run_ablation_study_100samples_2gpu.py)
        'use_multi_gpu': args.use_multi_gpu,
        'num_gpus': args.num_gpus,
        'batch_size_per_gpu': 1,

        # 检索配置 (基于59%准确率实验)
        'faiss_index_path': args.faiss_index_path,
        'corpus_path': args.corpus_path,
        'retrieval_model_path': args.retrieval_model_path,
        'retrieval_topk': args.retrieval_topk,

        # CLIP多模态检索配置 (基于59%准确率实验)
        'clip_model_path': args.clip_model_path,
        'clip_index_path': args.clip_index_path,
        'use_multimodal_retrieval': args.use_multimodal_retrieval,

        # 多模态检索权重 (BGE 60% + CLIP 40%)
        'text_retrieval_weight': args.text_retrieval_weight,
        'visual_retrieval_weight': args.visual_retrieval_weight,

        # 不确定性权重配置 (基于59%准确率实验)
        'text_weight': args.text_weight,
        'visual_weight': args.visual_weight,
        'alignment_weight': args.alignment_weight,

        # 不确定性阈值 (基于59%准确率实验的最佳配置)
        'uncertainty_threshold': args.uncertainty_threshold,

        # 不确定性估计器配置 (基于59%准确率实验)
        'use_improved_estimator': args.use_improved_estimator,

        # 评估配置 (基于run_ablation_study_okvqa.py)
        'save_detailed_results': args.save_detailed_results,
        'save_sample_results': args.save_sample_results,
        'enable_complete_metrics': args.enable_complete_metrics,
    }

    return config

# ============================================================================
# Baseline方法配置 - 7个对比方法
# ============================================================================

BASELINE_METHODS = [
    {
        'name': 'Self-Aware-MRAG',
        'description': 'Our Self-Aware Multimodal RAG system',
        'class': 'SelfAwarePipelineQwen3VL',
        'module': 'flashrag.pipeline.self_aware_pipeline_qwen3vl',
        'config': {
            'uncertainty_threshold': 0.43,
            'use_improved_estimator': True,
            'use_position_fusion': True,
            'use_attribution': True,
            'thinking': False,
            'clip_model_path': '/data0/home/zqwang/ACL/models/clip-vit-large-patch14-336'
        }
    },
    {
        'name': 'MuRAG',
        'description': 'Multimodal Retrieval-Augmented Generation',
        'class': 'MuRAGEnhanced',
        'module': 'experiments.baselines.murag_enhanced'
    },
    {
        'name': 'VisRAG',
        'description': 'Visual RAG with BGE reranking',
        'class': 'VisRAGEnhanced',
        'module': 'experiments.baselines.visrag_enhanced'
    },
    {
        'name': 'ViDoRAG',
        'description': 'Video and Document RAG with multi-agent system',
        'class': 'ViDoRAGPipeline',
        'module': 'experiments.baselines.vidorag_pipeline'
    },
    {
        'name': 'RagVL',
        'description': 'RAG for Vision-Language tasks with MLLM reranking',
        'class': 'RagVLEnhanced',
        'module': 'experiments.baselines.ragvl_enhanced'
    },
    {
        'name': 'SAM-RAG',
        'description': 'Self-Aware Memory RAG',
        'class': 'SAMRAGAdapted',
        'module': 'experiments.baselines.samrag_adapted'
    },
    {
        'name': 'mR²AG',
        'description': 'multi-step Reflection and Refinement Augmented Generation',
        'class': 'MR2AGFixed',
        'module': 'experiments.baselines.mr2ag_enhanced'
    }
]

# ============================================================================
# 数据加载 (基于run_ablation_study_okvqa.py)
# ============================================================================

def load_dataset(config):
    """加载数据集"""
    print("="*80)
    print("1. 加载数据集")
    print("="*80)

    try:
        if config['dataset_name'] == 'okvqa':
            from flashrag.dataset.okvqa_dataset_simple import OKVQADatasetSimple

            dataset = OKVQADatasetSimple({
                'data_dir': config['data_dir'],
                'split': config['split'],
                'load_images': config['load_images'],
            })

        elif config['dataset_name'] == 'mragbench':
            # 支持MRAG-Bench数据集 (基于59%准确率实验)
            import datasets
            dataset_dict = datasets.load_from_disk('/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MRAG-Bench/raw')
            test_data = dataset_dict['test']

            # 限制样本数
            if config['max_samples']:
                test_data = test_data.select(range(min(config['max_samples'], len(test_data))))

            # 转换为统一格式
            data_list = []
            for item in test_data:
                data_list.append({
                    'question': item['question'],
                    'image': item['image'],
                    'golden_answers': item['answer'],
                    'image_id': item.get('image_id', ''),
                })

            # 创建简单的数据集对象
            class SimpleDataset:
                def __init__(self, data):
                    self.data = data
                def __len__(self):
                    return len(self.data)
                def __getitem__(self, idx):
                    return self.data[idx]

            dataset = SimpleDataset(data_list)

        else:
            raise ValueError(f"不支持的数据集: {config['dataset_name']}")

        # 限制样本数
        if config['max_samples'] and len(dataset.data) > config['max_samples']:
            dataset.data = dataset.data[:config['max_samples']]

        print(f"✅ 数据集加载完成: {len(dataset.data)} 样本")
        print(f"   数据集: {config['dataset_name']}")
        print(f"   图像加载: {config['load_images']}")

        # 检查数据样本
        if dataset.data:
            sample = dataset.data[0]
            print(f"   样本示例: {sample.get('question', '')[:50]}...")
            if 'golden_answers' in sample:
                answers = sample['golden_answers']
                if isinstance(answers, list):
                    print(f"   答案示例: {answers[:3]}")
                else:
                    print(f"   答案示例: {answers}")
            if 'image_id' in sample:
                print(f"   图像ID: {sample['image_id']}")

        return dataset

    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# 模型和检索器初始化 (整合多个成功版本)
# ============================================================================

def init_models_and_retriever(config):
    """初始化模型和检索器"""
    print("\n" + "="*80)
    print("2. 初始化模型和检索器")
    print("="*80)

    # 初始化Qwen3-VL
    try:
        print("初始化Qwen3-VL...")
        from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper

        qwen3_vl = create_qwen3_vl_wrapper(
            model_path=config['model_path'],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print("✅ Qwen3-VL初始化成功")
    except Exception as e:
        print(f"❌ Qwen3-VL初始化失败: {e}")
        return None, None

    # 初始化检索器
    try:
        print("初始化检索器...")
        from flashrag.retriever import DenseRetriever

        # 基础BGE检索器 (使用成功配置)
        bge_retriever = DenseRetriever({
            'retrieval_method': 'dense',
            'retrieval_model_path': config['retrieval_model_path'],
            'index_path': config['faiss_index_path'],
            'corpus_path': config['corpus_path'],
            'retrieval_topk': config['retrieval_topk'],
            'save_retrieval_cache': False,
            'use_retrieval_cache': False,
            'retrieval_cache_path': None,
            'embedding_dim': 1024,  # BGE-large embedding dimension
            'use_reranker': False,  # 先不用reranker避免配置复杂
            'retrieval_query_max_length': 512,
            'retrieval_pooling_method': 'mean',
            'retrieval_use_fp16': False,
            'retrieval_batch_size': 32,
            'instruction': '',
            'use_sentence_transformer': True,
            'faiss_gpu': False,  # 先用CPU测试
            'silent_retrieval': False,
            'device': 'cpu',
            'rerank_model_name': None,
            'rerank_topk': 5,
            'rerank_max_length': 512,
            'rerank_batch_size': 32,
        })

        if config['use_multimodal_retrieval']:
            print("🔄 初始化多模态融合检索器...")

            try:
                # CLIP检索器
                clip_retriever = DenseRetriever({
                    'retrieval_method': 'clip',
                    'retrieval_model_path': config['clip_model_path'],
                    'index_path': config['clip_index_path'],
                    'corpus_path': config['corpus_path'],
                    'retrieval_query_max_length': 512,
                    'retrieval_use_fp16': True,
                    'retrieval_batch_size': 128,
                    'retrieval_topk': config['retrieval_topk'],
                    'retrieval_pooling_method': 'mean',
                    'save_retrieval_cache': False,
                    'use_retrieval_cache': False,
                    'retrieval_cache_path': None,
                    'use_reranker': True,
                    'rerank_model_path': '/data0/home/zqwang/ACL/models/bge-reranker-v2-m3',
                    'rerank_model_name': '/data0/home/zqwang/ACL/models/bge-reranker-v2-m3',
                    'rerank_topk': 5,
                    'rerank_max_length': 512,
                    'rerank_batch_size': 32,
                    'device': 'cuda',
                    'faiss_gpu': True,
                    'instruction': 'search query:',
                })

                # 多模态融合检索器 (基于59%准确率实验)
                from flashrag.retriever.multimodal_retriever import SelfAwareMultimodalRetriever

                multimodal_config = {
                    'retrieval_topk': config['retrieval_topk'],
                    'text_weight': config['text_retrieval_weight'],
                    'visual_weight': config['visual_retrieval_weight'],
                    'fusion_method': 'weighted',
                }

                retriever = SelfAwareMultimodalRetriever(
                    config=multimodal_config,
                    text_retriever=bge_retriever,
                    visual_retriever=clip_retriever
                )

                print(f"✅ 多模态融合检索器初始化成功 (BGE {config['text_retrieval_weight']*100:.0f}% + CLIP {config['visual_retrieval_weight']*100:.0f}%)")

            except Exception as e:
                print(f"⚠️ 多模态检索器初始化失败，降级到BGE: {e}")
                retriever = bge_retriever
        else:
            retriever = bge_retriever
            print("✅ BGE文本检索器初始化成功")

    except Exception as e:
        print(f"⚠️ 检索器初始化失败: {e}")
        print("使用模拟检索器...")
        retriever = None

    return qwen3_vl, retriever

# ============================================================================
# 单个变体实验运行 (基于多个成功版本)
# ============================================================================

def run_baseline_method(method_config, dataset, qwen3_vl, retriever, config):
    """运行单个baseline方法"""
    method_name = method_config['name']

    # 设置日志文件
    log_file = Path(config['output_dir']) / f"baselines_{config['timestamp']}.log"

    print(f"\n{'='*60}")
    print(f"运行方法: {method_name}")
    print(f"{'='*60}")
    print(f"📋 实时日志: {log_file}")

    # 写入日志开始
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n[{datetime.now().strftime('%H:%M:%S')}] 开始方法: {method_name}\n")
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] 描述: {method_config['description']}\n")
        f.flush()

    # 创建pipeline
    try:
        # 动态导入
        module_name = method_config['module']
        class_name = method_config['class']

        module = __import__(module_name, fromlist=[class_name])
        pipeline_class = getattr(module, class_name)

        if method_name == "Self-Aware-MRAG":
            # Self-Aware-MRAG需要特殊配置
            pipeline = pipeline_class(
                qwen3_vl,
                retriever=retriever,
                config=method_config.get('config', {})
            )
        elif method_name == "VisRAG":
            # VisRAG需要本地BGE reranker路径
            visrag_config = config.copy()
            visrag_config['bge_reranker_path'] = config.get('bge_reranker_path', '/data0/home/zqwang/ACL/models/bge-reranker-v2-m3')
            pipeline = pipeline_class(qwen3_vl, retriever, config=visrag_config)
        else:
            # 其他方法使用通用配置
            pipeline = pipeline_class(qwen3_vl, retriever, config)

        print(f"✅ {method_name} 创建成功")

        # 写入日志
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Pipeline创建成功\n")
            f.flush()

    except Exception as e:
        print(f"❌ Pipeline创建失败: {e}")

        # 写入错误日志
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Pipeline创建失败: {e}\n")
            f.flush()
        import traceback
        traceback.print_exc()
        return None

    # 运行实验
    try:
        start_time = time.time()

        # 写入开始运行日志
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] 开始运行实验...\n")
            f.flush()

        # 根据不同pipeline类型使用不同运行方法
        if hasattr(pipeline, 'run'):
            # Self-Aware-MRAG等有run方法
            results = pipeline.run(dataset, verbose=False)
            # 计算correct字段（如果pipeline没有计算）
            from experiments.baselines.evaluation_helper import evaluate_answer_correctness
            for result in results:
                if not result.get('correct', False) and 'answer' in result and 'golden_answers' in result:
                    result['correct'] = evaluate_answer_correctness(
                        result['answer'],
                        result['golden_answers']
                    )
        else:
            # 其他baseline需要逐个运行样本
            results = []
            for i, sample in enumerate(dataset):
                print(f"\r进度: {i+1}/{len(dataset)}", end='', flush=True)
                try:
                    result = pipeline.run_single(sample)
                    # 计算correct字段（如果pipeline没有计算）
                    if not result.get('correct', False) and 'answer' in result and 'golden_answers' in result:
                        from experiments.baselines.evaluation_helper import evaluate_answer_correctness
                        result['correct'] = evaluate_answer_correctness(
                            result['answer'],
                            result['golden_answers']
                        )
                    results.append(result)
                except Exception as e:
                    print(f"\n[ERROR] 样本 {i} 处理失败: {e}")
                    # 创建默认结果
                    results.append({
                        'question': sample.get('question', ''),
                        'answer': '',
                        'golden_answers': sample.get('golden_answers', []),
                        'retrieved_docs': [],
                        'correct': False,
                        'retrieved': False
                    })
            print()  # 换行

        end_time = time.time()

        # 计算基础指标
        correct_count = sum(1 for r in results if r.get('correct', False))
        total_count = len(results)
        accuracy = correct_count / total_count if total_count > 0 else 0
        retrieval_rate = sum(1 for r in results if r.get('retrieved', False)) / total_count

        # 收集基础统计
        base_stats = {
            'variant_name': method_name,
            'variant_description': method_config['description'],
            'config': method_config.get('config', {}),
            'total_samples': total_count,
            'correct_samples': correct_count,
            'accuracy': accuracy,
            'retrieval_rate': retrieval_rate,
            'execution_time': end_time - start_time,
            'timestamp': datetime.now().isoformat(),
            'seconds_per_sample': (end_time - start_time) / total_count if total_count > 0 else 0
        }

        # 应用增强评估指标
        enhanced_stats = enhance_evaluation_stats(base_stats, results)

        print(f"\n✅ 完成: {method_name}")
        print(f"   准确率: {enhanced_stats['accuracy']*100:.2f}% ({correct_count}/{total_count})")
        print(f"   检索率: {enhanced_stats['retrieval_rate']*100:.1f}%")
        print(f"   耗时: {end_time - start_time:.1f}秒 ({enhanced_stats['seconds_per_sample']:.2f}s/样本)")

        # 写入完成日志
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 实验完成\n")
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] 准确率: {enhanced_stats['accuracy']*100:.2f}% ({correct_count}/{total_count})\n")
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] 检索率: {enhanced_stats['retrieval_rate']*100:.1f}%\n")
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] 耗时: {end_time - start_time:.1f}秒 ({enhanced_stats['seconds_per_sample']:.2f}s/样本)\n")
            f.flush()

        return {
            'results': results,
            'stats': enhanced_stats
        }

    except Exception as e:
        print(f"❌ 实验运行失败: {e}")

        # 写入错误日志
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 实验运行失败: {e}\n")
            f.flush()
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# 结果保存 (整合多个成功版本的功能)
# ============================================================================

def save_results(all_results, config):
    """保存实验结果"""
    print(f"\n{'='*80}")
    print("4. 保存实验结果")
    print(f"{'='*80}")

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = config['timestamp']
    results_file = output_dir / f"unified_ablation_results_{timestamp}.json"
    report_file = output_dir / f"unified_ablation_report_{timestamp}.md"

    # 准备保存的数据
    save_data = {
        'experiment_info': {
            'dataset': config['dataset_name'],
            'samples': config['max_samples'],
            'timestamp': timestamp,
            'config': config,
            'based_on': 'Integrated configuration from 59% accuracy experiments',
        },
        'variants_summary': [],
        'detailed_results': {},
    }

    # 处理每个变体的结果
    for variant_result in all_results:
        if variant_result is None:
            continue

        stats = variant_result['stats']
        results = variant_result['results']

        # 添加到汇总
        save_data['variants_summary'].append({
            'variant_name': stats['variant_name'],
            'variant_description': stats['variant_description'],
            'accuracy': stats['accuracy'],
            'retrieval_rate': stats['retrieval_rate'],
            'execution_time': stats['execution_time'],
            'seconds_per_sample': stats['seconds_per_sample'],
            'config': stats['config']
        })

        # 保存详细结果
        if config['save_detailed_results']:
            sample_limit = min(50, len(results)) if config['save_sample_results'] else 0
            save_data['detailed_results'][stats['variant_name']] = {
                'stats': stats,
                'sample_results': results[:sample_limit] if sample_limit > 0 else []
            }

    # 应用增强评估指标到结果保存
    save_data = enhance_results_saving(save_data, all_results)

    # 写入JSON文件
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"✅ JSON结果已保存: {results_file}")

        # 生成Markdown报告
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# 统一消融实验报告\n\n")
            f.write(f"**实验时间**: {timestamp}\n")
            f.write(f"**数据集**: {config['dataset_name']}\n")
            f.write(f"**样本数**: {config['max_samples']}\n")
            f.write(f"**模型**: Qwen3-VL-8B-Instruct\n")
            f.write(f"**多模态检索**: BGE({config['text_retrieval_weight']*100:.0f}%) + CLIP({config['visual_retrieval_weight']*100:.0f}%)\n")
            f.write(f"**不确定性阈值**: {config['uncertainty_threshold']}\n\n")

            f.write(f"## 消融变体结果\n\n")
            f.write(f"| 变体 | 描述 | 准确率 | 检索率 | F1 | VQA-Score | Recall@5 | Faithfulness | Attribution |\n")
            f.write(f"|------|------|--------|--------|----|----------|----------|-------------|-------------|\n")

            for variant in save_data['variants_summary']:
                description = variant['variant_description'][:30]
                f.write(f"| {variant['variant_name']} | {description} | "
                       f"{variant['accuracy']*100:>6.2f}% | "
                       f"{variant['retrieval_rate']*100:>6.1f}% | "
                       f"{variant.get('F1', 0)*100:>6.2f}% | "
                       f"{variant.get('VQA_Score', 0)*100:>6.2f}% | "
                       f"{variant.get('Recall@5', 0)*100:>6.2f}% | "
                       f"{variant.get('Faithfulness', 0)*100:>6.2f}% | "
                       f"{variant.get('Attribution_Precision', 0)*100:>6.2f}% |\n")

            # 找出最佳结果
            if save_data['variants_summary']:
                best_variant = max(save_data['variants_summary'], key=lambda x: x['accuracy'])
                f.write(f"\n## 🏆 最佳结果\n\n")
                f.write(f"**变体**: {best_variant['variant_name']}\n")
                f.write(f"**准确率**: {best_variant['accuracy']*100:.2f}%\n")
                f.write(f"**描述**: {best_variant['variant_description']}\n")

                # 性能评估
                if best_variant['accuracy'] >= 0.5:
                    f.write(f"🎉 **达到高性能标准！** 准确率超过50%\n")
                elif best_variant['accuracy'] >= 0.4:
                    f.write(f"✅ **性能良好** 准确率超过40%\n")
                elif best_variant['accuracy'] >= 0.3:
                    f.write(f"✅ **性能合格** 准确率超过30%\n")
                else:
                    f.write(f"💡 **优化建议**: 当前准确率{best_variant['accuracy']*100:.1f}%，建议调整配置\n")

        print(f"✅ Markdown报告已保存: {report_file}")

    except Exception as e:
        print(f"❌ 保存结果失败: {e}")

    # 打印汇总表格
    print(f"\n{'='*120}")
    print("📊 实验结果汇总 - 综合评估指标")
    print(f"{'='*120}")
    print(f"{'变体':<25} {'准确率':<8} {'检索率':<8} {'F1':<8} {'VQA':<8} {'R@5':<8} {'Faith':<8} {'Attr':<8} {'耗时(s)':<8}")
    print("-" * 115)

    for variant in save_data['variants_summary']:
        print(f"{variant['variant_name']:<25} "
              f"{variant['accuracy']*100:>6.2f}%  "
              f"{variant['retrieval_rate']*100:>6.1f}%  "
              f"{variant.get('F1', 0)*100:>6.2f}%  "
              f"{variant.get('VQA_Score', 0)*100:>6.2f}%  "
              f"{variant.get('Recall@5', 0)*100:>6.2f}%  "
              f"{variant.get('Faithfulness', 0)*100:>6.2f}%  "
              f"{variant.get('Attribution_Precision', 0)*100:>6.2f}%  "
              f"{variant['execution_time']:>7.1f}")

    # 打印详细指标说明
    print(f"\n📈 指标说明:")
    print(f"   准确率: VQA官方评测准确率")
    print(f"   检索率: 成功检索到知识文档的比例")
    print(f"   F1: Token级别的F1分数")
    print(f"   VQA: VQA官方评测得分")
    print(f"   R@5: Recall@5检索召回率")
    print(f"   Faith: 答案与检索文档的忠实度")
    print(f"   Attr: 答案归因精确度")

    # 与59%基准对比
    print(f"\n{'='*80}")
    print("🎯 性能评估 (基于59%准确率基准)")
    print(f"{'='*80}")

    for variant in save_data['variants_summary']:
        acc = variant['accuracy']
        if acc >= 0.5:
            status = "🎉 达到高性能标准"
        elif acc >= 0.4:
            status = "✅ 性能良好"
        elif acc >= 0.3:
            status = "✅ 性能���格"
        else:
            status = "💡 需要优化"

        print(f"{variant['variant_name']:<35} {acc*100:>6.2f}% - {status}")

    return save_data

# ============================================================================
# 主函数
# ============================================================================

def parse_args():
    """解析命令行参数 (整合所有成功版本的参数)"""
    parser = argparse.ArgumentParser(description='统一消融实验 - 整合所有成功代码')

    # 数据集配置
    parser.add_argument('--dataset', type=str, default='okvqa',
                       choices=['okvqa', 'mragbench'], help='数据集选择')
    parser.add_argument('--data-dir', type=str,
                       default='/data0/home/zqwang/ACL/FlashRAG/flashrag/data/VQA',
                       help='数据集目录')
    parser.add_argument('--split', type=str, default='val', help='数据集split')
    parser.add_argument('--max-samples', type=int, default=100, help='最大样本数')

    # 模型配置
    parser.add_argument('--model-path', type=str,
                       default='/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
                       help='模型路径')
    parser.add_argument('--torch-dtype', type=str, default='float16',
                       choices=['float16', 'bfloat16', 'float32'], help='PyTorch数据类型')
    parser.add_argument('--max-new-tokens', type=int, default=30, help='最大生成token数 (兼顾简洁性和完整性)')
    parser.add_argument('--temperature', type=float, default=0.01, help='生成温度')

    # 检索配置
    parser.add_argument('--retrieval-topk', type=int, default=5, help='检索topk')
    parser.add_argument('--faiss-index-path', type=str,
                       default='/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
                       help='FAISS索引路径')
    parser.add_argument('--corpus-path', type=str,
                       default='/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl',
                       help='语料库路径')
    parser.add_argument('--retrieval-model-path', type=str,
                       default='/data0/home/zqwang/ACL/models/bge-large-en-v1.5',
                       help='检索模型路径')

    # 多模态检索配置 (基于59%准确率实验)
    parser.add_argument('--use-multimodal-retrieval', action='store_true', default=False,
                       help='使用多模态融合检索器 (BGE+CLIP)')
    parser.add_argument('--clip-model-path', type=str,
                       default='/data0/home/zqwang/ACL/models/clip-vit-large-patch14-336',
                       help='CLIP模型路径')
    parser.add_argument('--clip-index-path', type=str,
                       default='/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/clip/clip_Flat.index',
                       help='CLIP索引路径')
    parser.add_argument('--text-retrieval-weight', type=float, default=0.6,
                       help='文本检索权重 (BGE)')
    parser.add_argument('--visual-retrieval-weight', type=float, default=0.4,
                       help='视觉检索权重 (CLIP)')

    # GPU配置
    parser.add_argument('--use-multi-gpu', action='store_true', help='使用多GPU')
    parser.add_argument('--num-gpus', type=int, default=2, help='GPU数量')

    # 59%准确率配置
    parser.add_argument('--uncertainty-threshold', type=float, default=0.43,
                       help='不确定性阈值 (基于59%准确率实验)')
    parser.add_argument('--text-weight', type=float, default=0.4, help='文本不确定性权重')
    parser.add_argument('--visual-weight', type=float, default=0.3, help='视觉不确定性权重')
    parser.add_argument('--alignment-weight', type=float, default=0.3, help='对齐不确定性权重')
    parser.add_argument('--use-improved-estimator', action='store_true', default=True,
                       help='使用改进版不确定性估计器 (基于59%准确率实验)')

    # 输出配置
    parser.add_argument('--output-dir', type=str,
                       default='/data0/home/zqwang/ACL/FlashRAG/experiments/results_okvqa_baselines',
                       help='输出目录')
    parser.add_argument('--save-detailed-results', action='store_true', default=True,
                       help='保存详细结果')
    parser.add_argument('--save-sample-results', action='store_true', default=True,
                       help='保存样本结果')
    parser.add_argument('--enable-complete-metrics', action='store_true', default=False,
                       help='启用完整指标计算')

    # Baseline方法选择
    parser.add_argument('--methods', nargs='+', default=None,
                       choices=[m['name'] for m in BASELINE_METHODS],
                       help='要运行的baseline方法')

    return parser.parse_args()

def main():
    """主函数"""
    print("🚀 OK-VQA Baselines对比实验 - 基于消融实验框架")
    print("7个方法对比：Self-Aware-MRAG, MuRAG, VisRAG, ViDoRAG, RagVL, SAM-RAG, mR²AG")
    print("📋 实时日志将输出到 baselines_<timestamp>.log")
    print("="*80)

    # 解析参数
    args = parse_args()
    config = create_config(args)

    # 创建日志文件
    log_file = Path(config['output_dir']) / f"ablation_{config['timestamp']}.log"

    # 写入开始日志
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"# 统一消融实验日志\n")
        f.write(f"开始时间: {datetime.now()}\n")
        f.write(f"数据集: {config['dataset_name']}\n")
        f.write(f"样本数: {config['max_samples']}\n")
        f.write(f"多GPU: {config['num_gpus']}\n")
        f.write(f"多模态检索: {config['use_multimodal_retrieval']}\n")
        f.write(f"不确定性阈值: {config['uncertainty_threshold']}\n")
        f.write("="*60 + "\n")
        f.flush()

    print(f"📊 实验配置:")
    print(f"   数据集: {config['dataset_name']}")
    print(f"   样本数: {config['max_samples']}")
    print(f"   多模态检索: {config['use_multimodal_retrieval']}")
    print(f"   改进估计器: {config['use_improved_estimator']}")
    print(f"   不确定性阈值: {config['uncertainty_threshold']}")
    print(f"   多GPU: {config['num_gpus']}")

    # 选择要运行的方法
    if args.methods:
        methods = [m for m in BASELINE_METHODS if m['name'] in args.methods]
    else:
        methods = BASELINE_METHODS

    print(f"\n将运行 {len(methods)} 个baseline方法:")
    for m in methods:
        print(f"  - {m['name']}: {m['description']}")

    # 1. 加载数据集
    dataset = load_dataset(config)
    if dataset is None:
        print("❌ 数据集加载失败，退出")
        return

    # 2. 初始化模型和检索器
    qwen3_vl, retriever = init_models_and_retriever(config)
    if qwen3_vl is None:
        print("❌ 模型初始化失败，退出")
        return

    # 3. 运行baseline对比实验
    all_results = []

    for method in methods:
        # 清理GPU内存
        clear_gpu_memory()

        # 运行方法
        result = run_baseline_method(
            method, dataset, qwen3_vl, retriever, config
        )
        all_results.append(result)

    # 4. 保存结果
    save_results(all_results, config)

    print(f"\n{'='*80}")
    print("✅ OK-VQA Baselines对比实验完成！")
    print(f"📁 结果保存在: {config['output_dir']}")
    print(f"{'='*80}")

if __name__ == "__main__":
    # 设置环境
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings("ignore", category=UserWarning)

    main()