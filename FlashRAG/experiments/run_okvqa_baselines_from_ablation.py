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
# GPU和内存管理
# ============================================================================

def clear_gpu_memory():
    """清理GPU内存"""
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
    """设置GPU设备"""
    if not torch.cuda.is_available():
        print("⚠️ CUDA不可用，使用CPU")
        return ['cpu']

    device_count = torch.cuda.device_count()
    print(f"✅ 检测到 {device_count} 个GPU")

    if num_gpus is None:
        num_gpus = min(2, device_count)  # 推荐使用2GPU

    num_gpus = min(num_gpus, device_count)

    if num_gpus == 0:
        return ['cpu']

    devices = [f'cuda:{i}' for i in range(num_gpus)]
    print(f"🚀 使用GPU: {devices}")

    return devices

# ============================================================================
# Baseline方法配置
# ============================================================================

BASELINE_METHODS = [
    {
        'name': 'Self-Aware-MRAG',
        'description': 'Our Self-Aware Multimodal RAG system',
        'class': 'SelfAwarePipelineQwen3VL',
        'module': 'flashrag.pipeline.self_aware_pipeline_qwen3vl'
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
        'class': 'SAMRAGEnhanced',
        'module': 'experiments.baselines.samrag_adapted'
    },
    {
        'name': 'mR²AG',
        'description': 'multi-step Reflection and Refinement Augmented Generation',
        'class': 'MR2AGEnhanced',
        'module': 'experiments.baselines.mr2ag_enhanced'
    }
]

# ============================================================================
# 配置参数
# ============================================================================

def create_config(args):
    """根据命令行参数创建配置"""
    config = {
        # 数据集配置
        'dataset_name': 'okvqa',
        'data_dir': args.data_dir,
        'split': args.split,
        'max_samples': args.max_samples,
        'load_images': True,

        # 模型配置
        'qwen3_vl_path': args.qwen3_vl_path,
        'torch_dtype': args.torch_dtype,

        # 检索器配置
        'faiss_index_path': args.faiss_index_path,
        'corpus_path': args.corpus_path,
        'retrieval_model_path': args.retrieval_model_path,
        'retrieval_topk': args.retrieval_topk,
        'bge_reranker_path': args.bge_reranker_path,

        # Self-Aware MRAG配置
        'uncertainty_threshold': 0.43,
        'use_improved_estimator': True,
        'text_weight': 0.4,
        'visual_weight': 0.3,
        'alignment_weight': 0.3,

        # 输出配置
        'output_dir': args.output_dir,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
    }

    return config

# ============================================================================
# 数据加载
# ============================================================================

def load_dataset(config):
    """加载数据集"""
    print("="*80)
    print("1. 加载数据集")
    print("="*80)

    try:
        from flashrag.dataset.okvqa_dataset_simple import OKVQADatasetSimple

        dataset = OKVQADatasetSimple({
            'data_dir': config['data_dir'],
            'split': config['split'],
            'load_images': config['load_images'],
        })

        # 限制样本数
        if config['max_samples'] and len(dataset.data) > config['max_samples']:
            dataset.data = dataset.data[:config['max_samples']]

        print(f"✅ 数据集加载完成: {len(dataset.data)} 样本")
        print(f"   数据集: {config['dataset_name']}")
        print(f"   图像加载: {config['load_images']}")

        return dataset

    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# 模型和检索器初始化
# ============================================================================

def initialize_models(config, devices):
    """初始化模型和检索器"""
    print("\n" + "="*80)
    print("2. 初始化模型和检索器")
    print("="*80)

    # 初始化Qwen3-VL
    print("\n2.1 初始化Qwen3-VL模型")
    print("-"*40)

    try:
        from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper

        qwen3_vl = create_qwen3_vl_wrapper(
            model_path=config['qwen3_vl_path'],
            device=devices[0],
            torch_dtype=config['torch_dtype']
        )
        print("✅ Qwen3-VL模型加载成功")

    except Exception as e:
        print(f"❌ Qwen3-VL模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

    # 初始化检索器
    print("\n2.2 初始化检索器")
    print("-"*40)

    try:
        from flashrag.retriever import DenseRetriever

        retriever_config = {
            'retrieval_model_path': config['retrieval_model_path'],
            'faiss_index_path': config['faiss_index_path'],
            'corpus_path': config['corpus_path'],
            'retrieval_cache_path': None,
            'use_reranker': False,
            'use_sentence_transformer': False,
            'faiss_gpu': len(devices) > 1,
            'instruction': '',
        }

        retriever = DenseRetriever(retriever_config)
        print("✅ 检索器加载成功")

    except Exception as e:
        print(f"⚠️ 检索器加载失败: {e}")
        retriever = None

    return qwen3_vl, retriever

# ============================================================================
# 创建baseline pipeline
# ============================================================================

def create_baseline_pipeline(method_config, qwen3_vl, retriever, config):
    """创建指定的baseline pipeline"""

    module_name = method_config['module']
    class_name = method_config['class']

    try:
        # 动态导入
        module = __import__(module_name, fromlist=[class_name])
        pipeline_class = getattr(module, class_name)

        if method_config['name'] == 'Self-Aware-MRAG':
            # Self-Aware-MRAG需要特殊配置
            pipeline = pipeline_class(
                qwen3_vl,
                retriever=retriever,
                config={
                    'uncertainty_threshold': config['uncertainty_threshold'],
                    'use_improved_estimator': config['use_improved_estimator'],
                    'use_position_fusion': True,
                    'use_attribution': True,
                }
            )
        elif method_config['name'] == 'VisRAG':
            # VisRAG需要本地BGE reranker路径
            visrag_config = config.copy()
            visrag_config['bge_reranker_path'] = config['bge_reranker_path']
            pipeline = pipeline_class(qwen3_vl, retriever, config=visrag_config)
        else:
            # 其他方法使用通用配置
            pipeline = pipeline_class(qwen3_vl, retriever, config)

        print(f"✅ {method_config['name']} 初始化成功")
        return pipeline

    except Exception as e:
        print(f"❌ {method_config['name']} 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# 运行实验
# ============================================================================

def run_experiment(method_config, qwen3_vl, retriever, dataset, config):
    """运行单个baseline方法实验"""
    print(f"\n{'='*60}")
    print(f"运行方法: {method_config['name']}")
    print(f"描述: {method_config['description']}")
    print(f"{'='*60}")

    # 创建输出目录和日志文件
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / f"{method_config['name']}_log.txt"

    try:
        # 记录开始时间
        start_time = time.time()

        # 写入开始日志
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"方法: {method_config['name']}\n")
            f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n")
            f.flush()

        # 创建pipeline
        pipeline = create_baseline_pipeline(method_config, qwen3_vl, retriever, config)
        if pipeline is None:
            return None

        # 运行pipeline
        results = []
        for i, sample in enumerate(dataset.data):
            print(f"\r进度: {i+1}/{len(dataset.data)}", end='', flush=True)

            try:
                if hasattr(pipeline, 'run_single'):
                    result = pipeline.run_single(sample)
                else:
                    result = pipeline(sample)
                results.append(result)
            except Exception as e:
                print(f"\n[ERROR] 样本 {i} 处理失败: {e}")
                results.append({
                    'question': sample.get('question', ''),
                    'answer': '',
                    'retrieved_docs': [],
                    'golden_answers': sample.get('golden_answers', [])
                })

        print()  # 换行

        end_time = time.time()

        # 使用综合评估器计算指标
        metrics = evaluate_comprehensive_metrics(results)

        # 计算基础统计
        base_stats = {
            'method_name': method_config['name'],
            'method_description': method_config['description'],
            'total_samples': len(results),
            'execution_time': end_time - start_time,
            'timestamp': datetime.now().isoformat(),
            'seconds_per_sample': (end_time - start_time) / len(results) if len(results) > 0 else 0
        }

        # 合并指标
        base_stats.update(metrics)

        print(f"\n✅ 完成: {method_config['name']}")
        print(f"   准确率: {base_stats.get('accuracy', 0)*100:.2f}%")
        print(f"   F1分数: {base_stats.get('avg_F1', 0):.4f}")
        print(f"   检索率: {base_stats.get('retrieval_rate', 0)*100:.1f}%")
        print(f"   耗时: {end_time - start_time:.1f}秒 ({base_stats['seconds_per_sample']:.2f}s/样本)")

        # 写入完成日志
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 实验完成\n")
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] 准确率: {base_stats.get('accuracy', 0)*100:.2f}%\n")
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] F1分数: {base_stats.get('avg_F1', 0):.4f}\n")
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] 检索率: {base_stats.get('retrieval_rate', 0)*100:.1f}%\n")
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] 耗时: {end_time - start_time:.1f}秒\n")
            f.flush()

        return {
            'results': results,
            'stats': base_stats
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
# 结果保存
# ============================================================================

def save_results(all_results, config):
    """保存实验结果"""
    print(f"\n{'='*80}")
    print("4. 保存实验结果")
    print(f"{'='*80}")

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = config['timestamp']
    results_file = output_dir / f"okvqa_baselines_results_{timestamp}.json"
    summary_file = output_dir / f"okvqa_baselines_summary_{timestamp}.json"
    report_file = output_dir / f"okvqa_baselines_report_{timestamp}.md"

    # 准备保存的数据
    save_data = {
        'experiment_info': {
            'dataset': config['dataset_name'],
            'samples': config['max_samples'],
            'timestamp': timestamp,
            'config': config,
        },
        'methods_summary': [],
        'detailed_results': {},
    }

    # 处理每个方法的结果
    for method_result in all_results:
        if method_result is None:
            continue

        stats = method_result['stats']
        results = method_result['results']

        # 添加到汇总
        save_data['methods_summary'].append({
            'method_name': stats['method_name'],
            'method_description': stats['method_description'],
            'accuracy': stats.get('accuracy', 0),
            'F1': stats.get('avg_F1', 0),
            'retrieval_rate': stats.get('retrieval_rate', 0),
            'Recall@5': stats.get('avg_Recall@5', 0),
            'Faithfulness': stats.get('avg_Faithfulness', 0),
            'Attribution Precision': stats.get('avg_Attribution_Precision', 0),
            'Position Bias Score': stats.get('avg_Position_Bias_Score', 0),
            'execution_time': stats['execution_time'],
            'seconds_per_sample': stats['seconds_per_sample']
        })

        # 保存详细结果
        save_data['detailed_results'][stats['method_name']] = {
            'stats': stats,
            'results': enhance_results_saving(results)
        }

    # 保存JSON结果
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    # 创建并保存汇总
    summary_data = {
        'config': config,
        'timestamp': timestamp,
        'total_samples': config['max_samples'],
        'methods': {}
    }

    for method_summary in save_data['methods_summary']:
        summary_data['methods'][method_summary['method_name']] = {
            '准确率': method_summary['accuracy'],
            'F1': method_summary['F1'],
            '检索率': method_summary['retrieval_rate'],
            'Recall@5': method_summary['Recall@5'],
            'Faithfulness': method_summary['Faithfulness'],
            'Attribution Precision': method_summary['Attribution Precision'],
            'Position Bias Score': method_summary['Position Bias Score']
        }

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    # 生成Markdown报告
    generate_markdown_report(save_data, report_file)

    print(f"\n✅ 结果已保存到: {output_dir}")
    print(f"   详细结果: {results_file}")
    print(f"   汇总结果: {summary_file}")
    print(f"   Markdown报告: {report_file}")

    return save_data

# ============================================================================
# 生成Markdown报告
# ============================================================================

def generate_markdown_report(save_data, report_file):
    """生成Markdown格式的实验报告"""

    report_content = f"""# OK-VQA Baselines对比实验报告

## 实验信息

- **数据集**: {save_data['experiment_info']['dataset']}
- **样本数**: {save_data['experiment_info']['samples']}
- **时间**: {save_data['experiment_info']['timestamp']}

## 方法对比结果

| 方法 | 准确率 | F1 | 检索率 | Recall@5 | Faithfulness | Attribution Precision | Position Bias Score |
|------|--------|----|----|---------|-------------|----------------------|---------------------|
"""

    for method in save_data['methods_summary']:
        report_content += f"| {method['method_name']} | {method['accuracy']:.4f} | {method['F1']:.4f} | {method['retrieval_rate']:.4f} | {method['Recall@5']:.4f} | {method['Faithfulness']:.4f} | {method['Attribution Precision']:.4f} | {method['Position Bias Score']:.4f} |\n"

    report_content += f"""

## 详细结果

- **最佳准确率**: {max(m['accuracy'] for m in save_data['methods_summary']):.4f}
- **最佳F1分数**: {max(m['F1'] for m in save_data['methods_summary']):.4f}
- **最高检索率**: {max(m['retrieval_rate'] for m in save_data['methods_summary']):.4f}

"""

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)

# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='OK-VQA Baselines对比实验')
    parser.add_argument('--dataset', default='okvqa', help='数据集名称')
    parser.add_argument('--data_dir', default='/data1/userdata/zqwang/ACL_data/VQA', help='数据目录')
    parser.add_argument('--split', default='val', help='数据集分割')
    parser.add_argument('--max_samples', type=int, default=10, help='最大样本数')
    parser.add_argument('--qwen3_vl_path', default='/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct', help='Qwen3-VL模型路径')
    parser.add_argument('--torch_dtype', default='bfloat16', help='PyTorch数据类型')
    parser.add_argument('--faiss_index_path', default='/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index', help='FAISS索引路径')
    parser.add_argument('--corpus_path', default='/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl', help='语料库路径')
    parser.add_argument('--retrieval_model_path', default='/data0/home/zqwang/ACL/models/bge-large-en-v1.5', help='检索模型路径')
    parser.add_argument('--retrieval_topk', type=int, default=5, help='检索文档数量')
    parser.add_argument('--bge_reranker_path', default='/data0/home/zqwang/ACL/models/bge-reranker-v2-m3', help='BGE重排序模型路径')
    parser.add_argument('--output_dir', default='/data0/home/zqwang/ACL/FlashRAG/experiments/okvqa_baselines_results', help='输出目录')
    parser.add_argument('--num_gpus', type=int, default=2, help='使用的GPU数量')

    args = parser.parse_args()

    # 创建配置
    config = create_config(args)

    # 设置设备
    devices = setup_device(args.num_gpus)

    print("="*80)
    print("OK-VQA Baselines对比实验")
    print("="*80)
    print(f"数据集: {config['dataset_name']}")
    print(f"样本数: {config['max_samples']}")
    print(f"GPU: {devices}")
    print(f"输出目录: {config['output_dir']}")

    # 1. 加载数据集
    dataset = load_dataset(config)
    if dataset is None:
        return

    # 2. 初始化模型
    qwen3_vl, retriever = initialize_models(config, devices)
    if qwen3_vl is None:
        return

    # 3. 运行所有baseline方法
    print("\n" + "="*80)
    print("3. 运行Baseline方法")
    print("="*80)

    all_results = []
    successful_methods = 0

    for method_config in BASELINE_METHODS:
        # 清理GPU内存
        clear_gpu_memory()

        # 运行实验
        result = run_experiment(method_config, qwen3_vl, retriever, dataset, config)

        if result is not None:
            all_results.append(result)
            successful_methods += 1

        # 强制清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 4. 保存结果
    if all_results:
        save_results(all_results, config)

        print("\n" + "="*80)
        print("实验完成！")
        print("="*80)
        print(f"成功方法数: {successful_methods}/{len(BASELINE_METHODS)}")
        print(f"输出目录: {config['output_dir']}")
    else:
        print("\n❌ 没有成功完成的实验")

if __name__ == "__main__":
    main()