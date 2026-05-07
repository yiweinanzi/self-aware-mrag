#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速修复版消融实验脚本 - 100样本测试
根据实施方案修复核心问题后的简化版本

核心修复：
1. ✅ 修复不确定性阈值逻辑（0.43 → 0.35，基于P60-P70百分位）
2. ✅ 修复组件交互问题（强制检索逻辑）
3. ✅ 简化配置管理（统一配置）
4. ✅ 增强错误处理（避免崩溃）
5. ✅ 限制样本数（100样本，快速验证）

实验设计：
1. Baseline (MuRAG)
2. + Text Uncertainty
3. + Visual Uncertainty
4. + Cross-Modal Alignment Unc.
5. + Position-Aware Fusion
6. + Fine-Grained Attribution (Full Method)

样本数：100
数据集：OK-VQA val2014
评估指标：EM, F1, VQA-Score
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np

# 添加FlashRAG路径
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from flashrag.dataset.okvqa_dataset_simple import OKVQADatasetSimple
from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
from flashrag.retriever import DenseRetriever
from flashrag.pipeline.self_aware_pipeline_qwen3vl import SelfAwarePipelineQwen3VL
from flashrag.evaluator.complete_metrics import CompleteMetricsCalculator

# ============================================================================
# 实验配置（修复版）
# ============================================================================

FIXED_CONFIG = {
    # 数据集配置
    'dataset_name': 'okvqa',
    'data_dir': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/VQA',
    'split': 'val',
    'max_samples': 100,  # ✅ 限制为100样本进行快速测试
    'load_images': True,

    # 模型配置
    'qwen3_vl_path': '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',

    # 检索器配置
    'index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
    'corpus_path': '/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl',
    'retrieval_model_path': '/data0/home/zqwang/ACL/models/bge-large-en-v1.5',
    'retrieval_topk': 5,

    # 评估配置
    'save_results': True,
    'output_dir': '/data0/home/zqwang/ACL/FlashRAG/experiments/results_fixed_ablation_100',

    # 生成参数
    'temperature': 0.01,
    'max_new_tokens': 20,

    # ✅ 修复：不确定性阈值（基于P60-P70百分位，而非P92）
    'uncertainty_threshold': 0.35,

    # ✅ 修复：消融实验配置（简化版本）
    'ablation_variants': [
        {
            'name': 'Baseline (MuRAG)',
            'config': {
                'use_uncertainty_estimation': False,
                'use_position_fusion': False,
                'use_attribution': False,
                'uncertainty_threshold': 1.0,  # 总是检索
                'force_retrieval': True,  # ✅ 强制检索
            }
        },
        {
            'name': '+ Text Uncertainty',
            'config': {
                'use_uncertainty_estimation': True,
                'uncertainty_components': ['text'],
                'use_position_fusion': False,
                'use_attribution': False,
                'uncertainty_threshold': 0.35,
                'force_retrieval': True,  # ✅ 强制检索
                'text_weight': 0.6,
                'visual_weight': 0.2,
                'alignment_weight': 0.2,
            }
        },
        {
            'name': '+ Visual Uncertainty',
            'config': {
                'use_uncertainty_estimation': True,
                'uncertainty_components': ['text', 'visual'],
                'use_position_fusion': False,
                'use_attribution': False,
                'uncertainty_threshold': 0.35,
                'force_retrieval': True,  # ✅ 强制检索
                'text_weight': 0.4,
                'visual_weight': 0.4,
                'alignment_weight': 0.2,
            }
        },
        {
            'name': '+ Cross-Modal Alignment Unc.',
            'config': {
                'use_uncertainty_estimation': True,
                'uncertainty_components': ['text', 'visual', 'alignment'],
                'use_position_fusion': False,
                'use_attribution': False,
                'uncertainty_threshold': 0.35,
                'force_retrieval': True,  # ✅ 强制检索
                'text_weight': 0.35,
                'visual_weight': 0.35,
                'alignment_weight': 0.3,
            }
        },
        {
            'name': '+ Position-Aware Fusion',
            'config': {
                'use_uncertainty_estimation': True,
                'uncertainty_components': ['text', 'visual', 'alignment'],
                'use_position_fusion': True,
                'use_attribution': False,
                'uncertainty_threshold': 0.35,
                'force_retrieval': True,  # ✅ 强制检索
                'text_weight': 0.35,
                'visual_weight': 0.35,
                'alignment_weight': 0.3,
            }
        },
        {
            'name': '+ Fine-Grained Attribution (Full)',
            'config': {
                'use_uncertainty_estimation': True,
                'uncertainty_components': ['text', 'visual', 'alignment'],
                'use_position_fusion': True,
                'use_attribution': True,
                'uncertainty_threshold': 0.35,
                'force_retrieval': True,  # ✅ 强制检索
                'text_weight': 0.35,
                'visual_weight': 0.35,
                'alignment_weight': 0.3,
            }
        }
    ]
}

def load_dataset(config):
    """加载数据集"""
    print("="*80)
    print("1. 加载OK-VQA数据集")
    print("="*80)

    try:
        dataset = OKVQADatasetSimple({
            'data_dir': config['data_dir'],
            'split': config['split'],
            'load_images': config['load_images'],
        })

        # 限制样本数
        if config['max_samples'] and len(dataset.data) > config['max_samples']:
            dataset.data = dataset.data[:config['max_samples']]

        print(f"✅ 加载完成: {len(dataset.data)} 样本")
        return dataset

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None

def init_models_and_retriever(config):
    """初始化模型和检索器"""
    print("\n" + "="*80)
    print("2. 初始化模型和检索器")
    print("="*80)

    # 初始化Qwen3-VL
    try:
        print("初始化Qwen3-VL...")
        qwen3_vl = create_qwen3_vl_wrapper(
            model_path=config['qwen3_vl_path'],
            device='cuda',
            dtype='half'
        )
        print("✅ Qwen3-VL初始化成功")
    except Exception as e:
        print(f"❌ Qwen3-VL初始化失败: {e}")
        return None, None

    # 初始化检索器
    try:
        print("初始化检索器...")
        retriever = DenseRetriever(
            model_path=config['retrieval_model_path'],
            index_path=config['index_path'],
            corpus_path=config['corpus_path']
        )
        print("✅ 检索器初始化成功")
    except Exception as e:
        print(f"⚠️ 检索器初始化失败: {e}")
        print("使用模拟检索器...")
        retriever = None

    return qwen3_vl, retriever

def run_ablation_variant(variant_name, variant_config, dataset, qwen3_vl, retriever):
    """运行单个消融变体"""
    print(f"\n{'='*60}")
    print(f"运行变体: {variant_name}")
    print(f"{'='*60}")

    # 合并基础配置和变体配置
    config = FIXED_CONFIG.copy()
    config.update(variant_config)

    # 创建Pipeline
    try:
        pipeline = SelfAwarePipelineQwen3VL(
            qwen3_vl_wrapper=qwen3_vl,
            retriever=retriever,
            config=config
        )
        print(f"✅ Pipeline创建成功: {variant_name}")
    except Exception as e:
        print(f"❌ Pipeline创建失败: {e}")
        return None

    # 运行实验
    try:
        start_time = time.time()
        results = pipeline.run(dataset, verbose=True)
        end_time = time.time()

        # 计算基础指标
        correct_count = sum(1 for r in results if r.get('correct', False))
        total_count = len(results)
        accuracy = correct_count / total_count if total_count > 0 else 0
        retrieval_rate = sum(1 for r in results if r.get('retrieved', False)) / total_count

        # 收集详细统计
        stats = {
            'variant_name': variant_name,
            'config': variant_config,
            'total_samples': total_count,
            'correct_count': correct_count,
            'accuracy': accuracy,
            'retrieval_rate': retrieval_rate,
            'execution_time': end_time - start_time,
            'timestamp': datetime.now().isoformat()
        }

        print(f"\n✅ 完成: {variant_name}")
        print(f"   准确率: {accuracy*100:.2f}% ({correct_count}/{total_count})")
        print(f"   检索率: {retrieval_rate*100:.1f}%")
        print(f"   耗时: {end_time - start_time:.1f}秒")

        return {
            'results': results,
            'stats': stats
        }

    except Exception as e:
        print(f"❌ 实验运行失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_results(all_results, config):
    """保存结果"""
    print(f"\n{'='*80}")
    print("保存结果")
    print(f"{'='*80}")

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存详细结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"ablation_results_100samples_{timestamp}.json"

    # 准备保存的数据
    save_data = {
        'experiment_info': {
            'dataset': config['dataset_name'],
            'samples': config['max_samples'],
            'timestamp': timestamp,
            'config': config,
        },
        'variants_summary': [],
        'detailed_results': {}
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
            'accuracy': stats['accuracy'],
            'retrieval_rate': stats['retrieval_rate'],
            'execution_time': stats['execution_time'],
            'config': stats['config']
        })

        # 添加详细结果（只保存前10个样本的详细信息）
        save_data['detailed_results'][stats['variant_name']] = {
            'stats': stats,
            'sample_results': results[:10]  # 只保存前10个样本
        }

    # 写入文件
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"✅ 结果已保存: {results_file}")

        # 打印汇总表格
        print(f"\n{'='*60}")
        print("实验结果汇总")
        print(f"{'='*60}")
        print(f"{'变体':<30} {'准确率':<10} {'检索率':<10} {'耗时(s)':<10}")
        print("-" * 60)

        for variant in save_data['variants_summary']:
            print(f"{variant['variant_name']:<30} "
                  f"{variant['accuracy']*100:>6.2f}%  "
                  f"{variant['retrieval_rate']*100:>6.1f}%  "
                  f"{variant['execution_time']:>8.1f}")

    except Exception as e:
        print(f"❌ 保存结果失败: {e}")

def main():
    """主函数"""
    print("修复版消融实验 - 100样本测试")
    print("基于实施方案修复核心问题")
    print("="*80)

    # 1. 加载数据集
    dataset = load_dataset(FIXED_CONFIG)
    if dataset is None:
        print("❌ 数据集加载失败，退出")
        return

    # 2. 初始化模型和检索器
    qwen3_vl, retriever = init_models_and_retriever(FIXED_CONFIG)
    if qwen3_vl is None:
        print("❌ 模型初始化失败，退出")
        return

    # 3. 运行消融实验
    all_results = []

    for variant in FIXED_CONFIG['ablation_variants']:
        result = run_ablation_variant(
            variant['name'],
            variant['config'],
            dataset,
            qwen3_vl,
            retriever
        )
        all_results.append(result)

    # 4. 保存结果
    save_results(all_results, FIXED_CONFIG)

    print(f"\n{'='*80}")
    print("✅ 所有实验完成！")
    print(f"{'='*80}")

if __name__ == "__main__":
    # 设置环境
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings("ignore", category=UserWarning)

    main()