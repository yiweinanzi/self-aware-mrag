#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一评测脚本 - 所有方法在MRAG-Bench上的完整评测

评测配置：
- 数据集: MRAG-Bench (6153样本)
- 语料库: 3M (Wiki 1.5M + CC3M 1.5M)
- 指标: 7个核心指标
- 方法: 6个baseline + Our Method

运行方式:
```bash
cd /root/autodl-tmp/FlashRAG
python experiments/run_all_methods_evaluation.py
```
"""

import os
import sys
import json
import warnings
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# 添加FlashRAG路径
sys.path.insert(0, '/root/autodl-tmp/FlashRAG')

import torch
import numpy as np
from tqdm import tqdm

# 导入所有方法
from flashrag.pipeline.self_aware_pipeline_qwen3vl import SelfAwarePipelineQwen3VL
from flashrag.baseline.selfrag import SelfRAGBaseline
from flashrag.experiments.baselines.mr2ag_baseline import MR2AGBaseline
from flashrag.experiments.baselines.visrag_enhanced import VisRAGEnhanced
from flashrag.experiments.baselines.reveal_baseline import REVEALBaseline
from flashrag.experiments.baselines.ragvl_baseline import RagVLBaseline
from flashrag.experiments.baselines.murag_baseline import MuRAGBaseline

# 导入评估指标
from flashrag.evaluator.complete_metrics import CompleteMetricsCalculator

# 导入Qwen3-VL
from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper


# ============================================================================
# 配置
# ============================================================================

CONFIG = {
    # 数据集配置
    'dataset_name': 'mragbench',
    'dataset_path': '/root/autodl-tmp/FlashRAG/flashrag/data/MRAG-Bench/test.json',
    
    # 语料库配置
    'corpus_path': '/root/autodl-tmp/FlashRAG/corpus/corpus_3m.jsonl',
    'index_dir': '/root/autodl-tmp/FlashRAG/indexes/3m',
    
    # 模型配置
    'qwen3_vl_path': '/root/autodl-tmp/models/Qwen3-VL-8B-Instruct',
    'clip_model_path': '/root/autodl-tmp/models/clip-vit-large-patch14-336',
    'bge_model_path': '/root/autodl-tmp/models/bge-large-en-v1.5',
    
    # 评测配置
    'max_samples': None,  # None表示全部样本，可设置为100进行快速测试
    'batch_size': 1,
    'save_results': True,
    
    # 输出配置
    'output_dir': '/root/autodl-tmp/FlashRAG/experiments/results_all_methods_3m',
    
    # 通用参数（确保所有方法统一）
    'temperature': 0.2,
    'max_new_tokens': 100,
    'retrieval_topk': 5,
}


# ============================================================================
# 数据加载
# ============================================================================

def load_mragbench_dataset(dataset_path: str, max_samples: int = None):
    """
    加载MRAG-Bench数据集
    
    Args:
        dataset_path: 数据集路径
        max_samples: 最大样本数（None表示全部）
    
    Returns:
        List[Dict]: 样本列表
    """
    print(f"\n{'='*80}")
    print("加载MRAG-Bench数据集...")
    print(f"{'='*80}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    if max_samples is not None:
        dataset = dataset[:max_samples]
    
    print(f"✅ 加载完成: {len(dataset)} 样本")
    
    return dataset


# ============================================================================
# 检索器初始化
# ============================================================================

def init_retriever(config: Dict):
    """
    初始化检索器（使用3M语料库）
    
    Args:
        config: 配置字典
    
    Returns:
        retriever对象
    """
    print(f"\n{'='*80}")
    print("初始化检索器（3M语料库）...")
    print(f"{'='*80}")
    
    # 导入FlashRAG的检索器
    from flashrag.retriever.multimodal_retriever import SelfAwareMultimodalRetriever
    from flashrag.retriever import DenseRetriever
    
    try:
        # 方案1: 使用SelfAwareMultimodalRetriever（推荐）
        print("\n尝试初始化 SelfAwareMultimodalRetriever...")
        
        # 配置文本检索器（BGE）
        text_retriever_config = {
            'index_path': os.path.join(config['index_dir'], 'bge'),
            'corpus_path': config['corpus_path'],
            'retrieval_method': 'e5',
            'retrieval_model_path': config['bge_model_path'],
            'retrieval_query_max_length': 512,
            'retrieval_pooling_method': 'mean',
            'retrieval_use_fp16': True,
            'retrieval_batch_size': 128,
            'retrieval_topk': config['retrieval_topk'],
            'use_sentence_transformer': False,
            'faiss_gpu': False,
            'instruction': '',
            'save_retrieval_cache': False,
        }
        
        # 创建文本检索器
        text_retriever = DenseRetriever(text_retriever_config)
        
        # 配置多模态检索器
        retriever_config = {
            'retrieval_topk': config['retrieval_topk'],
            'use_clip': True,
            'clip_model_path': config['clip_model_path'],
            'text_weight': 0.5,
            'visual_weight': 0.5,
            'fusion_method': 'weighted',
        }
        
        # 创建多模态检索器
        retriever = SelfAwareMultimodalRetriever(
            config=retriever_config,
            text_retriever=text_retriever,
            visual_retriever=None  # 如果有CLIP索引可以添加
        )
        
        print(f"✅ SelfAwareMultimodalRetriever初始化成功")
        print(f"  - 语料库: {config['corpus_path']}")
        print(f"  - BGE索引: {os.path.join(config['index_dir'], 'bge')}")
        print(f"  - CLIP模型: {config['clip_model_path']}")
        
        return retriever
        
    except Exception as e:
        print(f"⚠️  SelfAwareMultimodalRetriever初始化失败: {e}")
        print("尝试简化版检索器...")
        
        try:
            # 方案2: 只使用BGE文本检索器
            print("\n尝试初始化 DenseRetriever（仅文本）...")
            
            retriever_config = {
                'index_path': os.path.join(config['index_dir'], 'bge'),
                'corpus_path': config['corpus_path'],
                'retrieval_method': 'e5',
                'retrieval_model_path': config['bge_model_path'],
                'retrieval_query_max_length': 512,
                'retrieval_pooling_method': 'mean',
                'retrieval_use_fp16': True,
                'retrieval_batch_size': 128,
                'retrieval_topk': config['retrieval_topk'],
                'use_sentence_transformer': False,
                'faiss_gpu': False,
                'instruction': '',
                'save_retrieval_cache': False,
            }
            
            retriever = DenseRetriever(retriever_config)
            
            print(f"✅ DenseRetriever初始化成功（仅文本检索）")
            print(f"  - BGE索引: {os.path.join(config['index_dir'], 'bge')}")
            
            return retriever
            
        except Exception as e2:
            print(f"⚠️  DenseRetriever初始化也失败: {e2}")
            print("使用Mock检索器（仅用于测试）...")
            
            # 方案3: Mock检索器（仅用于测试）
            class MockRetriever:
                def __init__(self, corpus_path, index_dir):
                    self.corpus_path = corpus_path
                    self.index_dir = index_dir
                    print(f"⚠️  使用Mock检索器（返回空结果）")
                    print(f"  - 语料库: {corpus_path}")
                    print(f"  - 索引目录: {index_dir}")
                
                def retrieve(self, query_text, query_image=None, top_k=5):
                    """Mock检索"""
                    return [], []
            
            retriever = MockRetriever(config['corpus_path'], config['index_dir'])
            
            return retriever


# ============================================================================
# 方法初始化
# ============================================================================

def init_all_methods(qwen3_vl, retriever, config):
    """
    初始化所有评测方法
    
    Args:
        qwen3_vl: Qwen3-VL模型
        retriever: 检索器
        config: 配置
    
    Returns:
        Dict[str, object]: 方法名到实例的映射
    """
    print(f"\n{'='*80}")
    print("初始化所有评测方法...")
    print(f"{'='*80}")
    
    # 统一配置
    base_config = {
        'temperature': config['temperature'],
        'max_new_tokens': config['max_new_tokens'],
        'retrieval_topk': config['retrieval_topk'],
    }
    
    methods = {}
    
    # 1. Our Method: Self-Aware MRAG (Qwen3-VL)
    print("\n1. 初始化 Self-Aware MRAG (Our Method)...")
    methods['Self-Aware-MRAG'] = SelfAwarePipelineQwen3VL(
        qwen3_vl_wrapper=qwen3_vl,
        retriever=retriever,
        config={
            **base_config,
            'uncertainty_threshold': 0.35,
            'use_position_fusion': True,
            'use_attribution': True,
            'enable_multimodal_output': False,
        }
    )
    
    # 2. Self-RAG
    print("\n2. 初始化 Self-RAG...")
    try:
        methods['Self-RAG'] = SelfRAGBaseline(
            qwen3_vl_wrapper=qwen3_vl,
            retriever=retriever,
            config=base_config
        )
    except Exception as e:
        warnings.warn(f"Self-RAG初始化失败: {e}")
    
    # 3. mR²AG
    print("\n3. 初始化 mR²AG...")
    try:
        methods['mR2AG'] = MR2AGBaseline(
            qwen3_vl_wrapper=qwen3_vl,
            retriever=retriever,
            config=base_config
        )
    except Exception as e:
        warnings.warn(f"mR²AG初始化失败: {e}")
    
    # 4. VisRAG
    print("\n4. 初始化 VisRAG...")
    try:
        methods['VisRAG'] = VisRAGEnhanced(
            qwen3_vl_wrapper=qwen3_vl,
            retriever=retriever,
            config=base_config
        )
    except Exception as e:
        warnings.warn(f"VisRAG初始化失败: {e}")
    
    # 5. REVEAL
    print("\n5. 初始化 REVEAL...")
    try:
        methods['REVEAL'] = REVEALBaseline(
            qwen3_vl_wrapper=qwen3_vl,
            retriever=retriever,
            config=base_config
        )
    except Exception as e:
        warnings.warn(f"REVEAL初始化失败: {e}")
    
    # 6. RagVL
    print("\n6. 初始化 RagVL...")
    try:
        methods['RagVL'] = RagVLBaseline(
            qwen3_vl_wrapper=qwen3_vl,
            retriever=retriever,
            config=base_config
        )
    except Exception as e:
        warnings.warn(f"RagVL初始化失败: {e}")
    
    # 7. MuRAG
    print("\n7. 初始化 MuRAG...")
    try:
        methods['MuRAG'] = MuRAGBaseline(
            qwen3_vl_wrapper=qwen3_vl,
            retriever=retriever,
            config=base_config
        )
    except Exception as e:
        warnings.warn(f"MuRAG初始化失败: {e}")
    
    print(f"\n✅ 成功初始化 {len(methods)} 个方法")
    
    return methods


# ============================================================================
# 单个方法评测
# ============================================================================

def evaluate_single_method(method_name: str, method, dataset: List[Dict],
                          config: Dict) -> Dict[str, Any]:
    """
    评测单个方法
    
    Args:
        method_name: 方法名称
        method: 方法实例
        dataset: 数据集
        config: 配置
    
    Returns:
        Dict: 评测结果
    """
    print(f"\n{'='*80}")
    print(f"评测方法: {method_name}")
    print(f"{'='*80}")
    
    results = []
    start_time = time.time()
    
    # 运行方法
    for sample in tqdm(dataset, desc=f"运行 {method_name}"):
        try:
            # 调用方法生成答案
            if hasattr(method, 'run_single'):
                result = method.run_single(sample)
            elif hasattr(method, 'generate'):
                result = {
                    'question': sample.get('question', ''),
                    'answer': method.generate(sample),
                    'golden_answers': sample.get('golden_answers', [])
                }
            else:
                warnings.warn(f"{method_name}没有run_single或generate方法")
                continue
            
            results.append(result)
            
        except Exception as e:
            warnings.warn(f"样本处理失败: {e}")
            continue
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 计算7个核心指标
    print(f"\n计算7个核心指标...")
    
    # 创建评估数据对象
    class EvalData:
        def __init__(self, results):
            self.pred = [r.get('answer', '') for r in results]
            self.golden_answers = [r.get('golden_answers', []) for r in results]
            self.choices = [[] for _ in results]
            
            # 可选字段
            self.retrieval_result = [r.get('retrieved_docs', []) for r in results]
            self.attributions = [r.get('attributions') for r in results]
    
    eval_data = EvalData(results)
    
    # 计算指标
    metrics_calculator = CompleteMetricsCalculator({
        'dataset_name': config['dataset_name']
    })
    
    metrics = metrics_calculator.calculate_all_metrics(eval_data)
    
    # 汇总结果
    summary = {
        'method_name': method_name,
        'num_samples': len(results),
        'elapsed_time': elapsed_time,
        'avg_time_per_sample': elapsed_time / len(results) if results else 0,
        'metrics': metrics,
        'detailed_results': results
    }
    
    # 打印结果
    print(f"\n{metrics_calculator.format_results(metrics)}")
    print(f"\n运行时间: {elapsed_time:.2f}秒 ({elapsed_time/len(results):.2f}秒/样本)")
    
    return summary


# ============================================================================
# 保存结果
# ============================================================================

def save_results(all_results: Dict[str, Any], config: Dict):
    """
    保存所有评测结果
    
    Args:
        all_results: 所有方法的结果
        config: 配置
    """
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 保存完整结果（JSON）
    results_file = output_dir / f"all_results_{timestamp}.json"
    
    # 移除detailed_results（太大）
    summary_results = {
        method_name: {
            'num_samples': result['num_samples'],
            'elapsed_time': result['elapsed_time'],
            'avg_time_per_sample': result['avg_time_per_sample'],
            'metrics': result['metrics']
        }
        for method_name, result in all_results.items()
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 保存结果到: {results_file}")
    
    # 2. 生成对比报告（Markdown）
    report_file = output_dir / f"EVALUATION_REPORT_{timestamp}.md"
    generate_comparison_report(all_results, report_file, config)
    
    print(f"✅ 生成报告: {report_file}")


def generate_comparison_report(all_results: Dict[str, Any],
                               report_file: Path,
                               config: Dict):
    """
    生成对比报告
    
    Args:
        all_results: 所有方法的结果
        report_file: 报告文件路径
        config: 配置
    """
    lines = []
    
    # 标题
    lines.append("# MRAG-Bench完整评测报告")
    lines.append("")
    lines.append(f"**评测时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**数据集**: MRAG-Bench")
    lines.append(f"**语料库**: 3M (Wiki 1.5M + CC3M 1.5M)")
    lines.append(f"**模型**: Qwen3-VL-8B-Instruct")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 7个核心指标对比表
    lines.append("## 7个核心指标对比")
    lines.append("")
    
    # 表头
    lines.append("| Method | EM | F1 | Recall@5 | VQA-Score | Faithfulness | Attribution | PosBias |")
    lines.append("|--------|----|----|----------|-----------|--------------|-------------|---------|")
    
    # 数据行
    for method_name, result in all_results.items():
        metrics = result['metrics']
        line = f"| **{method_name}** |"
        line += f" {metrics.get('em', 0):.4f} |"
        line += f" {metrics.get('f1', 0):.4f} |"
        line += f" {metrics.get('retrieval_recall_top5', 0):.4f} |"
        line += f" {metrics.get('vqa_score', 0):.4f} |"
        line += f" {metrics.get('faithfulness', 0):.4f} |"
        line += f" {metrics.get('attribution_precision', 0):.4f} |"
        line += f" {metrics.get('position_bias_score', 0):.4f} |"
        lines.append(line)
    
    lines.append("")
    lines.append("**注**: Position Bias Score越低越好（↓），其他指标越高越好（↑）")
    lines.append("")
    
    # 性能统计
    lines.append("## 性能统计")
    lines.append("")
    lines.append("| Method | 样本数 | 总时间(秒) | 平均时间(秒/样本) |")
    lines.append("|--------|-------|----------|----------------|")
    
    for method_name, result in all_results.items():
        lines.append(f"| {method_name} | "
                    f"{result['num_samples']} | "
                    f"{result['elapsed_time']:.2f} | "
                    f"{result['avg_time_per_sample']:.3f} |")
    
    lines.append("")
    
    # 详细分析
    lines.append("## 详细分析")
    lines.append("")
    
    for method_name, result in all_results.items():
        lines.append(f"### {method_name}")
        lines.append("")
        
        metrics = result['metrics']
        
        lines.append("**核心指标**:")
        lines.append(f"- EM: {metrics.get('em', 0):.4f}")
        lines.append(f"- F1: {metrics.get('f1', 0):.4f}")
        lines.append(f"- Recall@5: {metrics.get('retrieval_recall_top5', 0):.4f}")
        lines.append(f"- VQA-Score: {metrics.get('vqa_score', 0):.4f}")
        lines.append(f"- Faithfulness: {metrics.get('faithfulness', 0):.4f}")
        lines.append(f"- Attribution: {metrics.get('attribution_precision', 0):.4f}")
        lines.append(f"- Position Bias: {metrics.get('position_bias_score', 0):.4f} (↓)")
        lines.append("")
        
        lines.append("**性能**:")
        lines.append(f"- 处理样本数: {result['num_samples']}")
        lines.append(f"- 总时间: {result['elapsed_time']:.2f}秒")
        lines.append(f"- 平均时间: {result['avg_time_per_sample']:.3f}秒/样本")
        lines.append("")
    
    # 写入文件
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主评测流程"""
    
    print("\n" + "="*80)
    print("MRAG-Bench完整评测 - 所有方法 + 7个核心指标")
    print("="*80)
    
    # 1. 加载数据集
    dataset = load_mragbench_dataset(
        CONFIG['dataset_path'],
        max_samples=CONFIG['max_samples']
    )
    
    # 2. 初始化Qwen3-VL
    print(f"\n{'='*80}")
    print("初始化Qwen3-VL-8B-Instruct...")
    print(f"{'='*80}")
    
    qwen3_vl = create_qwen3_vl_wrapper(
        model_path=CONFIG['qwen3_vl_path'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 3. 初始化检索器
    retriever = init_retriever(CONFIG)
    
    # 4. 初始化所有方法
    methods = init_all_methods(qwen3_vl, retriever, CONFIG)
    
    # 5. 评测所有方法
    all_results = {}
    
    for method_name, method in methods.items():
        result = evaluate_single_method(
            method_name, method, dataset, CONFIG
        )
        all_results[method_name] = result
    
    # 6. 保存结果
    if CONFIG['save_results']:
        save_results(all_results, CONFIG)
    
    # 7. 最终总结
    print(f"\n{'='*80}")
    print("评测完成!")
    print(f"{'='*80}")
    print(f"\n共评测 {len(all_results)} 个方法")
    print(f"数据集: MRAG-Bench ({len(dataset)} 样本)")
    print(f"语料库: 3M")
    print(f"指标: 7个核心指标")
    print("\n结果已保存到:")
    print(f"  {CONFIG['output_dir']}")


if __name__ == '__main__':
    main()

