#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阈值敏感性实验（OK-VQA版本）
修复bug后的版本，使用OK-VQA数据集
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/root/autodl-tmp/FlashRAG')

import datasets
from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
from flashrag.retriever import DenseRetriever
from flashrag.pipeline.self_aware_pipeline_qwen3vl import SelfAwarePipelineQwen3VL
from flashrag.evaluator.complete_metrics import CompleteMetricsCalculator


# ============================================================================
# 配置
# ============================================================================

CONFIG = {
    'dataset_path': '/root/autodl-tmp/FlashRAG/flashrag/data/OK-VQA/raw',  # ← 改为OK-VQA
    'max_samples': 100,
    'qwen3_vl_path': '/root/autodl-tmp/models/Qwen3-VL-8B-Instruct',
    'index_path': '/root/autodl-tmp/FlashRAG/indexes/3m/bge/e5_Flat.index',
    'corpus_path': '/root/autodl-tmp/FlashRAG/corpus/corpus_3m.jsonl',
    'retrieval_model_path': '/root/autodl-tmp/models/bge-large-en-v1.5',
    'temperature': 0.01,
    'max_new_tokens': 10,
    'retrieval_topk': 5,
    'output_dir': '/root/autodl-tmp/FlashRAG/experiments/results_threshold_sweep_okvqa',
}

# 测试的阈值列表（基于之前50样本的结果）
THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45]


# ============================================================================
# MockData
# ============================================================================

class MockData:
    def __init__(self, predictions, golden_answers, retrieval_results):
        self.pred = predictions
        self.golden_answers = [[ans] if isinstance(ans, str) else ans for ans in golden_answers]
        self.retrieval_result = retrieval_results
        self.items = [{'golden_answers': ga} for ga in self.golden_answers]
        self.choices = [[] for _ in predictions]


# ============================================================================
# 主要函数
# ============================================================================

def load_dataset(dataset_path, max_samples):
    """加载OK-VQA数据集"""
    print(f"加载数据集: {dataset_path}")
    dataset_dict = datasets.load_from_disk(dataset_path)
    
    # OK-VQA使用validation split
    if 'validation' in dataset_dict:
        test_data = dataset_dict['validation']
    elif 'test' in dataset_dict:
        test_data = dataset_dict['test']
    else:
        raise ValueError(f"数据集中没有validation或test split: {dataset_dict.keys()}")
    
    if max_samples:
        test_data = test_data.select(range(min(max_samples, len(test_data))))
    
    samples = []
    for item in test_data:
        sample = {
            'question': item['question'],
            'image': item['image'],
            'answer': item['answer'] if isinstance(item['answer'], str) else item['answer'][0],
        }
        samples.append(sample)
    
    print(f"✅ 加载完成: {len(samples)} 样本（来自{list(dataset_dict.keys())}）")
    return samples


def init_models():
    """初始化模型和检索器（只初始化一次）"""
    print("\n" + "="*80)
    print("初始化模型和检索器")
    print("="*80)
    
    print("\n初始化Qwen3-VL...")
    qwen3_vl = create_qwen3_vl_wrapper(model_path=CONFIG['qwen3_vl_path'], device="cuda")
    print("✅ Qwen3-VL加载成功")
    
    print("\n初始化检索器...")
    retriever_config = {
        'index_path': CONFIG['index_path'],
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
    print("✅ DenseRetriever加载成功")
    
    return qwen3_vl, retriever


def run_with_threshold(qwen3_vl, retriever, samples, threshold):
    """使用指定阈值运行Our Method"""
    print(f"\n{'='*80}")
    print(f"测试阈值 τ = {threshold}")
    print(f"{'='*80}")
    
    # 创建pipeline
    pipeline = SelfAwarePipelineQwen3VL(
        qwen3_vl_wrapper=qwen3_vl,
        retriever=retriever,
        config={
            'uncertainty_threshold': threshold,
            'use_position_fusion': True,
            'use_attribution': False,
            'enable_multimodal_output': False,
        }
    )
    
    # 运行推理
    results = []
    start_time = time.time()
    
    from tqdm import tqdm
    for sample in tqdm(samples, desc=f"τ={threshold}"):
        result = pipeline.run_single(sample)
        result['question'] = sample['question']
        result['ground_truth'] = sample['answer']
        results.append(result)
    
    elapsed_time = time.time() - start_time
    
    # 计算指标
    predictions = [r['answer'] for r in results]
    golden_answers = [s['answer'] for s in samples]
    
    retrieval_results = []
    for r in results:
        docs = r.get('retrieved_docs', [])
        if docs:
            doc_list = [{'contents': doc} if isinstance(doc, str) else {'contents': str(doc)} for doc in docs]
        else:
            doc_list = []
        retrieval_results.append(doc_list)
    
    data = MockData(predictions, golden_answers, retrieval_results)
    
    config = {
        'use_llm_judge': False,
        'dataset_name': 'okvqa',
        'metric_setting': {
            'retrieval_recall_topk': 5,
        }
    }
    
    calculator = CompleteMetricsCalculator(config)
    metrics = calculator.calculate_all_metrics(data)
    
    # 添加运行时间
    metrics['runtime_seconds'] = elapsed_time
    metrics['seconds_per_sample'] = elapsed_time / len(samples)
    metrics['threshold'] = threshold
    
    # 计算检索触发率（修复后的版本）
    retrieval_triggered = sum(1 for r in results if r.get('retrieved_docs'))
    metrics['retrieval_trigger_rate'] = retrieval_triggered / len(samples)
    
    print(f"\n结果:")
    print(f"  EM: {metrics.get('em', 0):.4f}")
    print(f"  F1: {metrics.get('f1', 0):.4f}")
    print(f"  VQA-Score: {metrics.get('vqa_score', 0):.4f}")
    print(f"  Recall@5: {metrics.get('retrieval_recall_top5', 0):.4f}")
    print(f"  检索触发率: {metrics['retrieval_trigger_rate']:.2%}")
    print(f"  时间: {metrics['seconds_per_sample']:.2f}秒/样本")
    
    return metrics, results


def main():
    """主函数"""
    print("="*80)
    print("阈值敏感性实验 - OK-VQA版本")
    print("="*80)
    
    # 创建输出目录
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # 加载数据集
    samples = load_dataset(CONFIG['dataset_path'], CONFIG['max_samples'])
    
    # 初始化模型（只初始化一次）
    qwen3_vl, retriever = init_models()
    
    # 运行不同阈值
    all_metrics = {}
    all_results = {}
    
    for threshold in THRESHOLDS:
        metrics, results = run_with_threshold(qwen3_vl, retriever, samples, threshold)
        all_metrics[f'tau_{threshold}'] = {
            'metrics': metrics,
            'threshold': threshold
        }
        all_results[f'tau_{threshold}'] = results
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results_file = os.path.join(CONFIG['output_dir'], f'threshold_sweep_results_{timestamp}.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 详细结果: {results_file}")
    
    # 生成报告
    report_file = os.path.join(CONFIG['output_dir'], f'THRESHOLD_REPORT_{timestamp}.md')
    generate_report(all_metrics, report_file)
    print(f"✅ 分析报告: {report_file}")
    
    print("\n" + "="*80)
    print("阈值敏感性实验完成")
    print("="*80)


def generate_report(all_metrics, report_file):
    """生成Markdown报告"""
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 阈值敏感性实验报告（OK-VQA）\n\n")
        f.write(f"**评测时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**样本数**: {CONFIG['max_samples']}\n")
        f.write(f"**数据集**: OK-VQA\n")
        f.write(f"**方法**: Self-Aware-MRAG (Our Method)\n\n")
        
        f.write("---\n\n")
        f.write("## 核心指标对比\n\n")
        f.write("| 阈值(τ) | EM | F1 | Recall@5 | VQA | 检索触发率 | 时间(s) |\n")
        f.write("|---------|----|----|----------|-----|-----------|--------|\n")
        
        for tau_key in sorted(all_metrics.keys()):
            m = all_metrics[tau_key]['metrics']
            tau = all_metrics[tau_key]['threshold']
            f.write(f"| {tau:.2f} | {m.get('em', 0):.4f} | {m.get('f1', 0):.4f} | ")
            f.write(f"{m.get('retrieval_recall_top5', 0):.4f} | {m.get('vqa_score', 0):.4f} | ")
            f.write(f"{m.get('retrieval_trigger_rate', 0):.2%} | {m.get('seconds_per_sample', 0):.2f} |\n")
        
        # 找最佳配置
        best_tau = max(all_metrics.keys(), key=lambda k: all_metrics[k]['metrics'].get('f1', 0))
        best_metrics = all_metrics[best_tau]['metrics']
        best_threshold = all_metrics[best_tau]['threshold']
        
        f.write(f"\n## 最佳配置\n\n")
        f.write(f"**最佳阈值**: τ = {best_threshold:.2f}\n\n")
        f.write(f"- **EM**: {best_metrics.get('em', 0):.4f}\n")
        f.write(f"- **F1**: {best_metrics.get('f1', 0):.4f}\n")
        f.write(f"- **检索触发率**: {best_metrics.get('retrieval_trigger_rate', 0):.2%}\n")


if __name__ == '__main__':
    main()

