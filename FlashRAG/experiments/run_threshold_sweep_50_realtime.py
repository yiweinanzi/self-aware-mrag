#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阈值敏感性实验：50样本快速测试版（带实时日志输出）
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# 强制无缓冲输出
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, '/root/autodl-tmp/FlashRAG')

import datasets
from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
from flashrag.retriever import DenseRetriever
from flashrag.pipeline.self_aware_pipeline_qwen3vl import SelfAwarePipelineQwen3VL
from flashrag.evaluator.complete_metrics import CompleteMetricsCalculator


def log(msg):
    """带时间戳的日志输出，强制刷新"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {msg}", flush=True)


# ============================================================================
# 配置
# ============================================================================

CONFIG = {
    'dataset_path': '/root/autodl-tmp/FlashRAG/flashrag/data/MRAG-Bench/raw',
    'max_samples': 50,
    'qwen3_vl_path': '/root/autodl-tmp/models/Qwen3-VL-8B-Instruct',
    'index_path': '/root/autodl-tmp/FlashRAG/indexes/3m/bge/e5_Flat.index',
    'corpus_path': '/root/autodl-tmp/FlashRAG/corpus/corpus_3m.jsonl',
    'retrieval_model_path': '/root/autodl-tmp/models/bge-large-en-v1.5',
    'temperature': 0.01,
    'max_new_tokens': 10,
    'retrieval_topk': 5,
    'output_dir': '/root/autodl-tmp/FlashRAG/experiments/results_threshold_sweep_50',
}

# 测试的阈值列表
THRESHOLDS = [0.35, 0.45, 0.55]


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
    """加载数据集"""
    log(f"加载数据集: {dataset_path}")
    dataset_dict = datasets.load_from_disk(dataset_path)
    test_data = dataset_dict['test']
    
    if max_samples:
        test_data = test_data.select(range(min(max_samples, len(test_data))))
    
    samples = []
    for item in test_data:
        sample = {
            'question': item['question'],
            'image': item['image'],
            'answer': item['answer'],
            'A': item['A'],
            'B': item['B'],
            'C': item['C'],
            'D': item['D'],
        }
        samples.append(sample)
    
    log(f"✅ 加载完成: {len(samples)} 样本")
    return samples


def init_models():
    """初始化模型和检索器（只初始化一次）"""
    log("="*80)
    log("初始化模型和检索器")
    log("="*80)
    
    log("初始化Qwen3-VL...")
    qwen3_vl = create_qwen3_vl_wrapper(model_path=CONFIG['qwen3_vl_path'], device="cuda")
    log("✅ Qwen3-VL加载成功")
    
    log("初始化检索器...")
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
    log("✅ DenseRetriever加载成功")
    
    return qwen3_vl, retriever


def run_with_threshold(qwen3_vl, retriever, samples, threshold):
    """使用指定阈值运行Our Method"""
    log(f"{'='*80}")
    log(f"测试阈值 τ = {threshold}")
    log(f"{'='*80}")
    
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
    
    for i, sample in enumerate(samples, 1):
        result = pipeline.run_single(sample)
        result['question'] = sample['question']
        result['ground_truth'] = sample['answer']
        results.append(result)
        
        # 每10个样本输出一次进度
        if i % 10 == 0 or i == len(samples):
            log(f"进度: {i}/{len(samples)} ({i/len(samples)*100:.1f}%)")
    
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
        'dataset_name': 'mragbench',
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
    
    # 计算检索触发率
    retrieval_triggered = sum(1 for r in results if r.get('retrieved_docs'))
    metrics['retrieval_trigger_rate'] = retrieval_triggered / len(samples)
    
    log(f"结果:")
    log(f"  EM: {metrics.get('em', 0):.4f}")
    log(f"  F1: {metrics.get('f1', 0):.4f}")
    log(f"  VQA-Score: {metrics.get('vqa_score', 0):.4f}")
    log(f"  Recall@5: {metrics.get('retrieval_recall_top5', 0):.4f}")
    log(f"  检索触发率: {metrics['retrieval_trigger_rate']:.2%}")
    log(f"  时间: {metrics['seconds_per_sample']:.2f}秒/样本")
    
    return metrics, results


def main():
    """主函数"""
    log("="*80)
    log("阈值敏感性实验 - 50样本快速测试（实时日志版）")
    log("="*80)
    
    # 创建输出目录
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # 加载数据集
    samples = load_dataset(CONFIG['dataset_path'], CONFIG['max_samples'])
    
    # 初始化模型（只初始化一次）
    qwen3_vl, retriever = init_models()
    
    # 运行不同阈值
    all_metrics = {}
    all_results = {}
    
    for idx, threshold in enumerate(THRESHOLDS, 1):
        log(f"\n>>> 开始测试第 {idx}/{len(THRESHOLDS)} 个阈值: τ={threshold}")
        metrics, results = run_with_threshold(qwen3_vl, retriever, samples, threshold)
        all_metrics[f'tau_{threshold}'] = {
            'metrics': metrics,
            'threshold': threshold
        }
        all_results[f'tau_{threshold}'] = results
        log(f"<<< 完成阈值 τ={threshold} 的测试\n")
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results_file = os.path.join(CONFIG['output_dir'], f'threshold_sweep_results_{timestamp}.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    log(f"✅ 详细结果: {results_file}")
    
    # 生成报告
    report_file = os.path.join(CONFIG['output_dir'], f'THRESHOLD_REPORT_{timestamp}.md')
    generate_report(all_metrics, report_file)
    log(f"✅ 分析报告: {report_file}")
    
    log("="*80)
    log("阈值敏感性实验完成！")
    log("="*80)


def generate_report(all_metrics, report_file):
    """生成Markdown报告"""
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 阈值敏感性实验报告 - 50样本\n\n")
        f.write(f"**评测时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**样本数**: {CONFIG['max_samples']}\n")
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

