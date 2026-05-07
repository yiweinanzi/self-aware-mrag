#!/usr/bin/env python3
"""
测试MRAG-Bench使用BGE Reranker的效果（仅文本检索）
"""

import sys
import os
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from datetime import datetime
import json
import numpy as np
from tqdm import tqdm

# 数据处理
import datasets
from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
from flashrag.retriever import DenseRetriever
from flashrag.pipeline.self_aware_pipeline_qwen3vl import SelfAwarePipelineQwen3VL
from flashrag.evaluator.complete_metrics import CompleteMetricsCalculator
from flashrag.utils.comprehensive_evaluator import evaluate_comprehensive_metrics

# 配置
CONFIG = {
    'dataset_name': 'mragbench',
    'dataset_path': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/MRAG-Bench/raw',
    'max_samples': 50,  # 测试样本数

    # 模型配置
    'qwen3_vl_path': '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',

    # 检索器配置（仅文本）
    'index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
    'corpus_path': '/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl',
    'retrieval_model_path': '/data0/home/zqwang/ACL/models/bge-large-en-v1.5',
    'retrieval_topk': 5,

    # Reranker配置
    'use_reranker': True,
    'rerank_model_name': 'bge-reranker-v2-m3',
    'rerank_model_path': '/data0/home/zqwang/ACL/models/bge-reranker-v2-m3',
    'rerank_topk': 5,
    'rerank_max_length': 512,
    'rerank_batch_size': 32,
    'rerank_use_fp16': True,

    # 其他配置
    'uncertainty_threshold': 0.35,
    'use_improved_estimator': True,
    'text_weight': 0.6,
    'visual_weight': 0.4,
    'alignment_weight': 0.2,
    'temperature': 0.01,
    'max_new_tokens': 10,

    # 输出
    'output_dir': f'/data0/home/zqwang/ACL/FlashRAG/experiments/results_mrag_reranker_only_{datetime.now().strftime("%m%d_%H%M")}',
    'save_detailed_results': True,
    'save_sample_results': True,
    'enable_complete_metrics': True,
}

def load_dataset(dataset_path, max_samples=None):
    """加载MRAG-Bench数据集"""
    print(f"加载数据集: MRAG-Bench")
    print(f"数据路径: {dataset_path}")
    print(f"最大样本数: {max_samples if max_samples else '全部'}")

    # 加载数据
    dataset = datasets.load_dataset('mrag-bench', cache_dir=dataset_path)
    samples = dataset['test']

    if max_samples:
        samples = samples.select(range(min(max_samples, len(samples))))

    print(f"✅ 加载成功: {len(samples)} 样本")
    return samples

def init_qwen3_vl(model_path):
    """初始化Qwen3-VL模型"""
    print(f"初始化Qwen3-VL: {model_path}")
    wrapper = create_qwen3_vl_wrapper(model_path=model_path, device="cuda")
    print("✅ Qwen3-VL加载成功")
    return wrapper

def init_retriever(config):
    """初始化BGE检索器（带reranker）"""
    print("初始化BGE检索器（带Reranker）...")

    retriever_config = {
        'index_path': config['index_path'],
        'corpus_path': config['corpus_path'],
        'retrieval_method': 'e5',
        'retrieval_model_path': config['retrieval_model_path'],
        'retrieval_query_max_length': 512,
        'retrieval_pooling_method': 'cls',
        'retrieval_use_fp16': True,
        'retrieval_batch_size': 128,
        'retrieval_topk': config['retrieval_topk'],
        'save_retrieval_cache': False,
        'use_retrieval_cache': False,
        'retrieval_cache_path': None,
        'use_reranker': config['use_reranker'],
        'rerank_model_name': config.get('rerank_model_name'),
        'rerank_model_path': config.get('rerank_model_path'),
        'rerank_topk': config.get('rerank_topk', 5),
        'rerank_max_length': config.get('rerank_max_length', 512),
        'rerank_batch_size': config.get('rerank_batch_size', 32),
        'rerank_use_fp16': config.get('rerank_use_fp16', True),
        'device': 'cuda',
        'use_sentence_transformer': False,
        'faiss_gpu': False,
        'instruction': '',
    }

    retriever = DenseRetriever(retriever_config)
    print(f"✅ BGE检索器加载成功 (Reranker: {'启用' if config['use_reranker'] else '禁用'})")
    return retriever

def main():
    """主函数"""
    print("="*80)
    print("MRAG-Bench Reranker测试（仅文本检索）")
    print("="*80)
    print(f"开始时间: {datetime.now()}")
    print(f"样本数: {CONFIG['max_samples']}")

    # 创建输出目录
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # 加载数据
    print("\n" + "="*80)
    print("1. 加载数据集")
    print("="*80)
    samples = load_dataset(CONFIG['dataset_path'], CONFIG['max_samples'])

    # 初始化模型和检索器
    print("\n" + "="*80)
    print("2. 初始化模型和检索器")
    print("="*80)
    qwen3_vl = init_qwen3_vl(CONFIG['qwen3_vl_path'])
    bge_retriever = init_retriever(CONFIG)

    # 创建pipeline（使用BGE检索器）
    print("\n" + "="*80)
    print("3. 创建Self-Aware-MRAG Pipeline")
    print("="*80)
    pipeline = SelfAwarePipelineQwen3VL(
        qwen3_vl_wrapper=qwen3_vl,
        retriever=bge_retriever,  # 使用带reranker的BGE检索器
        config={
            'uncertainty_threshold': CONFIG['uncertainty_threshold'],
            'use_improved_estimator': CONFIG['use_improved_estimator'],
            'use_position_fusion': True,
            'use_attribution': True,
            'enable_multimodal_output': False,
            'retrieval_topk': CONFIG['retrieval_topk'],
            'text_weight': CONFIG['text_weight'],
            'visual_weight': CONFIG['visual_weight'],
            'alignment_weight': CONFIG['alignment_weight'],
            'thinking': False,
            'max_images': 20,
            'temperature': CONFIG['temperature'],
            'max_new_tokens': CONFIG['max_new_tokens'],
        }
    )

    # 运行测试
    print("\n" + "="*80)
    print("4. 运行评测")
    print("="*80)
    print("评测方法: Self-Aware-MRAG (BGE + Reranker)")

    # 运行pipeline
    results = pipeline.run(samples)

    # 保存结果
    print("\n" + "="*80)
    print("5. 保存结果")
    print("="*80)

    # 保存详细结果
    all_results = {
        'dataset': 'MRAG-Bench',
        'method': 'Self-Aware-MRAG (BGE + Reranker)',
        'num_samples': len(samples),
        'config': {
            'use_reranker': CONFIG['use_reranker'],
            'rerank_model': CONFIG['rerank_model_name'],
            'retrieval_topk': CONFIG['retrieval_topk'],
            'rerank_topk': CONFIG['rerank_topk'],
        },
        'results': results,
        'timestamp': datetime.now().isoformat(),
    }

    # 保存完整结果
    with open(os.path.join(CONFIG['output_dir'], 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 计算和保存指标
    if CONFIG['enable_complete_metrics']:
        print("\n计算完整指标...")
        calculator = CompleteMetricsCalculator()
        metrics = evaluate_comprehensive_metrics(results, calculator)

        # 保存指标
        with open(os.path.join(CONFIG['output_dir'], 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        # 打印主要指标
        print("\n主要指标:")
        print(f"- 准确率: {metrics.get('accuracy', 0):.4f}")
        print(f"- EM: {metrics.get('exact_match', 0):.4f}")
        print(f"- F1: {metrics.get('f1', 0):.4f}")
        print(f"- VQA-Score: {metrics.get('vqa_score', 0):.4f}")

    print(f"\n✅ 结果已保存到: {CONFIG['output_dir']}")
    print("="*80)
    print(f"完成时间: {datetime.now()}")
    print("="*80)

if __name__ == "__main__":
    main()