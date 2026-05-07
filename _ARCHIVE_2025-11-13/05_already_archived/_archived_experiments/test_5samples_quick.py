#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试：5样本验证指标计算
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


# 配置
CONFIG = {
    'dataset_path': '/root/autodl-tmp/FlashRAG/flashrag/data/MRAG-Bench/raw',
    'max_samples': 5,  # 快速测试：5样本
    'qwen3_vl_path': '/root/autodl-tmp/models/Qwen3-VL-8B-Instruct',
    'index_path': '/root/autodl-tmp/FlashRAG/indexes/3m/bge/e5_Flat.index',
    'corpus_path': '/root/autodl-tmp/FlashRAG/corpus/corpus_3m.jsonl',
    'retrieval_model_path': '/root/autodl-tmp/models/bge-large-en-v1.5',
    'temperature': 0.01,
    'max_new_tokens': 10,
    'retrieval_topk': 5,
}


class MockData:
    """模拟数据对象"""
    def __init__(self, predictions, golden_answers, retrieval_results):
        self.pred = predictions
        self.golden_answers = [[ans] if isinstance(ans, str) else ans for ans in golden_answers]
        self.retrieval_result = retrieval_results
        self.items = [{'golden_answers': ga} for ga in self.golden_answers]
        # 修复：添加choices属性（空列表表示不是多选题格式）
        self.choices = [[] for _ in predictions]


def load_dataset(dataset_path, max_samples):
    """加载数据集"""
    print(f"加载数据集: {dataset_path}")
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
    
    print(f"✅ 加载完成: {len(samples)} 样本")
    return samples


def init_models():
    """初始化模型和检索器"""
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


def test_pipeline(qwen3_vl, retriever, samples):
    """测试pipeline运行"""
    print("\n" + "="*80)
    print("测试 Self-Aware-MRAG Pipeline")
    print("="*80)
    
    pipeline = SelfAwarePipelineQwen3VL(
        qwen3_vl_wrapper=qwen3_vl,
        retriever=retriever,
        config={
            'uncertainty_threshold': 0.30,
            'use_position_fusion': True,
            'use_attribution': False,
            'enable_multimodal_output': False,
        }
    )
    
    results = []
    for i, sample in enumerate(samples):
        print(f"\n处理样本 {i+1}/{len(samples)}...")
        result = pipeline.run_single(sample)
        result['question'] = sample['question']
        result['ground_truth'] = sample['answer']
        results.append(result)
        
        print(f"  预测: {result['answer']}")
        print(f"  标准: {sample['answer']}")
        print(f"  正确: {'✅' if result['answer'].lower() == sample['answer'].lower() else '❌'}")
    
    return results


def test_metrics(results, samples):
    """测试指标计算"""
    print("\n" + "="*80)
    print("测试指标计算（7个核心指标）")
    print("="*80)
    
    # 准备数据
    predictions = [r['answer'] for r in results]
    golden_answers = [s['answer'] for s in samples]
    
    # 修复：retrieval_result应该是文档列表，每个文档是dict
    retrieval_results = []
    for r in results:
        docs = r.get('retrieved_docs', [])
        # 转换为正确格式：列表的列表，每个元素是字典
        if docs:
            doc_list = [{'contents': doc} if isinstance(doc, str) else {'contents': str(doc)} for doc in docs]
        else:
            doc_list = []
        retrieval_results.append(doc_list)
    
    # 创建MockData对象
    data = MockData(predictions, golden_answers, retrieval_results)
    
    # 计算所有指标
    print("\n创建CompleteMetricsCalculator...")
    config = {
        'use_llm_judge': False,
        'dataset_name': 'mragbench',
        'metric_setting': {
            'retrieval_recall_topk': 5,  # Recall@5
        }
    }
    
    try:
        calculator = CompleteMetricsCalculator(config)
        print("✅ CompleteMetricsCalculator创建成功")
    except Exception as e:
        print(f"❌ CompleteMetricsCalculator创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 计算指标
    print("\n计算所有指标...")
    try:
        metrics = calculator.calculate_all_metrics(data)
        print("✅ 指标计算成功")
        return metrics
    except Exception as e:
        print(f"❌ 指标计算失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    print("="*80)
    print("快速测试：5样本验证（确保指标计算正常）")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 1. 加载数据
    samples = load_dataset(CONFIG['dataset_path'], CONFIG['max_samples'])
    
    # 2. 初始化模型
    qwen3_vl, retriever = init_models()
    
    # 3. 测试pipeline
    results = test_pipeline(qwen3_vl, retriever, samples)
    
    # 4. 测试指标计算
    metrics = test_metrics(results, samples)
    
    # 5. 显示结果
    print("\n" + "="*80)
    print("测试结果")
    print("="*80)
    
    if metrics:
        print("\n✅ 所有测试通过！")
        print("\n核心指标结果:")
        print(f"  EM: {metrics.get('em', 0):.4f}")
        print(f"  F1: {metrics.get('f1', 0):.4f}")
        print(f"  Recall@5: {metrics.get('retrieval_recall_top5', 0):.4f}")
        print(f"  VQA-Score: {metrics.get('vqa_score', 0):.4f}")
        print(f"  Faithfulness: {metrics.get('faithfulness', 0):.4f}")
        print(f"  Attribution Precision: {metrics.get('attribution_precision', 0):.4f}")
        print(f"  Position Bias Score: {metrics.get('position_bias_score', 0):.4f}")
        
        print("\n" + "="*80)
        print("✅ 可以安全运行100样本完整实验！")
        print("="*80)
        return True
    else:
        print("\n❌ 测试失败！请检查错误信息")
        print("="*80)
        print("⚠️  不要运行完整实验，先修复问题！")
        print("="*80)
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

