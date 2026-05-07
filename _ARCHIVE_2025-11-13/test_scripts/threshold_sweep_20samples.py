#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Threshold参数扫描 - 快速测试（20样本）
找到最佳的uncertainty_threshold值
"""

import os
import sys
import json
import time
from datetime import datetime
from tqdm import tqdm

# 添加FlashRAG路径
sys.path.insert(0, '/root/autodl-tmp/FlashRAG')

import datasets
from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
from flashrag.retriever import DenseRetriever
from flashrag.pipeline.self_aware_pipeline_qwen3vl import SelfAwarePipelineQwen3VL

# 扫描的threshold值（从0.25开始）
THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

# 测试样本数
NUM_SAMPLES = 20

print("="*80)
print("Threshold参数扫描 - Self-Aware-MRAG")
print("="*80)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"测试样本数: {NUM_SAMPLES}")
print(f"扫描范围: {THRESHOLDS}")
print("="*80)

# 1. 加载数据集（只取前20个样本）
print("\n" + "="*80)
print("1. 加载数据集")
print("="*80)
dataset_path = '/root/autodl-tmp/FlashRAG/flashrag/data/MRAG-Bench/raw'
print(f"加载数据集: {dataset_path}")

dataset_dict = datasets.load_from_disk(dataset_path)
test_data = dataset_dict['test']
test_data = test_data.select(range(NUM_SAMPLES))

# 转换为列表
samples = []
for item in test_data:
    sample = {
        'question': item['question'],
        'image': item['image'],
        'answer': item['answer'],  # ✅ 修正字段名
        'answer_choice': item.get('answer_choice', ''),
    }
    samples.append(sample)

print(f"✅ 加载完成: {len(samples)} 样本")

# 2. 初始化模型（只初始化一次）
print("\n" + "="*80)
print("2. 初始化模型和检索器")
print("="*80)

print("初始化Qwen3-VL: /root/autodl-tmp/models/Qwen3-VL-8B-Instruct")
qwen3_vl = create_qwen3_vl_wrapper(
    model_path='/root/autodl-tmp/models/Qwen3-VL-8B-Instruct',
    device='cuda'
)
print("✅ Qwen3-VL加载成功")

print("初始化检索器...")
print("  模式: 纯文本 (BGE)")
text_retriever = DenseRetriever(
    config={
        'retrieval_method': 'e5',
        'model_path': '/root/autodl-tmp/models/bge-large-en-v1.5',
        'index_path': '/root/autodl-tmp/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
        'corpus_path': '/root/autodl-tmp/FlashRAG/corpus/corpus_wiki_3m.jsonl',
        'max_length': 512,
        'device': 'cuda',
    }
)
print("✅ BGE文本检索器加载成功")

# 3. 扫描不同threshold
results = {}

for threshold in THRESHOLDS:
    print(f"\n{'='*80}")
    print(f"测试 Threshold = {threshold}")
    print(f"{'='*80}")
    
    # 初始化pipeline
    pipeline = SelfAwarePipelineQwen3VL(
        qwen3_vl_wrapper=qwen3_vl,
        retriever=text_retriever,
        config={
            'uncertainty_threshold': threshold,
            'use_improved_estimator': False,  # 使用CrossModalUncertaintyEstimator
            'use_position_fusion': True,
            'use_attribution': True,
            'enable_multimodal_output': False,
            'clip_model_path': '/root/autodl-tmp/models/clip-vit-large-patch14-336',
            'retrieval_topk': 5,
            'max_new_tokens': 10,
            'temperature': 0.01,
        }
    )
    
    # 运行评测
    predictions = []
    ground_truths = []
    retrieval_count = 0
    
    for sample in tqdm(samples, desc=f"Threshold={threshold}"):
        try:
            # 运行pipeline
            result = pipeline.run(sample)
            
            # 记录结果
            pred = result.get('pred', result.get('answer', ''))
            predictions.append(pred)
            
            # 获取ground truth
            gt = sample['answer']
            ground_truths.append(gt if gt else '')
            
            # 统计检索次数
            if result.get('should_retrieve', False):
                retrieval_count += 1
                
        except Exception as e:
            print(f"\n⚠️  样本处理失败: {e}")
            predictions.append('')
            gt = sample['answer']
            ground_truths.append(gt if gt else '')
    
    # 计算EM和F1
    em_count = 0
    f1_scores = []
    
    for pred, gt in zip(predictions, ground_truths):
        # EM
        if pred.strip().lower() == gt.strip().lower():
            em_count += 1
        
        # F1
        pred_tokens = set(pred.strip().lower().split())
        gt_tokens = set(gt.strip().lower().split())
        
        if len(pred_tokens) == 0 and len(gt_tokens) == 0:
            f1 = 1.0
        elif len(pred_tokens) == 0 or len(gt_tokens) == 0:
            f1 = 0.0
        else:
            common = pred_tokens & gt_tokens
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(gt_tokens)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        f1_scores.append(f1)
    
    em = em_count / len(samples)
    f1_avg = sum(f1_scores) / len(f1_scores)
    retrieval_rate = retrieval_count / len(samples)
    
    # 保存结果
    results[threshold] = {
        'em': em,
        'f1': f1_avg,
        'retrieval_rate': retrieval_rate,
        'retrieval_count': retrieval_count,
    }
    
    print(f"\n结果:")
    print(f"  EM: {em:.2%}")
    print(f"  F1: {f1_avg:.4f}")
    print(f"  检索率: {retrieval_rate:.2%} ({retrieval_count}/{len(samples)})")

# 4. 汇总结果
print(f"\n\n{'='*80}")
print("参数扫描结果汇总")
print(f"{'='*80}")
print(f"{'Threshold':<12} {'EM':<10} {'F1':<10} {'检索率':<12} {'检索次数':<10}")
print("-"*80)

best_threshold = None
best_em = -1

for threshold in THRESHOLDS:
    res = results[threshold]
    marker = "  ← 最佳" if res['em'] > best_em else ""
    print(f"{threshold:<12.2f} {res['em']:<10.2%} {res['f1']:<10.4f} "
          f"{res['retrieval_rate']:<12.2%} {res['retrieval_count']:<10}{marker}")
    
    if res['em'] > best_em:
        best_em = res['em']
        best_threshold = threshold

print("-"*80)
print(f"✅ 推荐Threshold: {best_threshold} (EM: {best_em:.2%})")
print(f"{'='*80}")

# 5. 保存结果到文件
output_file = '/root/autodl-tmp/threshold_sweep_results.json'
with open(output_file, 'w') as f:
    json.dump({
        'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_samples': NUM_SAMPLES,
        'thresholds': THRESHOLDS,
        'results': results,
        'best_threshold': best_threshold,
        'best_em': best_em,
    }, f, indent=2)

print(f"\n✅ 结果已保存到: {output_file}")
print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
