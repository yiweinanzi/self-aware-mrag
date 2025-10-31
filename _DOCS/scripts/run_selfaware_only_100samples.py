#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试修复后的Self-Aware-MRAG - 仅100样本
"""
import os
import sys
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 添加路径
sys.path.insert(0, '/root/autodl-tmp/FlashRAG')
sys.path.insert(0, '/root/autodl-tmp')

from datasets import load_from_disk
from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
from flashrag.pipeline.self_aware_pipeline_qwen3vl import SelfAwarePipelineQwen3VL

# 从主脚本导入检索器初始化函数
sys.path.insert(0, '/root/autodl-tmp/FlashRAG/experiments')
from run_all_baselines_100samples import initialize_retriever

print("=" * 80)
print("Self-Aware-MRAG 修复验证测试 (100样本)")
print("=" * 80)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# 加载数据
print("加载数据集...")
dataset = load_from_disk('/root/autodl-tmp/FlashRAG/flashrag/data/MRAG-Bench/raw')
test_samples = dataset['test'].select(range(100))
print(f"✅ 加载完成: {len(test_samples)} 样本\n")

# 初始化模型和检索器
print("初始化Qwen3-VL...")
qwen3_vl = create_qwen3_vl_wrapper(model_path='/root/autodl-tmp/models/Qwen3-VL-8B-Instruct')
print("✅ Qwen3-VL加载成功\n")

print("初始化检索器...")
config = {
    'index_path': '/root/autodl-tmp/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
    'corpus_path': '/root/autodl-tmp/FlashRAG/corpus/corpus_wiki_3m.jsonl',
    'retrieval_method': 'e5',
    'faiss_gpu': False,
}
retriever = initialize_retriever(config)
print("✅ 检索器加载成功\n")

# 初始化pipeline
pipeline_config = {
    'uncertainty_threshold': 0.0,  # Always Retrieve
    'use_improved_estimator': True,
    'use_position_fusion': True,
    'use_attribution': True,
}

print("初始化Self-Aware-MRAG Pipeline...")
pipeline = SelfAwarePipelineQwen3VL(
    qwen3_vl_wrapper=qwen3_vl,
    retriever=retriever,
    config=pipeline_config
)
print("✅ Pipeline初始化完成\n")

# 运行测试
print("=" * 80)
print("开始测试...")
print("=" * 80 + "\n")

results = []
from tqdm import tqdm
import time

start_time = time.time()

for i, sample in enumerate(tqdm(test_samples, desc="处理样本")):
    try:
        result = pipeline.run_single(sample)
        results.append(result)
    except Exception as e:
        print(f"\n⚠️ 样本 {i} 处理失败: {e}")
        continue

elapsed = time.time() - start_time

# 计算指标
from flashrag.evaluator.utils import EM, F1, Sub_EM

em_scores = []
f1_scores = []

for r in results:
    pred = r.get('answer', '').strip()
    gold = r.get('golden_answers', [])
    
    em_scores.append(EM(pred, gold))
    f1_scores.append(F1(pred, gold))

metrics = {
    'EM': sum(em_scores) / len(em_scores) if em_scores else 0,
    'F1': sum(f1_scores) / len(f1_scores) if f1_scores else 0,
}

print("\n" + "=" * 80)
print("✅ 测试完成！")
print("=" * 80)
print(f"样本数: {len(results)}")
print(f"总耗时: {elapsed:.1f}秒")
print(f"平均速度: {elapsed/len(results):.2f}秒/样本")
print(f"\n📊 指标结果:")
print(f"  EM: {metrics['EM']:.4f} ({metrics['EM']*100:.2f}%)")
print(f"  F1: {metrics['F1']:.4f} ({metrics['F1']*100:.2f}%)")

# 保存结果
import json
output_file = f"selfaware_fixed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_file, 'w') as f:
    json.dump({
        'config': pipeline_config,
        'metrics': metrics,
        'num_samples': len(results),
        'elapsed_time': elapsed,
        'avg_time_per_sample': elapsed / len(results),
        'sample_results': results[:5]  # 只保存前5个样本
    }, f, indent=2, ensure_ascii=False)

print(f"\n✅ 结果已保存到: {output_file}")
print("\n" + "=" * 80)
