#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试两个数据集的完整对比实验
Test Full Comparison on Two Datasets

在OK-VQA和MRAG-Bench上运行7个方法的对比，使用10个样本
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# 设置环境
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

print("="*80)
print("测试两个数据集的完整对比实验")
print("数据集: OK-VQA, MRAG-Bench")
print("方法: 7种基线方法")
print("样本数: 10个/数据集")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# 导入所有需要的模块
print("\n1. 导入模块...")
try:
    from flashrag.dataset.unified_dataset_loader import load_unified_dataset
    from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
    from flashrag.evaluator.unified_evaluator import evaluate_unified
    print("✓ 核心模块导入成功")

    # 导入基线方法
    exec(open('/data0/home/zqwang/ACL/FlashRAG/experiments/run_all_baselines_100samples.py').read().split('# ============================================================================
# 评测主函数')[0])
    print("✓ 基线方法导入成功")

except Exception as e:
    print(f"✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试配置
TEST_CONFIG = {
    'max_samples': 10,
    'qwen3_vl_path': '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
    'retrieval_topk': 3,  # 减少检索量以节省时间
    'temperature': 0.1,
    'max_new_tokens': 20,
    'save_results': True,
    'output_dir': '/data0/home/zqwang/ACL/FlashRAG/experiments/test_two_datasets_results'
}

# 数据集列表
datasets = ['okvqa', 'mrag-bench']

# 方法列表
methods = [
    'Self-Aware-MRAG',
    'SAM-RAG',
    'mR2AG',
    'VisRAG',
    'ViDoRAG',
    'RagVL',
    'MuRAG'
]

# 初始化模型（全局共享）
print("\n2. 初始化模型...")
try:
    qwen3_vl = create_qwen3_vl_wrapper(
        model_path=TEST_CONFIG['qwen3_vl_path'],
        device="cuda"
    )
    print("✓ Qwen3-VL初始化成功")
except Exception as e:
    print(f"✗ 模型初始化失败: {e}")
    sys.exit(1)

# 创建模拟检索器
class MockRetriever:
    """模拟检索器，避免索引问题"""
    def __init__(self):
        self.top_k = TEST_CONFIG['retrieval_topk']

    def search(self, question, num=None):
        num = num or self.top_k
        docs = []
        for i in range(num):
            docs.append(f"Mock document {i+1} related to {question[:30]}...")
        return docs

    def retrieve(self, query_text=None, query_image=None, top_k=None, return_score=True):
        num = top_k or self.top_k
        docs = []
        scores = []
        for i in range(num):
            docs.append(f"Mock document {i+1} for query")
            scores.append(0.9 - i * 0.1)

        if return_score:
            return docs, scores
        else:
            return docs

print("✓ 使用模拟检索器")

# 创建输出目录
output_dir = Path(TEST_CONFIG['output_dir'])
output_dir.mkdir(parents=True, exist_ok=True)

# 存储所有结果
all_results = {}

# 测试每个数据集
for dataset_name in datasets:
    print(f"\n{'='*60}")
    print(f"测试数据集: {dataset_name.upper()}")
    print(f"{'='*60}")

    # 加载数据集
    try:
        dataset = load_unified_dataset(
            dataset_name,
            split='val',
            max_samples=TEST_CONFIG['max_samples']
        )

        print(f"✓ 加载了 {len(dataset)} 个样本")

        # 转换为所需格式
        samples = []
        for item in dataset.data:
            if dataset_name == 'mrag-bench':
                sample = {
                    'question': item.get('question', ''),
                    'A': item.get('A', ''),
                    'B': item.get('B', ''),
                    'C': item.get('C', ''),
                    'D': item.get('D', ''),
                    'answer': item.get('golden_answers', [''])[0],
                    'image': item.get('image'),
                    'golden_answers': item.get('golden_answers', []),
                    'scenario': item.get('scenario', 'Unknown')
                }
            else:
                sample = {
                    'question': item.get('question', ''),
                    'answer': item.get('golden_answers', [''])[0],
                    'image': item.get('image'),
                    'golden_answers': item.get('golden_answers', [])
                }
            samples.append(sample)

    except Exception as e:
        print(f"✗ 数据集加载失败: {e}")
        continue

    # 存储当前数据集的结果
    dataset_results = {}

    # 测试每个方法
    for method_name in methods:
        print(f"\n--- 测试方法: {method_name} ---")

        try:
            # 创建pipeline
            if method_name == 'Self-Aware-MRAG':
                # Self-Aware-MRAG需要特殊��置
                from flashrag.pipeline.self_aware_pipeline_qwen3vl import SelfAwarePipelineQwen3VL

                # 创建模拟的多模态检索器
                class MockMultimodalRetriever:
                    def __init__(self, config, text_retriever=None, visual_retriever=None):
                        self.config = config

                    def search(self, question, num=None):
                        return MockRetriever().search(question, num)

                pipeline = SelfAwarePipelineQwen3VL(
                    qwen3_vl_wrapper=qwen3_vl,
                    retriever=MockMultimodalRetriever(config={}),
                    config={
                        'uncertainty_threshold': 0.5,
                        'use_improved_estimator': False,  # 简化配置
                        'retrieval_topk': TEST_CONFIG['retrieval_topk'],
                        'thinking': False
                    }
                )

            elif method_name == 'ViDoRAG':
                pipeline = create_vidorag_pipeline(qwen3_vl, MockRetriever(), TEST_CONFIG)
            else:
                # 其他方法使用统一的创建方式
                pipeline_map = {
                    'SAM-RAG': SAMRAGPipeline,
                    'mR2AG': MR2AGPipeline,
                    'VisRAG': VisRAGPipeline,
                    'RagVL': RagVLPipeline,
                    'MuRAG': MuRAGPipeline
                }

                if method_name in pipeline_map:
                    pipeline = pipeline_map[method_name](qwen3_vl, MockRetriever(), TEST_CONFIG)
                else:
                    print(f"✗ 未知方法: {method_name}")
                    continue

            # 运行预测
            predictions = []
            start_time = time.time()

            for i, sample in enumerate(samples):
                try:
                    result = pipeline.run_single(sample)

                    # 格式化预测结果
                    pred = {
                        'id': i,
                        'question': sample['question'],
                        'answer': result.get('answer', ''),
                        'golden_answers': sample.get('golden_answers', []),
                        'retrieved_docs': result.get('retrieved_docs', []),
                        'retrieval_result': [{
                            'retrieved_docs': result.get('retrieved_docs', []),
                            'retrieval_scores': [0.9] * len(result.get('retrieved_docs', [])),
                            'retrieval_used': result.get('used_retrieval', False)
                        }],
                        'attributions': result.get('attributions', {}),
                        'position_bias_results': result.get('position_bias_results', {}),
                        'used_retrieval': result.get('used_retrieval', False)
                    }

                    # 添加数据集特定字段
                    if dataset_name == 'mrag-bench':
                        pred['scenario'] = sample.get('scenario', 'Unknown')

                    predictions.append(pred)

                except Exception as e:
                    print(f"  ⚠️ 样本 {i} 处理失败: {e}")
                    # 添加失败的结果
                    predictions.append({
                        'id': i,
                        'question': sample['question'],
                        'answer': '',
                        'golden_answers': sample.get('golden_answers', []),
                        'error': str(e)
                    })

            elapsed_time = time.time() - start_time

            # 准备参考答案
            references = []
            for sample in samples:
                ref = {
                    'question': sample.get('question', ''),
                    'golden_answers': sample.get('golden_answers', []),
                    'dataset': dataset_name
                }
                if dataset_name == 'mrag-bench':
                    ref['scenario'] = sample.get('scenario', 'Unknown')
                references.append(ref)

            # 计算评测指标
            print(f"  计算评测指标...")
            metrics = evaluate_unified(dataset_name, predictions, references)

            # 保存结果
            dataset_results[method_name] = {
                'metrics': metrics,
                'predictions': predictions if TEST_CONFIG['save_results'] else None,
                'elapsed_time': elapsed_time
            }

            # 打印关键指标
            print(f"  ✓ 完成 - 耗时: {elapsed_time:.1f}秒")
            print(f"    EM: {metrics.get('accuracy', 0)*100:.2f}%")
            print(f"    F1: {metrics.get('avg_F1', 0):.4f}")

        except Exception as e:
            print(f"  ✗ 方法 {method_name} 失败: {e}")
            dataset_results[method_name] = {'error': str(e)}

    # 保存当前数据集的结果
    all_results[dataset_name] = dataset_results

    # 立即保存数据集结果
    dataset_file = output_dir / f"{dataset_name}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(dataset_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"✓ 数据集结果已保存: {dataset_file}")

# 生成综合报告
print(f"\n{'='*60}")
print("生成综合报告")
print(f"{'='*60}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_file = output_dir / f"comparison_report_{timestamp}.md"

with open(report_file, 'w', encoding='utf-8') as f:
    f.write("# 两个数据集对比实验报告\n\n")
    f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"**数据集**: {', '.join(datasets)}\n")
    f.write(f"**方法**: {', '.join(methods)}\n")
    f.write(f"**样本数**: {TEST_CONFIG['max_samples']} 每个数据集\n\n")

    # 对比表
    f.write("## 性能对比表\n\n")
    f.write("| 数据集 | 方法 | EM(%) | F1 | 检索率(%) |\n")
    f.write("|--------|------|-------|----|-----------|\n")

    for dataset_name in datasets:
        if dataset_name not in all_results:
            continue

        for method_name in methods:
            if method_name not in all_results[dataset_name]:
                continue

            result = all_results[dataset_name][method_name]
            if 'metrics' in result:
                metrics = result['metrics']
                f.write(f"| {dataset_name.upper()} | {method_name} | ")
                f.write(f"{metrics.get('accuracy', 0)*100:.2f} | ")
                f.write(f"{metrics.get('avg_F1', 0):.4f} | ")
                f.write(f"{metrics.get('retrieval_rate', 0)*100:.1f} |\n")

    # 详细结果
    f.write("\n## 详细结果\n\n")
    for dataset_name in datasets:
        f.write(f"\n### {dataset_name.upper()}\n\n")

        if dataset_name not in all_results:
            f.write("测试失败\n\n")
            continue

        # 找出最佳方法
        best_method = None
        best_accuracy = 0.0
        for method_name, result in all_results[dataset_name].items():
            if 'metrics' in result:
                acc = result['metrics'].get('accuracy', 0)
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_method = method_name

        if best_method:
            f.write(f"**最佳方法**: {best_method} (EM: {best_accuracy*100:.2f}%)\n\n")

        f.write("所有方法:\n")
        for method_name, result in all_results[dataset_name].items():
            if 'metrics' in result:
                metrics = result['metrics']
                f.write(f"- {method_name}: EM {metrics.get('accuracy', 0)*100:.2f}%, "
                      f"F1 {metrics.get('avg_F1', 0):.4f}, "
                      f"检索率 {metrics.get('retrieval_rate', 0)*100:.1f}%\n")
            else:
                f.write(f"- {method_name}: 失败\n")

# 保存完整结果
complete_file = output_dir / f"all_results_{timestamp}.json"
with open(complete_file, 'w', encoding='utf-8') as f:
    json.dump({
        'test_info': {
            'datasets': datasets,
            'methods': methods,
            'max_samples': TEST_CONFIG['max_samples'],
            'timestamp': datetime.now().isoformat()
        },
        'results': all_results
    }, f, indent=2, ensure_ascii=False, default=str)

print(f"✓ 综合报告已生成: {report_file}")
print(f"✓ 完整结果已保存: {complete_file}")

# 总结
print(f"\n{'='*60}")
print("测试完成！")
print(f"{'='*60}")
print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 统计成功的方法
for dataset_name in datasets:
    if dataset_name in all_results:
        success_count = sum(1 for r in all_results[dataset_name].values() if 'metrics' in r)
        print(f"\n{dataset_name.upper()}:")
        print(f"  成功方法: {success_count}/{len(methods)}")
        print(f"  失败原因: 检查日志获取详细信息")

print("\n✓ 全链路测试完成！")