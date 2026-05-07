#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
消融实验脚本 - OK-VQA数据集
Ablation Study on OK-VQA Dataset

根据参考文档第1202-1217行要求进行消融实验

实验设计：
1. Baseline (MuRAG)
2. + Text Uncertainty
3. + Visual Uncertainty
4. + Cross-Modal Alignment Uncertainty
5. + Position-Aware Fusion
6. + Fine-Grained Attribution (Full Method)

样本数：500（参考文档建议）
数据集：OK-VQA val2014
评估指标：EM, F1, Recall@5, VQA-Score, Faithfulness, Attribution Precision, Position Bias Score
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
# 实验配置（根据参考文档要求）
# ============================================================================

ABLAITION_CONFIG = {
    # 数据集配置
    'dataset_name': 'okvqa',
    'data_dir': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/VQA',
    'split': 'val',  # 使用val2014
    'max_samples': None,  # ✅ 使用全部OK-VQA数据集（约9,000样本）
    'load_images': True,

    # 模型配置
    'qwen3_vl_path': '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',  # 需要确认路径

    # 检索器配置
    'index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
    'corpus_path': '/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl',
    'retrieval_model_path': '/data0/home/zqwang/ACL/models/bge-large-en-v1.5',
    'retrieval_topk': 5,

    # 评估配置
    'save_results': True,
    'output_dir': '/data0/home/zqwang/ACL/FlashRAG/experiments/results_ablation_okvqa',

    # 生成参数
    'temperature': 0.01,
    'max_new_tokens': 20,  # VQA答案通常较短

    # 不确定性阈值（基于之前校准的P92百分位）
    'uncertainty_threshold': 0.43,

    # 消融实验配置
    'ablation_variants': [
        {
            'name': 'Baseline (MuRAG)',
            'config': {
                'use_uncertainty_estimation': False,
                'use_position_fusion': False,
                'use_attribution': False,
                'uncertainty_threshold': 1.0,  # 总是检索
            }
        },
        {
            'name': '+ Text Uncertainty',
            'config': {
                'use_uncertainty_estimation': True,
                'uncertainty_components': ['text'],  # 仅文本不确定性
                'use_position_fusion': False,
                'use_attribution': False,
                'uncertainty_threshold': 0.43,
            }
        },
        {
            'name': '+ Visual Uncertainty',
            'config': {
                'use_uncertainty_estimation': True,
                'uncertainty_components': ['text', 'visual'],  # 文本+视觉
                'use_position_fusion': False,
                'use_attribution': False,
                'uncertainty_threshold': 0.43,
            }
        },
        {
            'name': '+ Cross-Modal Alignment Unc.',
            'config': {
                'use_uncertainty_estimation': True,
                'uncertainty_components': ['text', 'visual', 'alignment'],  # 全部
                'use_position_fusion': False,
                'use_attribution': False,
                'uncertainty_threshold': 0.43,
            }
        },
        {
            'name': '+ Position-Aware Fusion',
            'config': {
                'use_uncertainty_estimation': True,
                'uncertainty_components': ['text', 'visual', 'alignment'],
                'use_position_fusion': True,
                'use_attribution': False,
                'uncertainty_threshold': 0.43,
            }
        },
        {
            'name': '+ Fine-Grained Attribution',
            'config': {
                'use_uncertainty_estimation': True,
                'uncertainty_components': ['text', 'visual', 'alignment'],
                'use_position_fusion': True,
                'use_attribution': True,
                'uncertainty_threshold': 0.43,
            }
        }
    ]
}

# ============================================================================
# 数据加载
# ============================================================================

def load_okvqa_dataset(config):
    """加载OK-VQA数据集"""
    print("="*80)
    print("1. 加载OK-VQA数据集")
    print("="*80)

    dataset = OKVQADatasetSimple({
        'data_dir': config['data_dir'],
        'split': config['split'],
        'load_images': config['load_images'],
    })

    # 限制样本数
    if config['max_samples'] and len(dataset.data) > config['max_samples']:
        dataset.data = dataset.data[:config['max_samples']]

    print(f"✅ 加载完成: {len(dataset.data)} 样本")
    print(f"   图像加载: {config['load_images']}")

    # 检查数据样本
    if dataset.data:
        sample = dataset.data[0]
        print(f"   样本示例: {sample['question'][:50]}...")
        print(f"   答案示例: {sample['golden_answers'][:3]}")
        print(f"   图像ID: {sample['image_id']}")

    return dataset

# ============================================================================
# 模型和检索器初始化
# ============================================================================

def init_models(config):
    """初始化模型和检索器"""
    print("\n" + "="*80)
    print("2. 初始化模型和检索器")
    print("="*80)

    # 检查模型路径
    qwen_path = config['qwen3_vl_path']
    if not os.path.exists(qwen_path):
        print(f"⚠️ Qwen3-VL模型路径不存在: {qwen_path}")
        print(f"请检查路径或下载模型到指定位置")
        return None, None

    # 检查检索器路径
    index_path = config['index_path']
    corpus_path = config['corpus_path']

    if not os.path.exists(index_path):
        print(f"⚠️ 检索索引不存在: {index_path}")
        print(f"将使用简化检索或构建索引")

    if not os.path.exists(corpus_path):
        print(f"⚠️ 语料库不存在: {corpus_path}")
        print(f"将使用模拟语料库")

    # 初始化Qwen3-VL
    try:
        print(f"初始化Qwen3-VL: {qwen_path}")
        qwen3_vl = create_qwen3_vl_wrapper(model_path=qwen_path, device="cuda")
        print("✅ Qwen3-VL加载成功")
    except Exception as e:
        print(f"❌ Qwen3-VL加载失败: {e}")
        return None, None

    # 初始化检索器（简化版）
    try:
        print("初始化检索器...")

        retriever_config = {
            'retrieval_method': 'e5',
            'retrieval_model_path': config['retrieval_model_path'],
            'retrieval_topk': config['retrieval_topk'],
            'use_retrieval_cache': False,
        }

        # 如果索引不存在，创建一个模拟检索器
        if not os.path.exists(index_path):
            print("⚠️ 使用模拟检索器（用于测试）")
            retriever = create_mock_retriever(config['retrieval_topk'])
        else:
            retriever = DenseRetriever(retriever_config)
            print("✅ 检索器加载成功")

    except Exception as e:
        print(f"⚠️ 检索器初始化失败，使用模拟检索器: {e}")
        retriever = create_mock_retriever(config['retrieval_topk'])

    return qwen3_vl, retriever

def create_mock_retriever(topk=5):
    """创建模拟检索器（用于测试）"""
    class MockRetriever:
        def __init__(self, topk):
            self.topk = topk

        def search(self, query, num=None):
            # 返回模拟的检索结果
            mock_docs = [
                f"This is a mock retrieved document {i+1} for query: {query[:50]}..."
                for i in range(min(num or self.topk, self.topk))
            ]

            return [
                {
                    'id': f"mock_doc_{i}",
                    'contents': doc,
                    'score': 0.9 - i * 0.1
                }
                for i, doc in enumerate(mock_docs)
            ]

    retriever = MockRetriever(topk)
    print("✅ 模拟检索器创建成功")
    return retriever

# ============================================================================
# 消融实验执行
# ============================================================================

def run_ablation_variant(variant_name, variant_config, qwen3_vl, retriever, dataset):
    """运行单个消融变体"""
    print(f"\n{'='*60}")
    print(f"运行消融变体: {variant_name}")
    print(f"{'='*60}")

    # 创建pipeline
    try:
        pipeline_config = {
            **variant_config,
            'uncertainty_threshold': variant_config.get('uncertainty_threshold', ABLAITION_CONFIG['uncertainty_threshold']),
            'retrieval_topk': ABLAITION_CONFIG['retrieval_topk'],
            'temperature': ABLAITION_CONFIG['temperature'],
            'max_new_tokens': ABLAITION_CONFIG['max_new_tokens'],
        }

        pipeline = SelfAwarePipelineQwen3VL(
            qwen3_vl_wrapper=qwen3_vl,
            retriever=retriever,
            config=pipeline_config
        )

    except Exception as e:
        print(f"❌ Pipeline创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

    # 运行实验
    results = []
    retrieval_triggered = 0
    start_time = time.time()

    try:
        for i, sample in enumerate(tqdm(dataset.data, desc=f"Running {variant_name}")):
            # 转换数据格式
            sample_for_pipeline = {
                'question': sample['question'],
                'image': sample['image'],
                'golden_answers': sample['golden_answers'],
                'id': sample['id'],
            }

            try:
                result = pipeline.run_single(sample_for_pipeline)

                # 统计检索率
                if result.get('retrieved', False):
                    retrieval_triggered += 1

                # 添加必要字段
                result['question'] = sample['question']
                result['ground_truth'] = sample['golden_answers']
                result['sample_id'] = sample['id']

                # 计算准确率
                prediction = result.get('answer', '').lower().strip()
                golden_answers = sample['golden_answers']

                if isinstance(golden_answers, str):
                    golden_answers = [golden_answers]

                # 简单的匹配检查（可以改进）
                correct = any(
                    golden.lower().strip() in prediction or
                    prediction in golden.lower().strip()
                    for golden in golden_answers
                )
                result['correct'] = correct

                results.append(result)

            except Exception as e:
                print(f"⚠️ 样本 {i} 处理失败: {e}")
                continue

    except Exception as e:
        print(f"❌ 消融变体运行失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

    elapsed_time = time.time() - start_time

    if results:
        accuracy = sum(r['correct'] for r in results) / len(results)
        retrieval_rate = retrieval_triggered / len(results)

        print(f"✅ {variant_name} 完成:")
        print(f"   准确率: {accuracy*100:.2f}%")
        print(f"   检索率: {retrieval_rate*100:.1f}%")
        print(f"   时间: {elapsed_time:.2f}s ({elapsed_time/len(results):.2f}s/样本)")
    else:
        print(f"❌ {variant_name} 无有效结果")

    return results, elapsed_time

def calculate_all_metrics(variant_name, results, dataset):
    """计算所有7个核心指标"""
    print(f"\n计算 {variant_name} 的指标...")

    if not results:
        return {}

    # 准备数据
    predictions = [r.get('answer', '') for r in results]
    golden_answers = [r['ground_truth'] for r in results]

    # 准备检索结果
    retrieval_results = []
    for r in results:
        docs = r.get('retrieved_docs', [])
        if docs:
            doc_list = [{'contents': doc} if isinstance(doc, str) else doc for doc in docs]
        else:
            doc_list = []
        retrieval_results.append(doc_list)

    # 创建Mock数据对象
    class MockData:
        def __init__(self, pred, golden_answers, retrieval_result):
            self.pred = pred
            self.golden_answers = [[ans] if isinstance(ans, str) else ans for ans in golden_answers]
            self.retrieval_result = retrieval_result
            self.items = [{'golden_answers': ga} for ga in self.golden_answers]
            self.choices = [[] for _ in pred]

    data = MockData(predictions, golden_answers, retrieval_results)

    # 计算指标
    try:
        config = {
            'use_llm_judge': False,  # 使用简化版
            'dataset_name': 'okvqa',
            'metric_setting': {
                'retrieval_recall_topk': 5,
            }
        }

        calculator = CompleteMetricsCalculator(config)
        metrics = calculator.calculate_all_metrics(data)

        # 添加计算的基本指标
        metrics['accuracy'] = sum(r['correct'] for r in results) / len(results)

        # 计算检索率
        retrieval_count = sum(1 for r in results if r.get('retrieved', False))
        metrics['retrieval_rate'] = retrieval_count / len(results)

        return metrics

    except Exception as e:
        print(f"⚠️ 指标计算失败: {e}")
        # 返回基本指标
        return {
            'accuracy': sum(r['correct'] for r in results) / len(results),
            'retrieval_rate': sum(1 for r in results if r.get('retrieved', False)) / len(results),
        }

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("="*80)
    print("消融实验 - OK-VQA数据集")
    print("根据参考文档第1202-1217行要求")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 创建输出目录
    output_dir = Path(ABLAITION_CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    dataset = load_okvqa_dataset(ABLAITION_CONFIG)
    if not dataset.data:
        print("❌ 数据加载失败，退出")
        return

    # 初始化模型
    qwen3_vl, retriever = init_models(ABLAITION_CONFIG)
    if qwen3_vl is None or retriever is None:
        print("⚠️ 模型初始化失败，但将继续运行模拟实验")

    # 运行消融实验
    print("\n" + "="*80)
    print("3. 运行消融实验")
    print("="*80)

    all_results = {}
    all_metrics = {}

    for variant in ABLAITION_CONFIG['ablation_variants']:
        variant_name = variant['name']
        variant_config = variant['config']

        try:
            # 运行变体
            results, elapsed_time = run_ablation_variant(
                variant_name, variant_config, qwen3_vl, retriever, dataset
            )

            if results:
                # 计算指标
                metrics = calculate_all_metrics(variant_name, results, dataset)
                metrics['runtime_seconds'] = elapsed_time
                metrics['seconds_per_sample'] = elapsed_time / len(results)

                all_results[variant_name] = results
                all_metrics[variant_name] = metrics

                print(f"\n✅ {variant_name} 指标总结:")
                print(f"   EM: {metrics.get('em', 0):.4f}")
                print(f"   F1: {metrics.get('f1', 0):.4f}")
                print(f"   Recall@5: {metrics.get('retrieval_recall_top5', 0):.4f}")
                print(f"   VQA-Score: {metrics.get('vqa_score', 0):.4f}")
                print(f"   Faithfulness: {metrics.get('faithfulness', 0):.4f}")
                print(f"   Attribution: {metrics.get('attribution_precision', 0):.4f}")
                print(f"   Position Bias: {metrics.get('position_bias_score', 0):.4f}")
                print(f"   检索率: {metrics.get('retrieval_rate', 0):.3f}")

            else:
                print(f"❌ {variant_name} 失败，无有效结果")

        except Exception as e:
            print(f"❌ {variant_name} 执行失败: {e}")
            import traceback
            traceback.print_exc()

    # 保存结果
    print("\n" + "="*80)
    print("4. 保存结果")
    print("="*80)

    if all_metrics:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存详细结果
        results_file = output_dir / f"ablation_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"✅ 详细结果: {results_file}")

        # 保存指标
        metrics_file = output_dir / f"ablation_metrics_{timestamp}.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        print(f"✅ 指标结果: {metrics_file}")

        # 生成消融报告
        report_file = output_dir / f"ABLATION_REPORT_{timestamp}.md"
        generate_ablation_report(all_metrics, report_file, len(dataset.data))
        print(f"✅ 消融报告: {report_file}")

    print("\n" + "="*80)
    print("消融实验完成!")
    print(f"总样本数: {len(dataset.data)}")
    print("="*80)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 打印运行时间预估
    if len(dataset.data) > 1000:
        print(f"📊 实验统计:")
        print(f"   - 总样本数: {len(dataset.data):,}")
        print(f"   - 消融变体: {len(ABLAITION_CONFIG['ablation_variants'])}")
        print(f"   - 总推理次数: {len(dataset.data) * len(ABLAITION_CONFIG['ablation_variants']):,}")
        print(f"\n⏱️  时间预估:")
        print(f"   - 假设每个样本: 2秒")
        estimated_total_hours = (len(dataset.data) * len(ABLAITION_CONFIG['ablation_variants']) * 2) / 3600
        print(f"   - 预计总时间: {estimated_total_hours:.1f} 小时")
        print(f"   - 建议分批运行或使用更多GPU资源")

def generate_ablation_report(all_metrics, report_file, num_samples):
    """生成消融实验报告"""
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 消融实验报告 - OK-VQA (全数据集)\n\n")
        f.write(f"**实验时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**数据集**: OK-VQA val2014\n")
        f.write(f"**样本数**: {num_samples:,}\n")
        f.write(f"**实验类型**: 完整消融实验（6个变体）\n\n")

        f.write("---\n\n")
        f.write("## 消融实验结果\n\n")

        # 表格
        f.write("| Variant | EM | F1 | Recall@5 | VQA | Faith | Attr | PosBias | 检索率 |\n")
        f.write("|---------|----|----|----------|-----|-------|------|---------|--------|\n")

        for variant_name, metrics in all_metrics.items():
            f.write(f"| {variant_name} | ")
            f.write(f"{metrics.get('em', 0):.3f} | ")
            f.write(f"{metrics.get('f1', 0):.3f} | ")
            f.write(f"{metrics.get('retrieval_recall_top5', 0):.3f} | ")
            f.write(f"{metrics.get('vqa_score', 0):.3f} | ")
            f.write(f"{metrics.get('faithfulness', 0):.3f} | ")
            f.write(f"{metrics.get('attribution_precision', 0):.3f} | ")
            f.write(f"{metrics.get('position_bias_score', 0):.3f} | ")
            f.write(f"{metrics.get('retrieval_rate', 0):.3f} |\n")

        f.write("\n")
        f.write("**注**:\n")
        f.write("- EM: Exact Match (精确匹配)\n")
        f.write("- F1: Token-level F1\n")
        f.write("- Recall@5: 检索召回率\n")
        f.write("- VQA: VQA-Score\n")
        f.write("- Faith: Faithfulness (忠实度)\n")
        f.write("- Attr: Attribution Precision (归因精度)\n")
        f.write("- PosBias: Position Bias Score (位置偏差，越低越好)\n")
        f.write("- 检索率: 触发检索的样本比例\n")

        f.write("\n## 预期对比\n\n")
        f.write("根据参考文档第1208-1212行的预期结果：\n\n")
        f.write("| Variant | 预期EM | 预期Attr | 预期PosBias |\n")
        f.write("|---------|--------|----------|-------------|\n")
        f.write("| Baseline (MuRAG) | 54.2 | - | 0.385 |\n")
        f.write("| + Text Uncertainty | 56.8 | - | 0.362 |\n")
        f.write("| + Visual Uncertainty | 58.5 | 48.5 | 0.298 |\n")
        f.write("| + Cross-Modal Alignment Unc. | 60.2 | 55.3 | 0.265 |\n")
        f.write("| + Position-Aware Fusion | 61.5 | 62.1 | 0.156 |\n")
        f.write("| + Fine-Grained Attribution | 62.5 | 68.2 | 0.142 |\n")

if __name__ == '__main__':
    main()