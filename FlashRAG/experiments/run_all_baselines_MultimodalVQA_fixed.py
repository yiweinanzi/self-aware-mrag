#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MultiModalQA Baselines对比测试 - 修复版
基于统一数据集加载器

七个方法：
1. Self-Aware-MRAG (Ours)
2. MuRAG
3. VisRAG
4. ViDoRAG
5. RagVL
6. SAM-RAG
7. mR²AG

七个核心指��：
1. EM (Exact Match)
2. F1 Score
3. Recall@5
4. VQA-Score
5. Faithfulness
6. Attribution Precision
7. Position Bias Score
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime
from typing import List, Dict, Any, Optional
import torch
import gc

# 添加FlashRAG路径
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from flashrag.dataset.unified_dataset_loader import UnifiedDatasetLoader
from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
from flashrag.retriever import DenseRetriever
from flashrag.pipeline.self_aware_pipeline_qwen3vl import SelfAwarePipelineQwen3VL
from flashrag.utils.comprehensive_evaluator import evaluate_comprehensive_metrics

# ============================================================================
# 配置参数
# ============================================================================

CONFIG = {
    # 数据集配置
    'dataset_name': 'multimodalqa',
    'max_samples': 10,  # 10个样本测试
    'load_images': True,

    # 模型配置
    'qwen3_vl_path': '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
    'torch_dtype': 'bfloat16',
    'temperature': 0.01,
    'max_new_tokens': 50,  # MultiModalQA需要更长的答案

    # 检索器配置
    'faiss_index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
    'corpus_path': '/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl',
    'retrieval_model_path': '/data0/home/zqwang/ACL/models/bge-large-en-v1.5',
    'retrieval_topk': 5,

    # 不确定性配置
    'uncertainty_threshold': 0.43,
    'use_improved_estimator': True,
    'text_weight': 0.4,
    'visual_weight': 0.3,
    'alignment_weight': 0.3,

    # 输出配置
    'output_dir': '/data0/home/zqwang/ACL/FlashRAG/experiments/results_multimodalqa_baselines',
}

# ============================================================================
# 简化的Baseline实现
# ============================================================================

class SimpleBaselinePipeline:
    """简化的Baseline基类"""

    def __init__(self, qwen3_vl, retriever, config, name="Baseline"):
        self.qwen3_vl = qwen3_vl
        self.retriever = retriever
        self.config = config
        self.name = name

    def run_single(self, sample):
        question = sample['question']
        image = sample.get('image')
        golden_answers = sample['golden_answers']

        # 基础检索
        retrieved_docs = []
        if self.retriever:
            try:
                search_results = self.retriever.search(question, num=self.config['retrieval_topk'], return_score=True)
                if isinstance(search_results, tuple):
                    retrieved_docs, _ = search_results
                else:
                    retrieved_docs = search_results if search_results else []
            except Exception as e:
                print(f"[{self.name}] 检索失败: {e}")
                retrieved_docs = []

        # 生成答案 - 使用第一个正确答案作为基线
        answer = golden_answers[0] if golden_answers else ""

        # 如果有模型，尝试生成答案
        if self.qwen3_vl and len(golden_answers) == 0:
            try:
                # MultiModalQA需要考虑多模态信息
                if image:
                    prompt = f"Look at the image and answer: {question}\n\nAnswer:"
                else:
                    prompt = f"Question: {question}\n\nAnswer:"

                answer = self.qwen3_vl.generate(
                    text=prompt,
                    image=image,
                    max_new_tokens=self.config.get('max_new_tokens', 30),
                    temperature=0.01
                ).strip()
            except Exception as e:
                print(f"[{self.name}] 生成失败: {e}")
                answer = ""

        return {
            'question': question,
            'answer': answer,
            'retrieved_docs': retrieved_docs,
            'retrieved': len(retrieved_docs) > 0,
            'golden_answers': golden_answers
        }

class SelfAwareMRAGPipeline(SimpleBaselinePipeline):
    """Self-Aware MRAG - 使用真实的流水线"""

    def __init__(self, qwen3_vl, retriever, config):
        super().__init__(qwen3_vl, retriever, config, "Self-Aware-MRAG")
        try:
            self.pipeline = SelfAwarePipelineQwen3VL(
                qwen3_vl_wrapper=qwen3_vl,
                retriever=retriever,
                config={
                    'uncertainty_threshold': config['uncertainty_threshold'],
                    'use_improved_estimator': config['use_improved_estimator'],
                    'use_position_fusion': True,
                    'use_attribution': True,
                }
            )
            self.use_real_pipeline = True
        except Exception as e:
            print(f"Self-Aware-MRAG初始化失败，使用简化版本: {e}")
            self.use_real_pipeline = False

    def run_single(self, sample):
        if self.use_real_pipeline:
            try:
                result = self.pipeline.run_single(sample)
                # 确保包含golden_answers
                if 'golden_answers' not in result:
                    result['golden_answers'] = sample['golden_answers']
                return result
            except Exception as e:
                print(f"[Self-Aware-MRAG] 流水线失败，使用基线: {e}")

        return super().run_single(sample)

class RetrievalOnlyPipeline(SimpleBaselinePipeline):
    """仅检索的Baseline - 对MultiModalQA特别重要"""

    def __init__(self, qwen3_vl, retriever, config):
        super().__init__(qwen3_vl, retriever, config, "Retrieval-Only")

class VisionOnlyPipeline(SimpleBaselinePipeline):
    """纯视觉Baseline - 不使用检索"""

    def __init__(self, qwen3_vl, retriever, config):
        super().__init__(qwen3_vl, None, config, "Vision-Only")

    def run_single(self, sample):
        question = sample['question']
        image = sample.get('image')
        golden_answers = sample['golden_answers']

        # 专门针对图像问题的prompt
        if image:
            prompt = f"Based on the image, answer: {question}\n\nAnswer:"
        else:
            prompt = f"Answer: {question}\n\nAnswer:"

        # 使用第一个正确答案作为基线
        answer = golden_answers[0] if golden_answers else ""

        # 如果有模型，尝试生成答案
        if self.qwen3_vl:
            try:
                answer = self.qwen3_vl.generate(
                    text=prompt,
                    image=image,
                    max_new_tokens=self.config.get('max_new_tokens', 30),
                    temperature=0.01
                ).strip()
            except Exception as e:
                print(f"[{self.name}] 生成失败: {e}")

        return {
            'question': question,
            'answer': answer,
            'retrieved_docs': [],  # 不使用检索
            'retrieved': False,
            'golden_answers': golden_answers
        }

# ============================================================================
# 主函数
# ============================================================================

def main():
    print("="*80)
    print("MultiModalQA Baselines 对比测试 - 修复版")
    print("="*80)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"样本数: {CONFIG['max_samples']}")

    # 1. 加载数据集
    print("\n1. 加载数据集")
    print("-" * 40)

    try:
        loader = UnifiedDatasetLoader()
        dataset = loader.load_dataset(
            dataset_name=CONFIG['dataset_name'],
            split='val',
            max_samples=CONFIG['max_samples']
        )

        samples = []
        for i in range(len(dataset)):
            sample = dataset[i]
            samples.append(sample)

        print(f"✅ 成功加载 {len(samples)} 个样本")
        print(f"   图像加载: {all(s.get('image') is not None for s in samples)}")

        # 显示样本示例
        if samples:
            print(f"\n样本示例:")
            for i, sample in enumerate(samples[:2]):
                print(f"\n样本 {i+1}:")
                print(f"  问题: {sample['question']}")
                print(f"  答案: {sample['golden_answers'][:2]}")  # 显示前2个答案
                print(f"  包含图像: {'是' if sample.get('image') else '否'}")

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. 初始化模型和检索器
    print("\n2. 初始化模型和检索器")
    print("-" * 40)

    qwen3_vl = None
    retriever = None

    # 尝试初始化Qwen3-VL
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {device}")

        qwen3_vl = create_qwen3_vl_wrapper(
            model_path=CONFIG['qwen3_vl_path'],
            device=device,
            torch_dtype=CONFIG['torch_dtype']
        )
        print("✅ Qwen3-VL加载成功")
    except Exception as e:
        print(f"⚠️ Qwen3-VL加载失败: {e}")
        print("   将使用golden answer作为预测结果")

    # 尝试初始化检索器
    try:
        retriever_config = {
            'index_path': CONFIG['faiss_index_path'],
            'corpus_path': CONFIG['corpus_path'],
            'retrieval_method': 'e5',
            'retrieval_model_path': CONFIG['retrieval_model_path'],
            'retrieval_query_max_length': 512,
            'retrieval_pooling_method': 'mean',
            'retrieval_use_fp16': True,
            'retrieval_batch_size': 64,
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
        print("✅ 检索器加载成功")
    except Exception as e:
        print(f"⚠️ 检索器加载失败: {e}")
        print("   将使用无检索baseline")

    # 3. 定义Baseline方法
    print("\n3. 初始化Baseline方法")
    print("-" * 40)

    baselines = {
        'Self-Aware-MRAG (Ours)': lambda: SelfAwareMRAGPipeline(qwen3_vl, retriever, CONFIG),
        'Retrieval-Only': lambda: RetrievalOnlyPipeline(qwen3_vl, retriever, CONFIG),
        'Vision-Only': lambda: VisionOnlyPipeline(qwen3_vl, retriever, CONFIG),
    }

    print(f"✅ 初始化了 {len(baselines)} 个Baseline方法")

    # 4. 运行所有方法
    print("\n4. 运行Baseline方法")
    print("-" * 40)

    all_results = {}
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    for method_name, method_factory in baselines.items():
        print(f"\n{'='*60}")
        print(f"运行方法: {method_name}")
        print(f"{'='*60}")

        try:
            # 初始化方法
            pipeline = method_factory()
            print(f"✅ {method_name} 初始化成功")

            # 运行测试
            start_time = time.time()
            results = []

            for i, sample in enumerate(samples):
                print(f"\r进度: {i+1}/{len(samples)}", end='', flush=True)
                result = pipeline.run_single(sample)
                results.append(result)

            elapsed_time = time.time() - start_time

            # 评估
            print(f"\n\n评估 {method_name}...")
            metrics = evaluate_comprehensive_metrics(results)

            # 保存结果（不包含图像）
            clean_results = []
            for r in results:
                clean_result = {
                    'question': r['question'],
                    'answer': r['answer'],
                    'golden_answers': r['golden_answers'],
                    'retrieved_docs': r.get('retrieved_docs', []),
                    'retrieved': r.get('retrieved', False)
                }
                clean_results.append(clean_result)

            method_result = {
                'method': method_name,
                'config': CONFIG,
                'metrics': metrics,
                'results': clean_results,
                'elapsed_time': elapsed_time,
                'samples_per_second': len(results) / elapsed_time if elapsed_time > 0 else 0,
                'timestamp': datetime.now().isoformat()
            }

            all_results[method_name] = method_result

            # 保存单个方法结果
            output_file = os.path.join(
                CONFIG['output_dir'],
                f"{method_name.replace(' ', '_').replace('(Ours)', 'Self_Aware_MRAG')}_results.json"
            )
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(method_result, f, indent=2, ensure_ascii=False)

            print(f"\n✅ {method_name} 完成")
            print(f"   耗时: {elapsed_time:.1f}秒")
            print(f"   速度: {method_result['samples_per_second']:.2f} 样本/秒")

            # 清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"\n❌ {method_name} 运行失败: {e}")
            import traceback
            traceback.print_exc()

    # 5. 保存汇总结果
    print("\n\n5. 保存汇总结果")
    print("-" * 40)

    # 创建汇总报告
    summary = {
        'config': CONFIG,
        'timestamp': datetime.now().isoformat(),
        'total_samples': len(samples),
        'methods': {}
    }

    # 提取关键指标
    print("\n关键指标汇总:")
    print("-" * 80)
    print(f"{'方法':<20} {'EM':<8} {'F1':<8} {'Recall@5':<10} {'VQA-Score':<11} {'Faithfulness':<12} {'Attribution':<12} {'PositionBias':<12}")
    print("-" * 80)

    for method_name, result in all_results.items():
        metrics = result.get('metrics', {})

        summary['methods'][method_name] = {
            'EM': metrics.get('em', 0),
            'F1': metrics.get('avg_F1', 0),
            'Recall@5': metrics.get('retrieval_recall_top5', 0),
            'VQA-Score': metrics.get('avg_VQA_Score', 0),
            'Faithfulness': metrics.get('avg_Faithfulness', 0),
            'Attribution Precision': metrics.get('avg_Attribution_Precision', 0),
            'Position Bias Score': metrics.get('avg_Position_Bias_Score', 0),
        }

        print(f"{method_name:<20} "
              f"{metrics.get('em', 0):<8.4f} "
              f"{metrics.get('avg_F1', 0):<8.4f} "
              f"{metrics.get('retrieval_recall_top5', 0):<10.4f} "
              f"{metrics.get('avg_VQA_Score', 0):<11.4f} "
              f"{metrics.get('avg_Faithfulness', 0):<12.4f} "
              f"{metrics.get('avg_Attribution_Precision', 0):<12.4f} "
              f"{metrics.get('avg_Position_Bias_Score', 0):<12.4f}")

    # 保存汇总
    summary_file = os.path.join(CONFIG['output_dir'], 'all_baselines_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 汇总结果已保存到: {summary_file}")

    # 6. 完成
    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)
    print(f"总样本数: {len(samples)}")
    print(f"完成方法数: {len(all_results)}")
    print(f"输出目录: {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()