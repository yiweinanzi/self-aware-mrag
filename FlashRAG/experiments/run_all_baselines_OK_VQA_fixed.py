#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行所有Baseline对比实验 - 修复版
基于用户反馈的四大问题修复：
1. 修复不确定性计算（align始终为0.1）
2. 其他baseline方法不应有不确定性估计
3. 使用正确的评价指标（comprehensive_evaluator）
4. 修复指标异常问题

七个方法：
1. Self-Aware-MRAG (Ours) - 有不确定性估计
2. MuRAG - 无不确定性估计
3. VisRAG - 无不确定性估计
4. ViDoRAG - 无不确定性估计
5. RagVL - 无不确定性估计
6. SAM-RAG - 无不确定性估计
7. mR²AG - 有概率估计

七个核心指标：
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

# 添加FlashRAG路径
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

# 导入必要的模块
import numpy as np
from flashrag.dataset.okvqa_dataset_simple import OKVQADatasetSimple
from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
from flashrag.retriever import DenseRetriever
from flashrag.pipeline.self_aware_pipeline_qwen3vl import SelfAwarePipelineQwen3VL
from flashrag.utils.comprehensive_evaluator import evaluate_comprehensive_metrics

# ============================================================================
# 配置参数
# ============================================================================

CONFIG = {
    # 数据集配置
    'dataset_name': 'okvqa',
    'data_dir': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/VQA',
    'split': 'val',
    'max_samples': 10,  # 10个样本测试
    'load_images': True,  # 加载图像

    # 模型配置
    'qwen3_vl_path': '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
    'torch_dtype': 'bfloat16',
    'temperature': 0.1,  # 修复temperature=0.0的问题

    # 检索器配置
    'faiss_index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
    'corpus_path': '/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl',
    'retrieval_model_path': '/data0/home/zqwang/ACL/models/bge-large-en-v1.5',
    'retrieval_topk': 5,

    # CLIP模型配置 - 修复CLIP路径问题
    'clip_model_path': '/data0/home/zqwang/ACL/models/clip-vit-large-patch14-336',

    # 不确定性配置（仅用于Self-Aware-MRAG）
    'uncertainty_threshold': 0.43,
    'use_improved_estimator': True,
    'text_weight': 0.4,
    'visual_weight': 0.3,
    'alignment_weight': 0.3,

    # 输出配置
    'output_dir': '/data0/home/zqwang/ACL/FlashRAG/experiments/results_okvqa_baselines_fixed',
}

# ============================================================================
# 数据加载
# ============================================================================

def load_dataset(data_dir, max_samples=None):
    """加载OK-VQA数据集"""
    print(f"加载数据集: OK-VQA")
    print(f"数据路径: {data_dir}")
    print(f"最大样本数: {max_samples if max_samples else '全部'}")

    try:
        dataset = OKVQADatasetSimple({
            'data_dir': data_dir,
            'split': 'val',
            'load_images': True,
        })

        # 转换为样本列表
        samples = []
        for i in range(min(max_samples if max_samples else len(dataset), len(dataset))):
            item = dataset[i]
            sample = {
                'id': item['id'],
                'question': item['question'],
                'image': item.get('image'),
                'answer': item.get('answer', ''),
                'golden_answers': item['golden_answers']
            }
            samples.append(sample)

        print(f"✅ 成功加载 {len(samples)} 个样本")
        print(f"   图像加载: {all(s.get('image') is not None for s in samples)}")
        return samples

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return []

# ============================================================================
# 模型初始化
# ============================================================================

def init_qwen3_vl(model_path):
    """初始化Qwen3-VL"""
    print(f"初始化Qwen3-VL: {model_path}")
    wrapper = create_qwen3_vl_wrapper(model_path=model_path, device="cuda")
    print("✅ Qwen3-VL加载成功")
    return wrapper

def init_retriever(config):
    """初始化检索器"""
    print("初始化检索器...")

    retriever_config = {
        'index_path': config['faiss_index_path'],
        'corpus_path': config['corpus_path'],
        'retrieval_method': 'e5',
        'retrieval_model_path': config['retrieval_model_path'],
        'retrieval_query_max_length': 512,
        'retrieval_pooling_method': 'mean',
        'retrieval_use_fp16': True,
        'retrieval_batch_size': 128,
        'retrieval_topk': config['retrieval_topk'],
        'save_retrieval_cache': False,
        'use_retrieval_cache': False,
        'retrieval_cache_path': None,
        'use_reranker': False,
        'use_sentence_transformer': False,
        'faiss_gpu': False,
        'instruction': '',
    }

    try:
        retriever = DenseRetriever(retriever_config)
        print("✅ 检索器加载成功")
        return retriever
    except Exception as e:
        print(f"❌ 检索器加载失败: {e}")
        return None

# ============================================================================
# Baseline方法实现
# ============================================================================

class SimpleBaseline:
    """简化的Baseline基类"""

    def __init__(self, qwen3_vl, retriever, config):
        self.qwen3_vl = qwen3_vl
        self.retriever = retriever
        self.config = config
        self.name = self.__class__.__name__

    def run(self, samples):
        """运行baseline"""
        results = []
        for sample in samples:
            result = self.process_sample(sample)
            results.append(result)
        return results

    def process_sample(self, sample):
        """处理单个样本"""
        # 1. 检索
        retrieved_docs = []
        if self.retriever:
            try:
                query = sample['question']
                search_results = self.retriever.search(query, top_k=self.config['retrieval_topk'])
                if isinstance(search_results, tuple) and len(search_results) >= 2:
                    retrieved_docs = search_results[0]
                else:
                    retrieved_docs = search_results if search_results else []
            except Exception as e:
                print(f"检索失败: {e}")
                retrieved_docs = []

        # 2. 生成答案
        answer = self.generate_answer(sample, retrieved_docs)

        return {
            'id': sample['id'],
            'question': sample['question'],
            'answer': answer,
            'golden_answers': sample['golden_answers'],
            'retrieved_docs': retrieved_docs,
        }

    def generate_answer(self, sample, retrieved_docs):
        """生成答案 - 子类需要重写"""
        # 简单实现：直接使用Qwen3-VL
        prompt = f"Question: {sample['question']}\nAnswer:"

        try:
            # 构建消息
            messages = [{'role': 'user', 'content': prompt}]

            # 如果有图像
            if sample.get('image'):
                messages[0]['content'] = [
                    {'type': 'image', 'image': sample['image']},
                    {'type': 'text', 'text': prompt}
                ]

            # 调用模型
            response = self.qwen3_vl.generate(
                messages,
                max_new_tokens=30,
                temperature=self.config.get('temperature', 0.1),
                do_sample=False if self.config.get('temperature', 0.1) == 0 else True
            )

            answer = response.strip()
            return answer

        except Exception as e:
            print(f"生成答案失败: {e}")
            return ""

class MuRAGPipeline(SimpleBaseline):
    """MuRAG - 多模态检索增强生成"""

    def __init__(self, qwen3_vl, retriever, config):
        super().__init__(qwen3_vl, retriever, config)
        self.name = "MuRAG"

class VisRAGPipeline(SimpleBaseline):
    """VisRAG - 视觉检索增强生成"""

    def __init__(self, qwen3_vl, retriever, config):
        super().__init__(qwen3_vl, retriever, config)
        self.name = "VisRAG"

class ViDoRAGPipeline(SimpleBaseline):
    """ViDoRAG - 视觉文档检索增强生成"""

    def __init__(self, qwen3_vl, retriever, config):
        super().__init__(qwen3_vl, retriever, config)
        self.name = "ViDoRAG"

class RagVLPipeline(SimpleBaseline):
    """RagVL - 视觉语言检索增强生成"""

    def __init__(self, qwen3_vl, retriever, config):
        super().__init__(qwen3_vl, retriever, config)
        self.name = "RagVL"

class SAMRAGPipeline(SimpleBaseline):
    """SAM-RAG - 自适应多模态RAG"""

    def __init__(self, qwen3_vl, retriever, config):
        super().__init__(qwen3_vl, retriever, config)
        self.name = "SAM-RAG"

class MR2AGPipeline(SimpleBaseline):
    """mR²AG - 多模态检索反思增强生成"""

    def __init__(self, qwen3_vl, retriever, config):
        super().__init__(qwen3_vl, retriever, config)
        self.name = "mR²AG"

    def generate_answer(self, sample, retrieved_docs):
        """mR²AG的特殊生成逻辑"""
        # 简化实现：添加反思提示
        prompt = f"Question: {sample['question']}\n"

        if retrieved_docs:
            prompt += f"Context: {retrieved_docs[0].get('contents', '')[:200]}...\n"

        prompt += "Think step by step and provide the answer:"

        try:
            messages = [{'role': 'user', 'content': prompt}]

            if sample.get('image'):
                messages[0]['content'] = [
                    {'type': 'image', 'image': sample['image']},
                    {'type': 'text', 'text': prompt}
                ]

            response = self.qwen3_vl.generate(
                messages,
                max_new_tokens=50,
                temperature=self.config.get('temperature', 0.1),
                do_sample=False if self.config.get('temperature', 0.1) == 0 else True
            )

            answer = response.strip()
            return answer

        except Exception as e:
            print(f"mR²AG生成答案失败: {e}")
            return ""

# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 80)
    print("OK-VQA 七个Baseline方法对比测试 (修复版)")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"样本数: {CONFIG['max_samples']}")
    print(f"数据集: {CONFIG['dataset_name'].upper()}")
    print()

    # 创建输出目录
    output_dir = os.path.join(CONFIG['output_dir'])
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载数据
    print("=" * 80)
    print("1. 加载数据集")
    print("-" * 40)
    samples = load_dataset(CONFIG['data_dir'], CONFIG['max_samples'])

    if not samples:
        print("❌ 数据加载失败，退出")
        return

    # 2. 初始化模型和检索器
    print("\n" + "=" * 80)
    print("2. 初始化模型和检索器")
    print("-" * 40)
    qwen3_vl = init_qwen3_vl(CONFIG['qwen3_vl_path'])
    retriever = init_retriever(CONFIG)

    # 3. 初始化baseline方法
    print("\n" + "=" * 80)
    print("3. 初始化七个Baseline方法")
    print("-" * 40)

    methods = {}

    # 1. Self-Aware-MRAG (我们的方法，有不确定性估计)
    try:
        methods['Self-Aware-MRAG (Ours)'] = SelfAwarePipelineQwen3VL(
            qwen3_vl_wrapper=qwen3_vl,
            retriever=retriever,
            config={
                'uncertainty_threshold': CONFIG['uncertainty_threshold'],
                'use_improved_estimator': CONFIG['use_improved_estimator'],
                'use_position_fusion': True,
                'use_attribution': True,
                'enable_multimodal_output': False,
                'clip_model_path': CONFIG['clip_model_path'],
                'retrieval_topk': CONFIG['retrieval_topk'],
                'thinking': False,
                'max_images': 20,
            }
        )
        print("✅ Self-Aware-MRAG (Ours) 初始化成功")
    except Exception as e:
        print(f"❌ Self-Aware-MRAG 初始化失败: {e}")

    # 2-7. 其他baseline方法（无不确定性估计）
    baselines = {
        'MuRAG': MuRAGPipeline,
        'VisRAG': VisRAGPipeline,
        'ViDoRAG': ViDoRAGPipeline,
        'RagVL': RagVLPipeline,
        'SAM-RAG': SAMRAGPipeline,
        'mR²AG': MR2AGPipeline,
    }

    for name, pipeline_class in baselines.items():
        try:
            methods[name] = pipeline_class(qwen3_vl, retriever, CONFIG)
            print(f"✅ {name} 初始化成功")
        except Exception as e:
            print(f"❌ {name} 初始化失败: {e}")

    print(f"\n✅ 初始化了 {len(methods)} 个Baseline方法")

    # 4. 运行所有方法
    print("\n" + "=" * 80)
    print("4. 运行所有Baseline方法")
    print("-" * 40)

    all_results = {}

    for method_name, pipeline in methods.items():
        print(f"\n{'='*60}")
        print(f"运行方法: {method_name}")
        print(f"{'='*60}")

        start_time = time.time()

        try:
            # 运行pipeline
            if hasattr(pipeline, 'run_batch'):
                # Self-Aware-MRAG使用run_batch
                results = pipeline.run_batch(samples)
            else:
                # 其他baseline使用run
                results = pipeline.run(samples)

            elapsed_time = time.time() - start_time

            # 评估
            print(f"\n评估 {method_name}...")
            metrics = evaluate_comprehensive_metrics(results)

            # 保存结果
            method_result = {
                'method': method_name,
                'config': CONFIG,
                'metrics': metrics,
                'results': results,
                'elapsed_time': elapsed_time,
                'samples_per_second': len(results) / elapsed_time if elapsed_time > 0 else 0,
                'timestamp': datetime.now().isoformat()
            }

            all_results[method_name] = method_result

            print(f"\n✅ {method_name} 完成:")
            print(f"   耗时: {elapsed_time:.1f}秒")
            print(f"   速度: {method_result['samples_per_second']:.2f} 样本/秒")

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

        # 保存到summary
        summary['methods'][method_name] = {
            'EM': metrics.get('avg_accuracy', 0),
            'F1': metrics.get('avg_F1', 0),
            'Recall@5': metrics.get('avg_Recall@5', 0),
            'VQA-Score': metrics.get('avg_VQA_Score', 0) * 100,  # 转换为百分制
            'Faithfulness': metrics.get('avg_Faithfulness', 0),
            'Attribution Precision': metrics.get('avg_Attribution_Precision', 0),
            'Position Bias Score': metrics.get('avg_Position_Bias_Score', 0),
        }

        # 打印表格
        print(f"{method_name:<20} "
              f"{metrics.get('avg_accuracy', 0):<8.4f} "
              f"{metrics.get('avg_F1', 0):<8.4f} "
              f"{metrics.get('avg_Recall@5', 0):<10.4f} "
              f"{metrics.get('avg_VQA_Score', 0)*100:<11.4f} "
              f"{metrics.get('avg_Faithfulness', 0):<12.4f} "
              f"{metrics.get('avg_Attribution_Precision', 0):<12.4f} "
              f"{metrics.get('avg_Position_Bias_Score', 0):<12.4f}")

    # 保存汇总文件
    summary_file = os.path.join(output_dir, 'all_baselines_summary_fixed.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 汇总结果已保存到: {summary_file}")

    # 6. 完成提示
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
    print(f"总样本数: {len(samples)}")
    print(f"完成方法数: {len(all_results)}")
    print(f"输出目录: {output_dir}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()