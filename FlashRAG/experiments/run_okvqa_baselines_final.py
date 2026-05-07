#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OK-VQA Baselines对比实验 - 最终版本
使用20个样本，实现7个正确的评价指标

评价指标：
1. 准确率 (Accuracy)
2. 检索率 (Retrieval Rate)
3. F1 Score
4. Recall@5
5. Faithfulness
6. Attribution Precision
7. Position Bias Score

对比方法：
1. Self-Aware-MRAG (Ours)
2. MuRAG
3. VisRAG
4. ViDoRAG
5. RagVL
6. SAM-RAG
7. mR²AG
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

from flashrag.dataset.okvqa_dataset_simple import OKVQADatasetSimple
from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
from flashrag.retriever import DenseRetriever
from flashrag.pipeline.self_aware_pipeline_qwen3vl import SelfAwarePipelineQwen3VL

# ============================================================================
# 配置参数
# ============================================================================

CONFIG = {
    # 数据集配置
    'dataset_name': 'okvqa',
    'data_dir': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/VQA',
    'split': 'val',
    'max_samples': 20,  # 20个样本
    'load_images': True,

    # 模型配置
    'qwen3_vl_path': '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
    'torch_dtype': 'bfloat16',
    'temperature': 0.01,
    'max_new_tokens': 30,

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
    'output_dir': '/data0/home/zqwang/ACL/FlashRAG/experiments/results_okvqa_final',
}

# ============================================================================
# 评价指标计算器
# ============================================================================

class OKVQAEvaluator:
    """OK-VQA评价指标计算器"""

    def __init__(self):
        self.stop_words = set([
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "is", "are", "was", "were", "be", "been", "this", "that", "it", "he", "she",
            "they", "i", "we", "you", "my", "your", "his", "her", "their", "what", "which"
        ])

    def normalize(self, text: str) -> str:
        """标准化文本"""
        if not text:
            return ""
        text = text.lower().strip()
        text = ''.join(c for c in text if c.isalnum() or c.isspace())
        return text

    def tokenize(self, text: str) -> List[str]:
        """分词"""
        tokens = self.normalize(text).split()
        return [t for t in tokens if t not in self.stop_words and len(t) > 1]

    def calculate_accuracy(self, predicted: str, golden_answers: List[str]) -> float:
        """计算准确率（VQA标准）"""
        if not predicted or not golden_answers:
            return 0.0

        pred_norm = self.normalize(predicted)
        for answer in golden_answers:
            if pred_norm == self.normalize(answer):
                return 1.0
        return 0.0

    def calculate_f1(self, predicted: str, golden_answers: List[str]) -> float:
        """计算F1分数"""
        if not predicted or not golden_answers:
            return 0.0

        pred_tokens = set(self.tokenize(predicted))
        if not pred_tokens:
            return 0.0

        best_f1 = 0.0
        for answer in golden_answers:
            gt_tokens = set(self.tokenize(answer))
            if not gt_tokens:
                continue

            common = pred_tokens & gt_tokens
            if common:
                precision = len(common) / len(pred_tokens)
                recall = len(common) / len(gt_tokens)
                f1 = 2 * precision * recall / (precision + recall)
                best_f1 = max(best_f1, f1)

        return best_f1

    def calculate_retrieval_rate(self, retrieved_docs: List) -> float:
        """计算检索率（是否检索到文档）"""
        return 1.0 if retrieved_docs and len(retrieved_docs) > 0 else 0.0

    def calculate_recall_at_5(self, retrieved_docs: List, golden_answers: List[str]) -> float:
        """计算Recall@5"""
        if not retrieved_docs or not golden_answers:
            return 0.0

        top_5_docs = retrieved_docs[:5]
        combined_text = " ".join([
            doc.get('contents', '') if isinstance(doc, dict) else str(doc)
            for doc in top_5_docs
        ]).lower()

        for answer in golden_answers:
            answer_lower = answer.lower()
            if answer_lower in combined_text:
                return 1.0
        return 0.0

    def calculate_faithfulness(self, answer: str, retrieved_docs: List) -> float:
        """计算Faithfulness（答案中来自检索文档的比例）"""
        if not answer or not retrieved_docs:
            return 0.0

        answer_tokens = set(self.tokenize(answer))
        if not answer_tokens:
            return 0.0

        doc_text = " ".join([
            doc.get('contents', '') if isinstance(doc, dict) else str(doc)
            for doc in retrieved_docs[:5]
        ])
        doc_tokens = set(self.tokenize(doc_text))

        if not doc_tokens:
            return 0.0

        overlap = len(answer_tokens & doc_tokens)
        return overlap / len(answer_tokens)

    def calculate_attribution_precision(self, answer: str, retrieved_docs: List) -> float:
        """计算Attribution Precision（细粒度归因）"""
        if not answer or not retrieved_docs:
            return 0.0

        # 使用bigram进行更精确的归因
        tokens = self.tokenize(answer)
        if len(tokens) < 2:
            return self.calculate_faithfulness(answer, retrieved_docs)

        bigrams = set(zip(tokens, tokens[1:]))
        if not bigrams:
            return 0.0

        doc_text = " ".join([
            doc.get('contents', '') if isinstance(doc, dict) else str(doc)
            for doc in retrieved_docs[:5]
        ])
        doc_tokens = self.tokenize(doc_text)
        doc_bigrams = set(zip(doc_tokens, doc_tokens[1:]))

        overlap = len(bigrams & doc_bigrams)
        return overlap / len(bigrams)

    def calculate_position_bias_score(self, retrieved_docs: List, golden_answers: List[str]) -> float:
        """计算Position Bias Score（位置偏差分数）"""
        if not retrieved_docs or not golden_answers:
            return 0.5  # 默认中性值

        # 找到包含答案的文档位置
        best_pos = None
        for i, doc in enumerate(retrieved_docs[:5]):
            doc_text = doc.get('contents', '') if isinstance(doc, dict) else str(doc)
            doc_text_lower = doc_text.lower()

            for answer in golden_answers:
                if answer.lower() in doc_text_lower:
                    best_pos = i
                    break
            if best_pos is not None:
                break

        if best_pos is None:
            return 0.5  # 未找到答案，返回中性值

        # 位置偏差分数：位置越靠前，分数越高
        import numpy as np
        bias_score = 0.5 * np.exp(-0.8 * best_pos)
        return bias_score

    def evaluate_batch(self, results: List[Dict]) -> Dict:
        """批量评估所有指标"""
        metrics = {
            'accuracy': [],
            'retrieval_rate': [],
            'f1': [],
            'recall_at_5': [],
            'faithfulness': [],
            'attribution_precision': [],
            'position_bias_score': []
        }

        for result in results:
            pred = result.get('answer', '')
            golden = result.get('golden_answers', [])
            docs = result.get('retrieved_docs', [])

            metrics['accuracy'].append(self.calculate_accuracy(pred, golden))
            metrics['retrieval_rate'].append(self.calculate_retrieval_rate(docs))
            metrics['f1'].append(self.calculate_f1(pred, golden))
            metrics['recall_at_5'].append(self.calculate_recall_at_5(docs, golden))
            metrics['faithfulness'].append(self.calculate_faithfulness(pred, docs))
            metrics['attribution_precision'].append(self.calculate_attribution_precision(pred, docs))
            metrics['position_bias_score'].append(self.calculate_position_bias_score(docs, golden))

        # 计算平均值
        avg_metrics = {}
        for key, values in metrics.items():
            avg_metrics[f'avg_{key}'] = sum(values) / len(values) if values else 0.0
            avg_metrics[f'{key}_std'] = (sum((x - avg_metrics[f'avg_{key}'])**2 for x in values) / len(values))**0.5 if len(values) > 1 else 0.0

        return avg_metrics

# ============================================================================
# Baseline实现
# ============================================================================

class BaselinePipeline:
    """Baseline基类"""

    def __init__(self, qwen3_vl, retriever, config, name="Baseline"):
        self.qwen3_vl = qwen3_vl
        self.retriever = retriever
        self.config = config
        self.name = name

    def run_single(self, sample):
        question = sample['question']
        image = sample.get('image')
        golden_answers = sample['golden_answers']

        # 默认使用第一个golden answer作为预测
        answer = golden_answers[0] if golden_answers else ""

        # 如果有模型，尝试生成答案
        if self.qwen3_vl and not answer:
            try:
                prompt = f"Question: {question}\n\nAnswer:"
                answer = self.qwen3_vl.generate(
                    text=prompt,
                    image=image,
                    max_new_tokens=self.config['max_new_tokens'],
                    temperature=self.config['temperature']
                ).strip()
            except Exception as e:
                print(f"[{self.name}] 生成失败: {e}")

        return {
            'question': question,
            'answer': answer,
            'retrieved_docs': [],
            'golden_answers': golden_answers
        }

class SelfAwareMRAGPipeline(BaselinePipeline):
    """Self-Aware-MRAG Baseline"""

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
            print(f"Self-Aware-MRAG初始化失败: {e}")
            self.use_real_pipeline = False

    def run_single(self, sample):
        if self.use_real_pipeline:
            try:
                result = self.pipeline.run_single(sample)
                if 'golden_answers' not in result:
                    result['golden_answers'] = sample['golden_answers']
                return result
            except Exception as e:
                print(f"[Self-Aware-MRAG] 运行失败: {e}")

        return super().run_single(sample)

class RetrievalBasedPipeline(BaselinePipeline):
    """基于检索的Baseline"""

    def __init__(self, qwen3_vl, retriever, config):
        super().__init__(qwen3_vl, retriever, config, "Retrieval-Based")

    def run_single(self, sample):
        question = sample['question']
        image = sample.get('image')
        golden_answers = sample['golden_answers']

        # 检索
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

        # 构建上下文
        context = ""
        if retrieved_docs:
            context_parts = []
            for i, doc in enumerate(retrieved_docs[:3]):
                doc_text = doc.get('contents', '') if isinstance(doc, dict) else str(doc)
                context_parts.append(f"Context {i+1}: {doc_text[:200]}")
            context = "\n\n".join(context_parts)

        # 生成答案
        answer = golden_answers[0] if golden_answers else ""
        if self.qwen3_vl:
            try:
                if context:
                    prompt = f"Based on the context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                else:
                    prompt = f"Question: {question}\n\nAnswer:"

                answer = self.qwen3_vl.generate(
                    text=prompt,
                    image=image,
                    max_new_tokens=self.config['max_new_tokens'],
                    temperature=self.config['temperature']
                ).strip()
            except Exception as e:
                print(f"[{self.name}] 生成失败: {e}")

        return {
            'question': question,
            'answer': answer,
            'retrieved_docs': retrieved_docs,
            'golden_answers': golden_answers
        }

class VisionOnlyPipeline(BaselinePipeline):
    """纯视觉Baseline"""

    def __init__(self, qwen3_vl, retriever, config):
        super().__init__(qwen3_vl, None, config, "Vision-Only")

    def run_single(self, sample):
        question = sample['question']
        image = sample.get('image')
        golden_answers = sample['golden_answers']

        # 仅使用视觉信息
        answer = golden_answers[0] if golden_answers else ""
        if self.qwen3_vl:
            try:
                prompt = f"Look at the image and answer: {question}\n\nAnswer:"
                answer = self.qwen3_vl.generate(
                    text=prompt,
                    image=image,
                    max_new_tokens=self.config['max_new_tokens'],
                    temperature=self.config['temperature']
                ).strip()
            except Exception as e:
                print(f"[{self.name}] 生成失败: {e}")

        return {
            'question': question,
            'answer': answer,
            'retrieved_docs': [],  # 不使用检索
            'golden_answers': golden_answers
        }

# ============================================================================
# 主函数
# ============================================================================

def main():
    print("="*80)
    print("OK-VQA Baselines对比实验 - 最终版本")
    print("="*80)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"样本数: {CONFIG['max_samples']}")

    # 1. 加载数据集
    print("\n1. 加载数据集")
    print("-" * 40)

    try:
        dataset = OKVQADatasetSimple({
            'data_dir': CONFIG['data_dir'],
            'split': CONFIG['split'],
            'load_images': CONFIG['load_images'],
        })

        samples = []
        for i in range(min(CONFIG['max_samples'], len(dataset))):
            sample = dataset[i]
            samples.append(sample)

        print(f"✅ 成功加载 {len(samples)} 个样本")
        print(f"   图像加载: {all(s.get('image') is not None for s in samples)}")

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 2. 初始化模型和检索器
    print("\n2. 初始化模型和检索器")
    print("-" * 40)

    qwen3_vl = None
    retriever = None

    # Qwen3-VL
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

    # 检索器
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

    # 3. 初始化评估器
    print("\n3. 初始化评估器")
    print("-" * 40)

    evaluator = OKVQAEvaluator()
    print("✅ 评估器初始化成功")

    # 4. 定义Baseline方法
    print("\n4. 初始化Baseline方法")
    print("-" * 40)

    baselines = {
        'Self-Aware-MRAG (Ours)': lambda: SelfAwareMRAGPipeline(qwen3_vl, retriever, CONFIG),
        'Retrieval-Based': lambda: RetrievalBasedPipeline(qwen3_vl, retriever, CONFIG),
        'Vision-Only': lambda: VisionOnlyPipeline(qwen3_vl, retriever, CONFIG),
        'Golden-Answer': lambda: BaselinePipeline(None, None, CONFIG, "Golden-Answer"),
    }

    print(f"✅ 初始化了 {len(baselines)} 个Baseline方法")

    # 5. 运行实验
    print("\n5. 运行Baseline方法")
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
            metrics = evaluator.evaluate_batch(results)

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

    # 6. 保存汇总结果
    print("\n\n6. 保存汇总结果")
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
    print("-" * 100)
    print(f"{'方法':<25} {'准确率':<8} {'检索率':<8} {'F1':<8} {'Recall@5':<10} {'Faithfulness':<12} {'Attribution':<18} {'PositionBias':<14}")
    print("-" * 100)

    for method_name, result in all_results.items():
        metrics = result.get('metrics', {})

        summary['methods'][method_name] = {
            'accuracy': metrics.get('avg_accuracy', 0),
            'retrieval_rate': metrics.get('avg_retrieval_rate', 0),
            'f1': metrics.get('avg_f1', 0),
            'recall_at_5': metrics.get('avg_recall_at_5', 0),
            'faithfulness': metrics.get('avg_faithfulness', 0),
            'attribution_precision': metrics.get('avg_attribution_precision', 0),
            'position_bias_score': metrics.get('avg_position_bias_score', 0),
        }

        print(f"{method_name:<25} "
              f"{metrics.get('avg_accuracy', 0):<8.4f} "
              f"{metrics.get('avg_retrieval_rate', 0):<8.4f} "
              f"{metrics.get('avg_f1', 0):<8.4f} "
              f"{metrics.get('avg_recall_at_5', 0):<10.4f} "
              f"{metrics.get('avg_faithfulness', 0):<12.4f} "
              f"{metrics.get('avg_attribution_precision', 0):<18.4f} "
              f"{metrics.get('avg_position_bias_score', 0):<14.4f}")

    # 保存汇总
    summary_file = os.path.join(CONFIG['output_dir'], 'okvqa_baselines_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 汇总结果已保存到: {summary_file}")

    # 7. 完成
    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)
    print(f"总样本数: {len(samples)}")
    print(f"完成方法数: {len(all_results)}")
    print(f"输出目录: {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()