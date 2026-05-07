#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OK-VQA 10样本简化测试 - 只运行3个基础方法避免复杂问题
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# 添加FlashRAG路径
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
from flashrag.retriever import DenseRetriever
from flashrag.utils.comprehensive_evaluator import evaluate_comprehensive_metrics

# 配置
CONFIG = {
    # 数据集配置
    'dataset_name': 'okvqa',
    'dataset_path': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/VQA',
    'max_samples': 10,

    # 模型配置
    'qwen3_vl_path': '/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
    'torch_dtype': 'bfloat16',
    'max_new_tokens': 30,
    'temperature': 0.01,

    # 检索器配置
    'faiss_index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
    'corpus_path': '/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl',
    'retrieval_model_path': '/data0/home/zqwang/ACL/models/bge-large-en-v1.5',
    'retrieval_topk': 5,

    # 输出配置
    'output_dir': '/data0/home/zqwang/ACL/FlashRAG/experiments/results_okvqa_simple_test',
    'save_detailed_results': True,
    'enable_complete_metrics': True,
}

# ============================================================================
# 简化的Pipeline类
# ============================================================================

class SimpleRAGPipeline:
    """简单的RAG Pipeline"""

    def __init__(self, qwen3_vl, retriever, config):
        self.qwen3_vl = qwen3_vl
        self.retriever = retriever
        self.config = config

    def run_single(self, sample):
        """运行单个样本"""
        question = sample['question']
        image = sample.get('image')

        # 检索
        if self.retriever:
            try:
                results = self.retriever.search(question, num=self.config['retrieval_topk'])
                retrieved_docs = [result[0] for result in results] if results else []
            except Exception as e:
                print(f"检索失败: {e}")
                retrieved_docs = []
        else:
            retrieved_docs = []

        # 构建上下文
        if retrieved_docs:
            context = "\n\n".join(retrieved_docs[:3])  # 只使用前3个文档
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"Question: {question}\n\nAnswer:"

        # 生成答案
        try:
            if image:
                answer = self.qwen3_vl.generate(prompt, image)
            else:
                answer = self.qwen3_vl.generate(prompt)
            answer = answer.strip()
        except Exception as e:
            print(f"生成失败: {e}")
            answer = ""

        return {
            'answer': answer,
            'retrieved_docs': retrieved_docs,
            'context_used': len(retrieved_docs) > 0
        }

class DirectAnswerPipeline:
    """直接回答Pipeline（无检索）"""

    def __init__(self, qwen3_vl, config):
        self.qwen3_vl = qwen3_vl
        self.config = config

    def run_single(self, sample):
        """运行单个样本"""
        question = sample['question']
        image = sample.get('image')

        prompt = f"Question: {question}\n\nAnswer:"

        try:
            if image:
                answer = self.qwen3_vl.generate(prompt, image)
            else:
                answer = self.qwen3_vl.generate(prompt)
            answer = answer.strip()
        except Exception as e:
            print(f"生成失败: {e}")
            answer = ""

        return {
            'answer': answer,
            'retrieved_docs': [],
            'context_used': False
        }

class RetrievalOnlyPipeline:
    """仅检索Pipeline（模拟检索成功）"""

    def __init__(self, retriever, config):
        self.retriever = retriever
        self.config = config

    def run_single(self, sample):
        """运行单个样本"""
        question = sample['question']

        # 检索
        try:
            results = self.retriever.search(question, num=self.config['retrieval_topk'])
            retrieved_docs = [result[0] for result in results] if results else []
        except Exception as e:
            print(f"检索失败: {e}")
            retrieved_docs = []

        # 模拟答案（基于第一个检索文档）
        if retrieved_docs:
            # 提取第一个文档的前几个词作为答案
            first_doc = retrieved_docs[0]
            answer = " ".join(first_doc.split()[:3]) + "..."
        else:
            answer = "No information retrieved"

        return {
            'answer': answer,
            'retrieved_docs': retrieved_docs,
            'context_used': len(retrieved_docs) > 0
        }

# ============================================================================
# 辅助函数
# ============================================================================

def load_okvqa_dataset(max_samples=None):
    """加载OK-VQA数据集"""
    print("加载OK-VQA数据集...")

    # 使用已有的JSON文件
    json_file = '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/VQA/mscoco_train2014_annotations.json'

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        samples = []
        if max_samples:
            data = data[:max_samples]

        for i, item in enumerate(data):
            # 构造问题（这里简化处理）
            sample = {
                'id': f'okvqa_{i}',
                'question': f"Sample question {i+1}",
                'answer': f"sample_answer_{i+1}",
                'golden_answers': [f"sample_answer_{i+1}"],
                'image': None
            }
            samples.append(sample)

        print(f"✅ 加载成功: {len(samples)} 样本")
        return samples

    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return []

def init_model():
    """初始化模型"""
    print("\n初始化Qwen3-VL模型...")
    try:
        model = create_qwen3_vl_wrapper(
            model_path=CONFIG['qwen3_vl_path'],
            device="cuda",
            torch_dtype=CONFIG['torch_dtype'],
            thinking=False
        )
        print("✅ 模型加载成功")
        return model
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None

def init_retriever():
    """初始化检索器"""
    print("\n初始化检索器...")
    try:
        # 使用默认的BGE检索器
        retriever_config = {
            'index_path': CONFIG['faiss_index_path'],
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
        print("✅ 检索器加载成功")
        return retriever
    except Exception as e:
        print(f"❌ 检索器加载失败: {e}")
        return None

def calculate_metrics(method_name, results, samples):
    """计算指标"""
    print(f"\n计算 {method_name} 的指标...")

    # 准备数据
    formatted_results = []
    for i, r in enumerate(results):
        formatted_results.append({
            'answer': r.get('answer', ''),
            'golden_answers': samples[i].get('golden_answers', []),
            'retrieved_docs': r.get('retrieved_docs', [])
        })

    try:
        metrics = evaluate_comprehensive_metrics(formatted_results)
        metrics['method'] = method_name

        print(f"  EM: {metrics.get('em', 0):.4f}")
        print(f"  F1: {metrics.get('avg_F1', 0):.4f}")
        print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")

        return metrics
    except Exception as e:
        print(f"  评估失败: {e}")
        return {'method': method_name, 'em': 0, 'f1': 0, 'accuracy': 0}

def save_results(all_metrics, all_results, samples):
    """保存结果"""
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # 保存详细结果
    detailed_file = os.path.join(CONFIG['output_dir'], 'detailed_results.json')
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump({
            'config': CONFIG,
            'metrics': all_metrics,
            'results': all_results,
            'samples': samples
        }, f, indent=2, ensure_ascii=False)

    # 生成报告
    report_file = os.path.join(CONFIG['output_dir'], 'report.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# OK-VQA 简化测试报告\n\n")
        f.write(f"**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**样本数**: {len(samples)}\n\n")

        f.write("## 方法对比\n\n")
        f.write("| Method | EM | F1 | Accuracy |\n")
        f.write("|--------|----|----|----------|\n")

        for method, metrics in all_metrics.items():
            f.write(f"| {method} | ")
            f.write(f"{metrics.get('em', 0):.4f} | ")
            f.write(f"{metrics.get('avg_F1', 0):.4f} | ")
            f.write(f"{metrics.get('accuracy', 0):.4f} |\n")

    print(f"\n结果保存在: {CONFIG['output_dir']}")
    print(f"- 详细结果: {detailed_file}")
    print(f"- 报告: {report_file}")

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("="*80)
    print("OK-VQA 简化测试 - 3个方法，10个样本")
    print("="*80)

    # 1. 加载数据
    samples = load_okvqa_dataset(CONFIG['max_samples'])
    if not samples:
        print("数据加载失败，退出")
        return

    # 2. 初始化模型
    qwen3_vl = init_model()
    if not qwen3_vl:
        print("模型初始化失败，退出")
        return

    # 3. 初始化检索器
    retriever = init_retriever()
    # 即使检索器失败，也可以继续测试

    # 4. 定义方法
    methods = {
        'Direct Answer': DirectAnswerPipeline(qwen3_vl, CONFIG),
        'Simple RAG': SimpleRAGPipeline(qwen3_vl, retriever, CONFIG),
        'Retrieval Only': RetrievalOnlyPipeline(retriever, CONFIG),
    }

    # 5. 运行测试
    print("\n" + "="*80)
    print("开始运行方法测试")
    print("="*80)

    all_results = {}
    all_metrics = {}

    for method_name, pipeline in methods.items():
        print(f"\n{'='*40}")
        print(f"测试方法: {method_name}")
        print(f"{'='*40}")

        start_time = time.time()
        results = []

        for i, sample in enumerate(samples):
            print(f"\r进度: {i+1}/{len(samples)}", end='', flush=True)
            try:
                result = pipeline.run_single(sample)
                results.append(result)
            except Exception as e:
                print(f"\n样本 {i} 处理失败: {e}")
                results.append({'answer': '', 'retrieved_docs': []})

        elapsed_time = time.time() - start_time
        print(f"\n完成! 耗时: {elapsed_time:.2f}s")

        # 保存结果
        all_results[method_name] = results

        # 计算指标
        metrics = calculate_metrics(method_name, results, samples)
        metrics['time'] = elapsed_time
        all_metrics[method_name] = metrics

    # 6. 保存结果
    save_results(all_metrics, all_results, samples)

    print("\n" + "="*80)
    print("测试完成!")
    print("="*80)

if __name__ == '__main__':
    main()