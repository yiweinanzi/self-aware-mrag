#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OK-VQA测试 - 使用模拟数据
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

# 模拟数据
MOCK_SAMPLES = [
    {
        'id': 'okvqa_0',
        'question': 'What color is the sky on a clear day?',
        'answer': 'blue',
        'golden_answers': ['blue', 'blue sky', 'light blue'],
        'image': None
    },
    {
        'id': 'okvqa_1',
        'question': 'What do you use to write on paper?',
        'answer': 'pen',
        'golden_answers': ['pen', 'pencil', 'ballpoint pen'],
        'image': None
    },
    {
        'id': 'okvqa_2',
        'question': 'What animal says "woof"?',
        'answer': 'dog',
        'golden_answers': ['dog', 'dogs', 'puppy'],
        'image': None
    },
    {
        'id': 'okvqa_3',
        'question': 'What is the largest planet in our solar system?',
        'answer': 'jupiter',
        'golden_answers': ['jupiter', 'Jupiter'],
        'image': None
    },
    {
        'id': 'okvqa_4',
        'question': 'How many days are in a week?',
        'answer': 'seven',
        'golden_answers': ['seven', '7', 'seven days'],
        'image': None
    },
    {
        'id': 'okvqa_5',
        'question': 'What do you drink when you are thirsty?',
        'answer': 'water',
        'golden_answers': ['water', 'drink', 'beverage'],
        'image': None
    },
    {
        'id': 'okvqa_6',
        'question': 'What time of day is the sun at its highest?',
        'answer': 'noon',
        'golden_answers': ['noon', 'midday', '12pm'],
        'image': None
    },
    {
        'id': 'okvqa_7',
        'question': 'What season comes after summer?',
        'answer': 'autumn',
        'golden_answers': ['autumn', 'fall'],
        'image': None
    },
    {
        'id': 'okvqa_8',
        'question': 'What do birds use to fly?',
        'answer': 'wings',
        'golden_answers': ['wings', 'feathers'],
        'image': None
    },
    {
        'id': 'okvqa_9',
        'question': 'What color are leaves in autumn?',
        'answer': 'orange',
        'golden_answers': ['orange', 'brown', 'yellow', 'red'],
        'image': None
    }
]

# ============================================================================
# Pipeline类
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
            context = "\n\n".join(retrieved_docs[:3])
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

# ============================================================================
# 测试函数
# ============================================================================

def test_okvqa():
    """测试OK-VQA"""
    print("="*80)
    print("OK-VQA测试 - 10个模拟样本")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 初始化模型
    print("\n1. 初始化Qwen3-VL...")
    try:
        qwen3_vl = create_qwen3_vl_wrapper(
            model_path='/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
            device="cuda",
            torch_dtype="bfloat16"
        )
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 2. 初始化检索器（可选）
    print("\n2. 初始化检索器...")
    retriever = None
    try:
        retriever_config = {
            'index_path': '/data0/home/zqwang/ACL/FlashRAG/indexes/wiki_3m/bge/e5_Flat.index',
            'corpus_path': '/data0/home/zqwang/ACL/FlashRAG/corpus/corpus_wiki_3m.jsonl',
            'retrieval_method': 'e5',
            'retrieval_model_path': '/data0/home/zqwang/ACL/models/bge-large-en-v1.5',
            'retrieval_query_max_length': 512,
            'retrieval_pooling_method': 'mean',
            'retrieval_use_fp16': True,
            'retrieval_batch_size': 128,
            'retrieval_topk': 5,
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
        print("继续使用无检索模式测试...")

    # 3. 定义方法
    config = {'retrieval_topk': 5}
    methods = {
        'Direct Answer': DirectAnswerPipeline(qwen3_vl, config),
        'Simple RAG': SimpleRAGPipeline(qwen3_vl, retriever, config),
    }

    # 4. 运行测试
    print("\n3. 运行方法测试...")
    all_results = {}

    for method_name, pipeline in methods.items():
        print(f"\n{'='*40}")
        print(f"测试方法: {method_name}")
        print(f"{'='*40}")

        start_time = time.time()
        results = []

        for i, sample in enumerate(MOCK_SAMPLES):
            print(f"\r进度: {i+1}/{len(MOCK_SAMPLES)}", end='', flush=True)
            try:
                result = pipeline.run_single(sample)
                results.append(result)

                # 打印第一个样本的详细信息
                if i == 0:
                    print(f"\n第一个样本:")
                    print(f"  问题: {sample['question']}")
                    print(f"  生成答案: {result['answer']}")
                    print(f"  检索文档数: {len(result.get('retrieved_docs', []))}")

            except Exception as e:
                print(f"\n样本 {i} 处理失败: {e}")
                results.append({'answer': '', 'retrieved_docs': []})

        elapsed_time = time.time() - start_time
        print(f"\n完成! 耗时: {elapsed_time:.2f}s")
        all_results[method_name] = results

    # 5. 评估结果
    print("\n4. 评估结果...")
    print("-" * 80)

    for method_name, results in all_results.items():
        print(f"\n方法: {method_name}")

        # 准备评估数据
        formatted_results = []
        for i, r in enumerate(results):
            formatted_results.append({
                'answer': r.get('answer', ''),
                'golden_answers': MOCK_SAMPLES[i]['golden_answers'],
                'retrieved_docs': r.get('retrieved_docs', [])
            })

        try:
            metrics = evaluate_comprehensive_metrics(formatted_results)
            print(f"  EM: {metrics.get('em', 0):.4f}")
            print(f"  F1: {metrics.get('avg_F1', 0):.4f}")
            print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
            print(f"  Retrieval Rate: {metrics.get('retrieval_rate', 0):.4f}")

        except Exception as e:
            print(f"  评估失败: {e}")

    # 6. 保存结果
    print("\n5. 保存结果...")
    output_dir = Path('/data0/home/zqwang/ACL/FlashRAG/test_results_okvqa_mock')
    output_dir.mkdir(exist_ok=True)

    # 保存详细结果
    results_file = output_dir / 'results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'samples': MOCK_SAMPLES,
            'results': all_results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)

    print(f"✅ 结果已保存到: {results_file}")

    print("\n" + "="*80)
    print("测试完成!")
    print("="*80)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    test_okvqa()