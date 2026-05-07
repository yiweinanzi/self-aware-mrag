#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MultiModalQA测试 - 使用模拟数据
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
from flashrag.utils.comprehensive_evaluator import evaluate_comprehensive_metrics

# 模拟MultiModalQA数据
MOCK_SAMPLES = [
    {
        'id': 'mmqa_0',
        'question': 'What is shown in the image?',
        'answer': 'cat',
        'golden_answers': ['cat', 'a cat', 'the cat'],
        'question_type': 'ImageQ',
        'image': None,
        'metadata': {
            'table': None,
            'text': None,
            'answer_candidates': ['cat', 'dog', 'bird', 'fish']
        }
    },
    {
        'id': 'mmqa_1',
        'question': 'Which country has the largest population?',
        'answer': 'China',
        'golden_answers': ['China', 'Chinese', 'People\'s Republic of China'],
        'question_type': 'TableQ',
        'image': None,
        'metadata': {
            'table': 'Country | Population | Capital\nChina | 1.4B | Beijing\nIndia | 1.3B | New Delhi\nUSA | 330M | Washington D.C.',
            'text': None,
            'answer_candidates': ['China', 'India', 'USA', 'Indonesia']
        }
    },
    {
        'id': 'mmqa_2',
        'question': 'What is the capital of France according to the text?',
        'answer': 'Paris',
        'golden_answers': ['Paris', 'The capital is Paris'],
        'question_type': 'TextQ',
        'image': None,
        'metadata': {
            'table': None,
            'text': 'France is a country in Western Europe. Its capital city is Paris. France is known for the Eiffel Tower and fine cuisine.',
            'answer_candidates': ['Paris', 'London', 'Berlin', 'Madrid']
        }
    },
    {
        'id': 'mmqa_3',
        'question': 'How many items are in the table?',
        'answer': '3',
        'golden_answers': ['3', 'three', 'Three items'],
        'question_type': 'Compose',
        'image': None,
        'metadata': {
            'table': 'Item | Quantity\nApples | 5\nOranges | 3\nBananas | 7',
            'text': 'The table shows fruit quantities.',
            'answer_candidates': ['3', '4', '5', '6']
        }
    },
    {
        'id': 'mmqa_4',
        'question': 'What animal is depicted?',
        'answer': 'elephant',
        'golden_answers': ['elephant', 'an elephant', 'the elephant'],
        'question_type': 'ImageQ',
        'image': None,
        'metadata': {
            'table': None,
            'text': None,
            'answer_candidates': ['elephant', 'lion', 'tiger', 'bear']
        }
    },
    {
        'id': 'mmqa_5',
        'question': 'Which team won the championship?',
        'answer': 'Team A',
        'golden_answers': ['Team A', 'The winner is Team A'],
        'question_type': 'TableQ',
        'image': None,
        'metadata': {
            'table': 'Team | Score | Position\nTeam A | 85 | 1st\nTeam B | 78 | 2nd\nTeam C | 72 | 3rd',
            'text': None,
            'answer_candidates': ['Team A', 'Team B', 'Team C', 'Team D']
        }
    },
    {
        'id': 'mmqa_6',
        'question': 'What year was the company founded according to the text?',
        'answer': '2010',
        'golden_answers': ['2010', 'in 2010', 'Founded in 2010'],
        'question_type': 'TextQ',
        'image': None,
        'metadata': {
            'table': None,
            'text': 'The company was founded in 2010 and has grown to over 1000 employees. It specializes in software development.',
            'answer_candidates': ['2010', '2005', '2015', '2020']
        }
    },
    {
        'id': 'mmqa_7',
        'question': 'What is the total value shown in the table and text?',
        'answer': '150',
        'golden_answers': ['150', '$150', '150 dollars'],
        'question_type': 'Compose',
        'image': None,
        'metadata': {
            'table': 'Item | Value ($)\nProduct A | 100\nProduct B | 50',
            'text': 'The total value of all products is $150.',
            'answer_candidates': ['150', '200', '250', '300']
        }
    },
    {
        'id': 'mmqa_8',
        'question': 'Which fruit is mentioned in the text?',
        'answer': 'apple',
        'golden_answers': ['apple', 'apples', 'the apple'],
        'question_type': 'TextQ',
        'image': None,
        'metadata': {
            'table': None,
            'text': 'The apple is a popular fruit. It comes in different colors like red, green, and yellow. Apples are rich in fiber and vitamins.',
            'answer_candidates': ['apple', 'banana', 'orange', 'grape']
        }
    },
    {
        'id': 'mmqa_9',
        'question': 'What color is the car in the image?',
        'answer': 'red',
        'golden_answers': ['red', 'a red car', 'the car is red'],
        'question_type': 'ImageQ',
        'image': None,
        'metadata': {
            'table': None,
            'text': None,
            'answer_candidates': ['red', 'blue', 'green', 'yellow']
        }
    }
]

# ============================================================================
# Pipeline类
# ============================================================================

class MultiModalQADirectPipeline:
    """MultiModalQA直接回答Pipeline"""

    def __init__(self, qwen3_vl, config):
        self.qwen3_vl = qwen3_vl
        self.config = config

    def run_single(self, sample):
        """运行单个样本"""
        question = sample['question']
        question_type = sample['question_type']
        image = sample.get('image')
        metadata = sample.get('metadata', {})

        # 构建上下文
        context_parts = []

        if metadata.get('table'):
            context_parts.append(f"Table:\n{metadata['table']}")

        if metadata.get('text'):
            context_parts.append(f"Text:\n{metadata['text']}")

        context = "\n\n".join(context_parts) if context_parts else ""

        # 构建prompt
        if context:
            prompt = f"{context}\n\nQuestion: {question}\n\nAnswer:"
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
            'retrieved_docs': [],
            'question_type': question_type,
            'metadata': metadata
        }

# ============================================================================
# 测试函数
# ============================================================================

def test_multimodalqa():
    """测试MultiModalQA"""
    print("="*80)
    print("MultiModalQA测试 - 10个模拟样本")
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

    # 2. 定义方法
    config = {'retrieval_topk': 5}
    methods = {
        'Direct Answer': MultiModalQADirectPipeline(qwen3_vl, config),
    }

    # 3. 运行测试
    print("\n2. 运行方法测试...")
    all_results = {}
    type_counts = {}

    # 统计问题类型
    for sample in MOCK_SAMPLES:
        qtype = sample['question_type']
        type_counts[qtype] = type_counts.get(qtype, 0) + 1

    print(f"问题类型分布: {type_counts}")

    for method_name, pipeline in methods.items():
        print(f"\n{'='*40}")
        print(f"测试方法: {method_name}")
        print(f"{'='*40}")

        start_time = time.time()
        results = []
        type_results = {qtype: [] for qtype in type_counts.keys()}

        for i, sample in enumerate(MOCK_SAMPLES):
            print(f"\r进度: {i+1}/{len(MOCK_SAMPLES)}", end='', flush=True)
            try:
                result = pipeline.run_single(sample)
                results.append(result)

                # 按类型分组
                qtype = sample['question_type']
                type_results[qtype].append(result)

                # 打印第一个样本的详细信息
                if i == 0:
                    print(f"\n第一个样本:")
                    print(f"  问题: {sample['question']}")
                    print(f"  问题类型: {sample['question_type']}")
                    print(f"  生成答案: {result['answer'][:100]}...")

            except Exception as e:
                print(f"\n样本 {i} 处理失败: {e}")
                results.append({'answer': '', 'retrieved_docs': [], 'question_type': sample.get('question_type', 'Unknown')})

        elapsed_time = time.time() - start_time
        print(f"\n完成! 耗时: {elapsed_time:.2f}s")
        all_results[method_name] = results

        # 按类型显示结果
        print(f"\n按问题类型统计:")
        for qtype, type_res in type_results.items():
            print(f"  {qtype}: {len(type_res)} 样本")

    # 4. 评估结果
    print("\n3. 评估结果...")
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

            # 按问题类型计算准确率
            print(f"\n按问题类型准确率:")
            for qtype in type_counts.keys():
                type_correct = 0
                type_total = 0

                for i, sample in enumerate(MOCK_SAMPLES):
                    if sample['question_type'] == qtype:
                        type_total += 1
                        if i < len(results):
                            answer = results[i].get('answer', '').lower()
                            # 简单匹配
                            for golden in sample['golden_answers']:
                                if golden.lower() in answer or answer in golden.lower():
                                    type_correct += 1
                                    break

                type_acc = type_correct / type_total * 100 if type_total > 0 else 0
                print(f"  {qtype}: {type_acc:.1f}% ({type_correct}/{type_total})")

        except Exception as e:
            print(f"  评估失败: {e}")

    # 5. 保存结果
    print("\n4. 保存结果...")
    output_dir = Path('/data0/home/zqwang/ACL/FlashRAG/test_results_multimodalqa_mock')
    output_dir.mkdir(exist_ok=True)

    # 保存详细结果
    results_file = output_dir / 'results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'samples': MOCK_SAMPLES,
            'results': all_results,
            'type_counts': type_counts,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)

    print(f"✅ 结果已保存到: {results_file}")

    print("\n" + "="*80)
    print("测试完成!")
    print("="*80)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    test_multimodalqa()