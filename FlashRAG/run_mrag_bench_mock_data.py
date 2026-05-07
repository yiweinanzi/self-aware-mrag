#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MRAG-Bench测试 - 使用模拟数据
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

# 模拟MRAG-Bench数据
MOCK_SAMPLES = [
    {
        'id': 'mrag_0',
        'question': 'What animal is shown in the image?',
        'answer': 'D',
        'answer_text': 'dog',
        'choices': ['cat', 'bird', 'fish', 'dog'],
        'A': 'cat',
        'B': 'bird',
        'C': 'fish',
        'D': 'dog',
        'correct_choice_idx': 3,
        'scenario': 'Basic',
        'golden_answers': ['D']
    },
    {
        'id': 'mrag_1',
        'question': 'Which object is larger?',
        'answer': 'A',
        'answer_text': 'elephant',
        'choices': ['elephant', 'mouse', 'cat', 'dog'],
        'A': 'elephant',
        'B': 'mouse',
        'C': 'cat',
        'D': 'dog',
        'correct_choice_idx': 0,
        'scenario': 'Comparison',
        'golden_answers': ['A']
    },
    {
        'id': 'mrag_2',
        'question': 'What color is the apple?',
        'answer': 'B',
        'answer_text': 'red',
        'choices': ['yellow', 'red', 'green', 'blue'],
        'A': 'yellow',
        'B': 'red',
        'C': 'green',
        'D': 'blue',
        'correct_choice_idx': 1,
        'scenario': 'Color',
        'golden_answers': ['B']
    },
    {
        'id': 'mrag_3',
        'question': 'How many objects are there?',
        'answer': 'C',
        'answer_text': 'three',
        'choices': ['one', 'two', 'three', 'four'],
        'A': 'one',
        'B': 'two',
        'C': 'three',
        'D': 'four',
        'correct_choice_idx': 2,
        'scenario': 'Counting',
        'golden_answers': ['C']
    },
    {
        'id': 'mrag_4',
        'question': 'Where is the object located?',
        'answer': 'B',
        'answer_text': 'indoors',
        'choices': ['outdoors', 'indoors', 'left', 'right'],
        'A': 'outdoors',
        'B': 'indoors',
        'C': 'left',
        'D': 'right',
        'correct_choice_idx': 1,
        'scenario': 'Location',
        'golden_answers': ['B']
    },
    {
        'id': 'mrag_5',
        'question': 'What time of day is shown?',
        'answer': 'A',
        'answer_text': 'morning',
        'choices': ['morning', 'afternoon', 'evening', 'night'],
        'A': 'morning',
        'point_mrag': 'afternoon',
        'C': 'evening',
        'D': 'night',
        'correct_choice_idx': 0,
        'scenario': 'Time',
        'golden_answers': ['A']
    },
    {
        'id': 'mrag_6',
        'question': 'What season is depicted?',
        'answer': 'C',
        'answer_text': 'winter',
        'choices': ['spring', 'summer', 'winter', 'fall'],
        'A': 'spring',
        'B': 'summer',
        'C': 'winter',
        'D': 'fall',
        'correct_choice_idx': 2,
        'scenario': 'Season',
        'golden_answers': ['C']
    },
    {
        'id': 'mrag_7',
        'question': 'Which direction is the object facing?',
        'answer': 'D',
        'answer_text': 'right',
        'choices': ['left', 'right', 'up', 'right'],
        'A': 'left',
        'B': 'right',
        'C': 'up',
        'D': 'right',
        'correct_choice_idx': 1,
        'scenario': 'Direction',
        'golden_answers': ['D']
    },
    {
        'id': 'mrag_8',
        'question': 'What is the material of the object?',
        'answer': 'B',
        'answer_text': 'wood',
        'choices': ['metal', 'wood', 'plastic', 'glass'],
        'A': 'metal',
        'B': 'wood',
        'C': 'plastic',
        'D': 'glass',
        'correct_choice_idx': 1,
        'scenario': 'Material',
        'golden_answers': ['B']
    },
    {
        'id': 'mrag_9',
        'question': 'What action is being performed?',
        'answer': 'C',
        'answer_text': 'jumping',
        'choices': ['running', 'walking', 'jumping', 'sitting'],
        'A': 'running',
        'B': 'walking',
        'C': 'jumping',
        'D': 'sitting',
        'correct_choice_idx': 2,
        'scenario': 'Action',
        'golden_answers': ['C']
    }
]

# ============================================================================
# Pipeline类
# ============================================================================

class MRAGDirectPipeline:
    """MRAG直接回答Pipeline"""

    def __init__(self, qwen3_vl, config):
        self.qwen3_vl = qwen3_vl
        self.config = config

    def run_single(self, sample):
        """运行单个样本"""
        question = sample['question']
        choices = sample['choices']
        image = sample.get('image')

        # 构造多选题prompt
        prompt = "Question: {}\n\nChoices:\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "\nAnswer with the letter only (A, B, C, or D):"

        # 生成答案
        try:
            if image:
                answer = self.qwen3_vl.generate(prompt, image)
            else:
                answer = self.qwen3_vl.generate(prompt)
            answer = answer.strip()

            # 提取答案字母
            answer = answer[0].upper() if answer else 'A'
            if answer not in ['A', 'B', 'C', 'D']:
                answer = 'A'
        except Exception as e:
            print(f"生成失败: {e}")
            answer = 'A'

        return {
            'answer': answer,
            'retrieved_docs': [],
            'choices': choices,
            'correct_choice_idx': sample['correct_choice_idx']
        }

def extract_mc_answer(prediction):
    """提取多选题答案"""
    prediction = prediction.strip().upper()

    # 直接匹配选项
    for choice in ['A', 'B', 'C', 'D']:
        if choice in prediction:
            return choice

    # 提取模式
    patterns = [
        r'ANSWER IS ([ABCD])',
        r'CHOICE IS ([ABCD])',
        r'CORRECT: ([ABCD])',
        r'^([ABCD])$'
    ]

    import re
    for pattern in patterns:
        match = re.search(pattern, prediction)
        if match:
            return match.group(1)

    # 默认返回A
    return 'A'

# ============================================================================
# 测试函数
# ============================================================================

def test_mrag_bench():
    """测试MRAG-Bench"""
    print("="*80)
    print("MRAG-Bench测试 - 10个模拟样本")
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
        'Direct Answer': MRAGDirectPipeline(qwen3_vl, config),
    }

    # 3. 运行测试
    print("\n2. 运行方法测试...")
    all_results = {}

    for method_name, pipeline in methods.items():
        print(f"\n{'='*40}")
        print(f"测试方法: {method_name}")
        print(f"{'='*40}")

        start_time = time.time()
        results = []
        correct = 0

        for i, sample in enumerate(MOCK_SAMPLES):
            print(f"\r进度: {i+1}/{len(MOCK_SAMPLES)}", end='', flush=True)
            try:
                result = pipeline.run_single(sample)
                results.append(result)

                # 检查答案
                if result['answer'] == chr(65 + sample['correct_choice_idx']):
                    correct += 1

                # 打印第一个样本的详细信息
                if i == 0:
                    print(f"\n第一个样本:")
                    print(f"  问题: {sample['question']}")
                    print(f"  选项: {sample['choices']}")
                    print(f"  正确答案: {chr(65 + sample['correct_choice_idx'])}")
                    print(f"  生成答案: {result['answer']}")

            except Exception as e:
                print(f"\n样本 {i} 处理失败: {e}")
                results.append({'answer': 'A', 'retrieved_docs': []})

        elapsed_time = time.time() - start_time
        accuracy = correct / len(results) * 100
        print(f"\n完成! 耗时: {elapsed_time:.2f}s")
        print(f"准确率: {accuracy:.1f}%")
        all_results[method_name] = results

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
                'golden_answers': [chr(65 + MOCK_SAMPLES[i]['correct_choice_idx'])],
                'retrieved_docs': r.get('retrieved_docs', [])
            })

        try:
            metrics = evaluate_comprehensive_metrics(formatted_results)
            print(f"  EM: {metrics.get('em', 0):.4f}")
            print(f"  F1: {metrics.get('avg_F1', 0):.4f}")
            print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")

            # MRAG特有的准确率
            correct = sum(1 for r in results
                            if r.get('answer', 'A') == chr(65 + MOCK_SAMPLES[results.index(r)]['correct_choice_idx']))
            actual_accuracy = correct / len(results) * 100 if results else 0
            print(f"  实际准确率: {actual_accuracy:.1f}%")

        except Exception as e:
            print(f"  评估失败: {e}")

    # 5. 保存结果
    print("\n4. 保存结果...")
    output_dir = Path('/data0/home/zqwang/ACL/FlashRAG/test_results_mrag_mock')
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
    test_mrag_bench()