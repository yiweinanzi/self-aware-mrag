#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A-OKVQA测试 - 使用模拟数据
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

# 模拟A-OKVQA数据
MOCK_SAMPLES = [
    {
        'id': 'aokvqa_0',
        'question': 'What color is the apple in the image?',
        'choices': ['red', 'green', 'yellow', 'blue'],
        'answer': 'red',
        'rationale': 'The apple shown in the image is clearly red, which is a common color for apples.',
        'golden_answers': ['red'],
        'image': None
    },
    {
        'id': 'aokvqa_1',
        'question': 'Which animal is larger?',
        'choices': ['elephant', 'mouse', 'cat', 'dog'],
        'answer': 'elephant',
        'rationale': 'Elephants are the largest land animals, much bigger than mice, cats, or dogs.',
        'golden_answers': ['elephant'],
        'image': None
    },
    {
        'id': 'aokvqa_2',
        'question': 'How many wheels does a car have?',
        'choices': ['2', '3', '4', '6'],
        'answer': '4',
        'rationale': 'A standard car has four wheels - two in the front and two in the back.',
        'golden_answers': ['4', 'four'],
        'image': None
    },
    {
        'id': 'aokvqa_3',
        'question': 'What season is shown when leaves are falling?',
        'choices': ['spring', 'summer', 'autumn', 'winter'],
        'answer': 'autumn',
        'rationale': 'Autumn (fall) is the season when trees shed their leaves.',
        'golden_answers': ['autumn', 'fall'],
        'image': None
    },
    {
        'id': 'aokvqa_4',
        'question': 'Which object can fly?',
        'choices': ['rock', 'bird', 'car', 'boat'],
        'answer': 'bird',
        'rationale': 'Birds have wings and can fly, while rocks, cars, and boats cannot.',
        'golden_answers': ['bird'],
        'image': None
    },
    {
        'id': 'aokvqa_5',
        'question': 'What time of day is the sun at its highest?',
        'choices': ['morning', 'noon', 'afternoon', 'night'],
        'answer': 'noon',
        'rationale': 'The sun reaches its highest point in the sky at noon (midday).',
        'golden_answers': ['noon', 'midday'],
        'image': None
    },
    {
        'id': 'aokvqa_6',
        'question': 'Which of these is a fruit?',
        'choices': ['carrot', 'potato', 'apple', 'lettuce'],
        'answer': 'apple',
        'rationale': 'Apples are fruits, while carrots, potatoes, and lettuce are vegetables.',
        'golden_answers': ['apple'],
        'image': None
    },
    {
        'id': 'aokvqa_7',
        'question': 'How many days are in a week?',
        'choices': ['5', '6', '7', '8'],
        'answer': '7',
        'rationale': 'A week has seven days: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, and Sunday.',
        'golden_answers': ['7', 'seven'],
        'image': None
    },
    {
        'id': 'aokvqa_8',
        'question': 'What color do you get when mixing blue and yellow?',
        'choices': ['red', 'green', 'purple', 'orange'],
        'answer': 'green',
        'rationale': 'Mixing blue and yellow paint or light produces green.',
        'golden_answers': ['green'],
        'image': None
    },
    {
        'id': 'aokvqa_9',
        'question': 'Which direction does the sun rise?',
        'choices': ['north', 'south', 'east', 'west'],
        'answer': 'east',
        'rationale': 'The sun rises in the east and sets in the west due to Earth\'s rotation.',
        'golden_answers': ['east'],
        'image': None
    }
]

# ============================================================================
# Pipeline类
# ============================================================================

class AOKVQADirectPipeline:
    """A-OKVQA直接回答Pipeline"""

    def __init__(self, qwen3_vl, config):
        self.qwen3_vl = qwen3_vl
        self.config = config

    def run_single(self, sample):
        """运行单个样本"""
        question = sample['question']
        choices = sample['choices']
        image = sample.get('image')

        # 构造多选题prompt
        prompt = "Question: {}\n\nChoices:\n".format(question)
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "\nFirst, think step by step and provide your reasoning. Then, give the final answer by selecting the letter of the correct choice (A, B, C, or D)."

        # 生成答案
        try:
            if image:
                answer = self.qwen3_vl.generate(prompt, image)
            else:
                answer = self.qwen3_vl.generate(prompt)
            answer = answer.strip()
        except Exception as e:
            print(f"生成失败: {e}")
            answer = "A"

        # 提取答案字母
        answer_letter = self.extract_choice_letter(answer)

        return {
            'answer': answer_letter,
            'full_answer': answer,
            'retrieved_docs': [],
            'choices': choices,
            'correct_choice': chr(65 + choices.index(sample['answer']))
        }

    def extract_choice_letter(self, text):
        """提取答案字母"""
        import re

        # 寻找模式
        patterns = [
            r'(?:final answer|answer|choice|correct) is? ([ABCD])',
            r'([ABCD])\s*[.:\)]',
            r'choice ([ABCD])',
            r'^([ABCD])$',
            r'答案是 ([ABCD])'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        # 如果找不到，默认返回A
        return 'A'

class AOKVQAReasoningPipeline:
    """A-OKVQA推理Pipeline"""

    def __init__(self, qwen3_vl, config):
        self.qwen3_vl = qwen3_vl
        self.config = config

    def run_single(self, sample):
        """运行单个样本"""
        question = sample['question']
        choices = sample['choices']
        rationale = sample.get('rationale', '')
        image = sample.get('image')

        # 使用提供的rationale作为上下文
        if rationale:
            prompt = f"Context: {rationale}\n\n"
        else:
            prompt = ""

        prompt += "Question: {}\n\nChoices:\n".format(question)
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "\nBased on the context above, what is the correct answer? Provide the letter of your choice (A, B, C, or D)."

        # 生成答案
        try:
            if image:
                answer = self.qwen3_vl.generate(prompt, image)
            else:
                answer = self.qwen3_vl.generate(prompt)
            answer = answer.strip()
        except Exception as e:
            print(f"生成失败: {e}")
            answer = "A"

        # 提取答案字母
        answer_letter = self.extract_choice_letter(answer)

        return {
            'answer': answer_letter,
            'full_answer': answer,
            'retrieved_docs': [rationale] if rationale else [],
            'choices': choices,
            'correct_choice': chr(65 + choices.index(sample['answer'])),
            'used_rationale': True
        }

    def extract_choice_letter(self, text):
        """提取答案字母"""
        import re

        # 寻找模式
        patterns = [
            r'(?:final answer|answer|choice|correct) is? ([ABCD])',
            r'([ABCD])\s*[.:\)]',
            r'choice ([ABCD])',
            r'^([ABCD])$',
            r'答案是 ([ABCD])'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        # 如果找不到，默认返回A
        return 'A'

# ============================================================================
# 测试函数
# ============================================================================

def test_aokvqa():
    """测试A-OKVQA"""
    print("="*80)
    print("A-OKVQA测试 - 10个模拟样本")
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
        'Direct Answer': AOKVQADirectPipeline(qwen3_vl, config),
        'With Rationale': AOKVQAReasoningPipeline(qwen3_vl, config),
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
                if result['answer'] == result['correct_choice']:
                    correct += 1

                # 打印第一个样本的详细信息
                if i == 0:
                    print(f"\n第一个样本:")
                    print(f"  问题: {sample['question']}")
                    print(f"  选项: {sample['choices']}")
                    print(f"  正确答案: {result['correct_choice']}")
                    print(f"  生成答案: {result['answer']}")
                    if method_name == 'With Rationale':
                        print(f"  推理依据: {sample.get('rationale', '')[:50]}...")

            except Exception as e:
                print(f"\n样本 {i} 处理失败: {e}")
                results.append({'answer': 'A', 'retrieved_docs': [], 'choices': sample['choices']})

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
                'golden_answers': [MOCK_SAMPLES[i]['answer']],
                'retrieved_docs': r.get('retrieved_docs', [])
            })

        try:
            metrics = evaluate_comprehensive_metrics(formatted_results)
            print(f"  EM: {metrics.get('em', 0):.4f}")
            print(f"  F1: {metrics.get('avg_F1', 0):.4f}")
            print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")

            # A-OKVQA特有的准确率（基于字母选择）
            correct = sum(1 for r in results if r.get('answer', 'A') == r.get('correct_choice', 'A'))
            actual_accuracy = correct / len(results) * 100 if results else 0
            print(f"  选择准确率: {actual_accuracy:.1f}%")

            # 推理质量评估（如果有rationale）
            if method_name == 'With Rationale':
                with_rationale = sum(1 for r in results if r.get('used_rationale', False))
                print(f"  使用推理依据: {with_rationale}/{len(results)}")

        except Exception as e:
            print(f"  评估失败: {e}")

    # 5. 保存结果
    print("\n4. 保存结果...")
    output_dir = Path('/data0/home/zqwang/ACL/FlashRAG/test_results_aokvqa_mock')
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
    test_aokvqa()