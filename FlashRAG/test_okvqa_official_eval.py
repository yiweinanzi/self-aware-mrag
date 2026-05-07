#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用官方VQA评估标准测试OK-VQA
"""

import os
import sys
import json
import re
from datetime import datetime

sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from flashrag.modules.qwen3_vl import create_qwen3_vl_wrapper
from flashrag.dataset.okvqa_dataset_simple import OKVQADatasetSimple

class VQAEvaluator:
    """
    基于官方VQA评估标准的评估器
    参考: /data0/home/zqwang/ACL/open_resource/VQA-master/PythonEvaluationTools/vqaEvaluation/vqaEval.py
    """

    def __init__(self):
        # 标点符号
        self.punct = [
            ';', '/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_',
            '-', '>', '<', '@', '`', ',', '?', '!'
        ]
        self.periodStrip = re.compile("(?!<=\\d)(\\.)(?!\\d)")
        self.commaStrip = re.compile("(?<=\\d)(\\,)+(?=\\d)")

        # 冠词
        self.articles = ['a', 'an', 'the']

        # 数字映射
        self.manualMap = {
            'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3',
            'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8',
            'nine': '9', 'ten': '10'
        }

    def processPunctuation(self, inText):
        """处理标点符号"""
        outText = inText
        for p in self.punct:
            if (p + ' ' in inText or ' ' + p in inText) or (re.search(self.commaStrip, inText) != None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        """处理数字和冠词"""
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
        return ' '.join(outText)

    def standardize_answer(self, answer):
        """标准化答案"""
        if not answer:
            return ""

        # 处理标点符号
        answer = self.processPunctuation(answer)

        # 处理数字和冠词
        answer = self.processDigitArticle(answer)

        # 转小写
        answer = answer.lower().strip()

        return answer

    def compute_accuracy(self, pred_answer, gt_answers):
        """
        计算VQA准确率（官方标准）
        Args:
            pred_answer: 预测答案
            gt_answers: 标准答案列表（通常10个）
        Returns:
            准确率 (0-1)
        """
        if not pred_answer or not gt_answers:
            return 0.0

        # 标准化预测答案
        pred_answer = self.standardize_answer(pred_answer)

        # 统计匹配次数
        match_count = 0

        for gt_answer in gt_answers:
            # 标准化标准答案
            gt_answer_std = self.standardize_answer(gt_answer)

            # 检查是否匹配
            if pred_answer == gt_answer_std:
                match_count += 1

        # VQA准确率公式：min(1, match_count/3)
        accuracy = min(1.0, float(match_count) / 3.0)

        return accuracy

def main():
    print("="*80)
    print("OK-VQA 官方评估标准测试")
    print("="*80)

    # 1. 加载数据
    print("\n1. 加载数据集（带图像）")
    dataset = OKVQADatasetSimple({
        'data_dir': '/data0/home/zqwang/ACL/FlashRAG/flashrag/data/VQA',
        'split': 'val',
        'load_images': True,
    })
    print(f"✅ 加载了 {len(dataset)} 个样本，全部包含图像")

    # 2. 初始化评估器
    evaluator = VQAEvaluator()

    # 3. 查看数据格式
    print("\n2. 查看数据格式")
    for i in range(3):
        sample = dataset[i]
        print(f"\n样本 {i+1}:")
        print(f"  问题: {sample['question']}")
        print(f"  图像ID: {sample['image_id']}")
        print(f"  标准答案: {sample['golden_answers'][:5]}... (共{len(sample['golden_answers'])}个)")

    # 4. 测试官方评估标准
    print("\n3. 测试官方评估标准")
    print("-" * 40)

    test_cases = [
        # 精确匹配
        ("squash", ["squash", "squash", "tennis", "sport", "game", "play", "racquet", "ball", "court", "match"]),
        # 部分匹配（1个）
        ("race", ["race", "race", "motocross", "ride", "bike", "motorcycle", "sport", "fast", "competition", "speed"]),
        # 部分匹配（2个）
        ("tennis", ["tennis", "tennis", "sport", "game", "ball", "racquet", "court", "play", "match", "player"]),
        # 完全匹配（3个以上）
        ("cat", ["cat", "cat", "cat", "kitten", "pet", "animal", "feline", "mammal", "domestic", "pet"]),
        # 数字测试
        ("2", ["two", "2", "pair", "couple", "double", "duo", "two of them", "2 items", "pair of", "twice"]),
        # 冠词测试
        ("cat", ["a cat", "the cat", "cat", "feline", "pet", "animal", "kitty", "kitten", "puss", "tomcat"]),
    ]

    for pred, gt_answers in test_cases:
        acc = evaluator.compute_accuracy(pred, gt_answers)
        match_count = sum(1 for gt in gt_answers if evaluator.standardize_answer(pred) == evaluator.standardize_answer(gt))
        print(f"\n预测: {pred!r}")
        print(f"标准答案: {gt_answers[:5]}...")
        print(f"匹配数: {match_count}/10")
        print(f"准确率: {acc:.3f} (公式: min(1, {match_count}/3) = {acc})")

    # 5. 初始化模型进行实际测试
    print("\n\n4. 初始化Qwen3-VL模型进行实际测试")
    print("-" * 40)

    qwen3_vl = create_qwen3_vl_wrapper(
        model_path='/data0/home/zqwang/ACL/models/Qwen3-VL-8B-Instruct',
        device='cuda',
        torch_dtype='bfloat16'
    )
    print("✅ 模型加载成功")

    # 6. 测试前5个样本
    print("\n5. 测试前5个样本（使用官方评估标准）")
    print("-" * 40)

    total_accuracy = 0.0
    correct_count = 0

    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        question = sample['question']
        image = sample['image']
        golden_answers = sample['golden_answers']

        print(f"\n样本 {i+1}:")
        print(f"问题: {question}")

        # 生成答案
        try:
            # 改进的prompt
            prompt = f"""Answer the following question about the image with a short phrase. If the answer is a number, write the digit. If it's a yes/no question, answer "yes" or "no".

Question: {question}

Answer:"""

            # 生成
            answer = qwen3_vl.generate(
                text=prompt,
                image=image,
                max_new_tokens=10,
                temperature=0.0,
                do_sample=False
            ).strip()

            # 使用官方标准计算准确率
            accuracy = evaluator.compute_accuracy(answer, golden_answers)
            total_accuracy += accuracy

            # 判断是否算正确（accuracy > 0）
            if accuracy > 0:
                correct_count += 1

            print(f"生成答案: {answer!r}")
            print(f"标准化后: {evaluator.standardize_answer(answer)!r}")
            print(f"标准答案: {golden_answers[:5]}...")
            print(f"准确率: {accuracy:.3f} ({accuracy*100:.1f}%)")
            print(f"结果: {'✅' if accuracy > 0 else '❌'}")

        except Exception as e:
            print(f"❌ 生成失败: {e}")

    # 7. 统计结果
    avg_accuracy = total_accuracy / 5 * 100
    print("\n" + "="*80)
    print("测试结果（官方VQA评估标准）")
    print("="*80)
    print(f"平均准确率: {avg_accuracy:.1f}%")
    print(f"正确样本数: {correct_count}/5")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n说明:")
    print("- VQA使用软匹配：答案在10个人工标注中出现至少3次得满分")
    print("- 出现1次或2次按比例得分")
    print("- 答案会经过标准化处理（去除标点、冠词、转换数字等）")

if __name__ == "__main__":
    main()