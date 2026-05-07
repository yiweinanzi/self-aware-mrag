#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VQA官方评测标准适配器
基于VQA-master官方评测代码，实现答案标准化和soft-accuracy评分
"""

import re
import json
from typing import List, Dict, Any


class VQAEvaluator:
    """
    VQA官方评测标准实现
    参考VQA-master/PythonEvaluationTools/vqaEvaluation/vqaEval.py
    """

    def __init__(self):
        # 标点符号映射表
        self.punct = [
            ';', '/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_',
            '-', '>', '<', ',', '.', '?', '!', '$', '&', '*', '%', '@', '#', '^',
            ':', "'", '|', '~'
        ]
        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(?<=\d)(\,)+(?=\d)")

        # 冠词列表
        self.articles = ['a', 'an', 'the']

        # 缩写映射
        self.contractions = {
            "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
            "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
            "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
            "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
            "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
            "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
            "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've",
            "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
            "itll": "it'll", "let's": "let's", "mightnt": "mightn't",
            "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
            "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
            "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
            "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at",
            "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
            "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
            "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
            "somebody'd": "somebodyd", "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
            "somebodys": "somebody's", "someoned": "someone'd",
            "someoned've": "someone'd've", "someone'dve": "someone'd've",
            "someonell": "someone'll", "someones": "someone's",
            "somethingd": "something'd", "somethingd've": "something'd've",
            "something'dve": "something'd've", "somethingll": "something'll",
            "thats": "that's", "thered": "there'd", "thered've": "there'd've",
            "there'dve": "there'd've", "therere": "there're", "theres": "there's",
            "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've",
            "theyll": "they'll", "theyre": "they're", "theyve": "they've",
            "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've",
            "we'dve": "we'd've", "weve": "we've", "werent": "weren't",
            "whatll": "what'll", "whatre": "what're", "whats": "what's",
            "whatve": "what've", "whens": "when's", "whered": "where'd",
            "wheres": "where's", "whereve": "where've", "whod": "who'd",
            "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll",
            "whos": "who's", "whove": "who've", "whyll": "why'll", "whyre": "why're",
            "whys": "why's", "wont": "won't", "wouldve": "would've",
            "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've",
            "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
            "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've",
            "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've",
            "youll": "you'll", "youre": "you're", "youve": "you've"
        }

        # 数字映射表
        self.manualMap = {
            'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3',
            'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8',
            'nine': '9', 'ten': '10', 'eleven': '11', 'twelve': '12'
        }

    def process_punctuation(self, inText: str) -> str:
        """处理标点符号 - 移除或替换为空格"""
        outText = inText
        for p in self.punct:
            if (p + ' ' in inText or ' ' + p in inText) or (re.search(self.commaStrip, inText) != None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def process_digit_article(self, inText: str) -> str:
        """处理数字和冠词 - 标准化数字文本，移除冠词"""
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = ' '.join(outText)
        return outText

    def standardize_answer(self, answer: str) -> str:
        """
        标准化答案格式 - 按照VQA官方标准处理
        """
        if not answer or not isinstance(answer, str):
            return ""

        # 基本清理
        answer = answer.strip()
        answer = answer.replace('\n', ' ')
        answer = answer.replace('\t', ' ')
        answer = ' '.join(answer.split())  # 标准化空格

        # VQA官方标准化处理
        answer = self.process_punctuation(answer)
        answer = self.process_digit_article(answer)

        # 转换为小写
        answer = answer.lower().strip()

        # 确保答案是短格式（1-3个单词）
        words = answer.split()
        if len(words) > 3:
            answer = ' '.join(words[:3])  # 截取前3个单词

        return answer

    def extract_short_answer(self, long_answer: str) -> str:
        """
        从长答案中提取1-3个单词的核心答案
        这是针对OK-VQA的专门优化
        """
        if not long_answer:
            return ""

        # 标准化处理
        answer = self.standardize_answer(long_answer)

        # 如果答案已经很短（1-3个单词），直接返回
        words = answer.split()
        if len(words) <= 3:
            return answer

        # OK-VQA答案提取策略

        # 规则1: 移除常见的前缀词
        prefixes_to_remove = ['answer', 'the', 'it', 'this', 'that', 'is', 'are', 'was', 'were']
        filtered_words = []
        skip_next = False

        for i, word in enumerate(words):
            if skip_next:
                skip_next = False
                continue

            if word in prefixes_to_remove and i < len(words) - 1:
                # 如果是前缀词，跳过它和下一个词（通常是冠词等）
                skip_next = True
                continue
            elif word in prefixes_to_remove:
                continue
            else:
                filtered_words.append(word)

        # 规则2: 寻找关键名词和动作词（通常在答案的后半部分）
        if len(filtered_words) > 6:
            # 如果答案很长，取最后3-4个词，因为答案通常在末尾
            candidate_words = filtered_words[-4:]
        else:
            candidate_words = filtered_words

        # 规则3: 过滤常见的停用词
        common_stopwords = {'a', 'an', 'the', 'in', 'on', 'at', 'by', 'for', 'with', 'about', 'of', 'to', 'from', 'and', 'or', 'but'}
        final_words = [word for word in candidate_words if word not in common_stopwords]

        if final_words:
            # 取前3个最终词
            return ' '.join(final_words[:3])
        elif candidate_words:
            # 如果过滤后没有词，取候选词的前3个
            return ' '.join(candidate_words[:3])
        else:
            # 最后备选：取原答案的最后3个词
            return ' '.join(words[-3:])

    def calculate_vqa_accuracy(self, predicted: str, ground_truths: List[str]) -> Dict[str, Any]:
        """
        计算VQA标准的准确率
        使用soft-accuracy: min(1, len(matchingAns)/3)
        """
        if not predicted or not ground_truths:
            return {'accuracy': 0.0, 'matches': 0, 'processed_pred': '', 'processed_gts': []}

        # 标准化预测答案
        processed_pred = self.standardize_answer(predicted)

        # 标准化所有ground truth答案
        processed_gts = [self.standardize_answer(gt) for gt in ground_truths]

        # 计算匹配数量
        matches = 0
        for gt in processed_gts:
            if gt == processed_pred:
                matches += 1

        # VQA soft-accuracy公式
        accuracy = min(1.0, float(matches) / 3.0) * 100.0

        return {
            'accuracy': accuracy,
            'matches': matches,
            'processed_pred': processed_pred,
            'processed_gts': processed_gts,
            'is_correct': accuracy > 0
        }

    def batch_evaluate(self, predictions: List[str], ground_truths: List[List[str]]) -> Dict[str, Any]:
        """
        批量评估多个答案
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("预测答案和标准答案数量不匹配")

        results = []
        total_accuracy = 0.0
        correct_count = 0

        for pred, gts in zip(predictions, ground_truths):
            result = self.calculate_vqa_accuracy(pred, gts)
            results.append(result)
            total_accuracy += result['accuracy']
            if result['is_correct']:
                correct_count += 1

        avg_accuracy = total_accuracy / len(results) if results else 0.0

        return {
            'average_accuracy': avg_accuracy,
            'correct_count': correct_count,
            'total_count': len(results),
            'exact_match_accuracy': (correct_count / len(results) * 100.0) if results else 0.0,
            'detailed_results': results
        }


# 全局实例
vqa_evaluator = VQAEvaluator()


def standardize_vqa_answer(answer: str) -> str:
    """便捷函数：标准化单个VQA答案"""
    return vqa_evaluator.standardize_answer(answer)


def extract_okvqa_answer(long_answer: str) -> str:
    """便捷函数：为OK-VQA提取短答案"""
    return vqa_evaluator.extract_short_answer(long_answer)


def evaluate_vqa_accuracy(predicted: str, ground_truths: List[str]) -> Dict[str, Any]:
    """便捷函数：计算VQA准确率"""
    return vqa_evaluator.calculate_vqa_accuracy(predicted, ground_truths)