#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
综合评估指标计算器 - 科研级修正版
Comprehensive Evaluator for Multimodal RAG

修正内容：
1. Position Bias: 基于正确答案来源的Rank分布计算 (Rank Decay)
2. Attribution: 使用 Bigram (N-gram) 匹配实现细粒度归因
3. Recall: 增加停用词过滤和严格匹配
"""

import re
import numpy as np
from typing import List, Dict, Any
from collections import Counter
from flashrag.utils.vqa_evaluator import VQAEvaluator, evaluate_vqa_accuracy

# 停用词表
STOP_WORDS = set([
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "is", "are", "was", "were", "be", "been", "this", "that", "it", "he", "she",
    "they", "i", "we", "you", "my", "your", "his", "her", "their", "what", "which"
])

class ComprehensiveEvaluator:
    """综合评估指标计算器"""

    def __init__(self):
        self.vqa_evaluator = None

    def _normalize(self, text: str) -> str:
        if not text: return ""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in self._normalize(text).split() if t not in STOP_WORDS]

    def calculate_f1(self, predicted: str, ground_truths: List[str]) -> float:
        """计算F1 Score (Bag of Words)"""
        if not predicted or not ground_truths: return 0.0

        pred_tokens = self._tokenize(predicted)
        if not pred_tokens: return 0.0

        best_f1 = 0.0
        for gt in ground_truths:
            gt_tokens = self._tokenize(gt)
            if not gt_tokens: continue

            common = Counter(pred_tokens) & Counter(gt_tokens)
            num_same = sum(common.values())

            if num_same == 0:
                f1 = 0.0
            else:
                p = num_same / len(pred_tokens)
                r = num_same / len(gt_tokens)
                f1 = 2 * p * r / (p + r)
            best_f1 = max(best_f1, f1)
        return best_f1

    def calculate_recall_at_k(self, retrieved_docs: List[Dict], ground_truths: List[str], k: int = 5) -> float:
        """计算Recall@k (基于内容覆盖)"""
        if not retrieved_docs or not ground_truths: return 0.0

        top_k_docs = retrieved_docs[:k]
        combined_text = " ".join([d.get('contents', '').lower() for d in top_k_docs])

        # 只要任意一个 Ground Truth 被文档包含，就算召回成功
        for gt in ground_truths:
            if not gt: continue
            gt_clean = gt.lower().strip()
            # 策略：如果GT很短，直接查字符串包含；如果长，查Token覆盖
            if len(gt_clean.split()) <= 3:
                if gt_clean in combined_text:
                    return 1.0
            else:
                gt_tokens = set(self._tokenize(gt))
                if not gt_tokens: continue
                doc_tokens = set(self._tokenize(combined_text))
                if len(gt_tokens & doc_tokens) / len(gt_tokens) > 0.7:
                    return 1.0
        return 0.0

    def calculate_faithfulness(self, answer: str, retrieved_docs: List[Dict], ground_truths: List[str] = None) -> float:
        """计算 Faithfulness (单词级覆盖率)"""
        if not answer or not retrieved_docs: return 0.0

        pred_tokens = self._tokenize(answer)
        if not pred_tokens: return 0.0

        doc_text = " ".join([d.get('contents', '') for d in retrieved_docs[:5]])
        doc_tokens = set(self._tokenize(doc_text))

        overlap_count = sum(1 for t in pred_tokens if t in doc_tokens)
        return overlap_count / len(pred_tokens)

    def calculate_attribution_precision(self, answer: str, retrieved_docs: List[Dict], ground_truths: List[str] = None) -> float:
        """计算 Attribution Precision (Bigram级精确归因)"""
        if not answer or not retrieved_docs: return 0.0

        # 使用 Bigram (2-gram) 进行匹配，比单个词更看重短语的一致性
        tokens = self._tokenize(answer)
        if len(tokens) < 2:
            return self.calculate_faithfulness(answer, retrieved_docs)

        pred_bigrams = set(zip(tokens, tokens[1:]))
        if not pred_bigrams: return 0.0

        doc_text = " ".join([d.get('contents', '') for d in retrieved_docs[:5]])
        doc_tokens = self._tokenize(doc_text)
        doc_bigrams = set(zip(doc_tokens, doc_tokens[1:]))

        overlap = len(pred_bigrams & doc_bigrams)
        return overlap / len(pred_bigrams)

    def calculate_position_bias_score(self, retrieved_docs: List[Dict], ground_truths: List[str] = None) -> float:
        if not retrieved_docs: return 0.5
        if not ground_truths: return 0.5  # 无法判断来源，默认中性

        # 找到哪篇文档包含答案
        best_doc_idx = -1
        for i, doc in enumerate(retrieved_docs[:5]):
            doc_content = doc.get('contents', '').lower()
            for gt in ground_truths:
                if gt.lower() in doc_content:
                    best_doc_idx = i
                    break
            if best_doc_idx != -1:
                break

        if best_doc_idx == -1:
            return 0.5 # 未召回，无法计算Bias

        bias_score = 0.5 * np.exp(-0.8 * best_doc_idx)
        return bias_score

    def evaluate_batch(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量评估所有指标"""
        if not results: return {}

        metric_names = ['accuracy', 'retrieved', 'F1', 'VQA_Score', 'Recall@5', 'Faithfulness', 'Attribution_Precision', 'Position_Bias_Score']
        sample_metrics = []

        for result in results:
            metrics = {}
            pred = result.get('answer', '')
            gts = result.get('golden_answers', [])
            docs = result.get('retrieved_docs', []) # 这里必须是模型实际看到的docs

            # 1. 准确率
            try:
                vqa_res = evaluate_vqa_accuracy(pred, gts)
                metrics['accuracy'] = vqa_res['accuracy']
            except:
                metrics['accuracy'] = 0.0

            # 2. 其他指标
            metrics['retrieved'] = len(docs) > 0
            metrics['F1'] = self.calculate_f1(pred, gts)
            metrics['VQA_Score'] = metrics['accuracy']
            metrics['Recall@5'] = self.calculate_recall_at_k(docs, gts)
            metrics['Faithfulness'] = self.calculate_faithfulness(pred, docs)
            metrics['Attribution_Precision'] = self.calculate_attribution_precision(pred, docs)
            metrics['Position_Bias_Score'] = self.calculate_position_bias_score(docs, gts)

            sample_metrics.append(metrics)

        # 聚合
        avg_metrics = {}
        for name in metric_names:
            values = [m.get(name, 0.0) for m in sample_metrics]
            avg_metrics[f'avg_{name}'] = np.mean(values) if values else 0.0

        # 兼容性字段
        avg_metrics['accuracy'] = avg_metrics['avg_accuracy']
        avg_metrics['retrieval_rate'] = avg_metrics['avg_retrieved']

        return avg_metrics

# 全局实例
comprehensive_evaluator = ComprehensiveEvaluator()

def evaluate_comprehensive_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return comprehensive_evaluator.evaluate_batch(results)