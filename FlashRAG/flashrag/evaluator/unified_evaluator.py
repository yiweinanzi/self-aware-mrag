#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一评测器 - 支持4个数据集的7个核心指标
Unified Evaluator for 4 Datasets with 7 Core Metrics

7个核心指标：
1. EM (Exact Match) - 精确匹配
2. F1 Score - Token级别F1
3. Recall@5 - 检索召回率
4. VQA-Score - VQA官方评分
5. Faithfulness - 忠实度
6. Attribution Precision - 归因精确度
7. Position Bias Score - 位置偏差分数

支持的数据集：
- OK-VQA
- A-OKVQA
- MultiModalQA
- MRAG-Bench
"""

import os
import sys
import json
import re
import string
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm

# 添加FlashRAG路径
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

# 导入现有评测器
from flashrag.utils.comprehensive_evaluator import evaluate_comprehensive_metrics
from flashrag.evaluator import Evaluator


class UnifiedEvaluator:
    """统一评测器"""

    def __init__(self, config: Dict = None):
        self.config = config or {}

        # 停用词列表
        self.stop_words = {
            "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if",
            "in", "into", "is", "it", "no", "not", "of", "on", "or", "such",
            "that", "the", "their", "then", "there", "these", "they", "this",
            "to", "was", "will", "with", "we", "you", "your", "i", "our", "us"
        }

        # 数据集特定的评测配置
        self.dataset_configs = {
            'okvqa': {
                'is_multiple_choice': False,
                'use_vqa_evaluator': True,
                'case_sensitive': False
            },
            'a-okvqa': {
                'is_multiple_choice': True,
                'use_vqa_evaluator': True,
                'case_sensitive': False
            },
            'multimodalqa': {
                'is_multiple_choice': False,
                'use_vqa_evaluator': True,
                'case_sensitive': False
            },
            'mrag-bench': {
                'is_multiple_choice': True,
                'use_vqa_evaluator': False,  # 使用精确匹配
                'case_sensitive': True  # 多选题对大小写敏感
            }
        }

    def evaluate(self, dataset_name: str, predictions: List[Dict],
                 references: List[Dict]) -> Dict[str, Any]:
        """
        评测预测结果

        Args:
            dataset_name: 数据集名称
            predictions: 预测结果列表
            references: 参考答案列表

        Returns:
            包含所有指标的评测结果
        """
        print(f"\n{'='*60}")
        print(f"评测数据集: {dataset_name.upper()}")
        print(f"样本数: {len(predictions)}")
        print(f"{'='*60}")

        # 获取数据集配置
        dataset_config = self.dataset_configs.get(dataset_name, {})

        # 准备评测数据
        eval_data = self._prepare_eval_data(predictions, references, dataset_config)

        # 使用现有的综合评测器
        try:
            # 转换格式以兼容现有评测器
            results = []
            for pred, ref in zip(predictions, references):
                result = {
                    'answer': pred.get('answer', ''),
                    'golden_answers': ref.get('golden_answers', []),
                    'retrieved_docs': pred.get('retrieved_docs', []),
                    'retrieval_result': pred.get('retrieval_result', []),
                    'attributions': pred.get('attributions', {}),
                    'position_bias_results': pred.get('position_bias_results', {})
                }
                results.append(result)

            # 使用综合评测器计算指标
            metrics = evaluate_comprehensive_metrics(results)

            # 添加数据集特定的指标
            if dataset_name == 'mrag-bench':
                # MRAG-Bench的场景准确率
                scenario_metrics = self._evaluate_mragbench_scenarios(predictions, references)
                metrics.update(scenario_metrics)

            # 格式化输出
            self._print_metrics(metrics, dataset_name)

            return metrics

        except Exception as e:
            print(f"❌ 评测失败: {e}")
            # 返回基础指标
            return self._calculate_basic_metrics(predictions, references, dataset_config)

    def _prepare_eval_data(self, predictions: List[Dict], references: List[Dict],
                          dataset_config: Dict) -> List[Dict]:
        """准备评测数据"""
        eval_data = []

        for pred, ref in zip(predictions, references):
            # 统一答案格式
            pred_answer = self._normalize_answer(
                pred.get('answer', ''),
                dataset_config.get('case_sensitive', False)
            )

            # 处理参考答案
            golden_answers = ref.get('golden_answers', [])
            if isinstance(golden_answers, str):
                golden_answers = [golden_answers]

            normalized_answers = [
                self._normalize_answer(ans, dataset_config.get('case_sensitive', False))
                for ans in golden_answers
            ]

            eval_data.append({
                'pred_answer': pred_answer,
                'golden_answers': normalized_answers,
                'retrieved_docs': pred.get('retrieved_docs', []),
                'question': ref.get('question', ''),
                'dataset': ref.get('dataset', '')
            })

        return eval_data

    def _normalize_answer(self, answer: str, case_sensitive: bool = False) -> str:
        """标准化答案"""
        if not answer:
            return ""

        # 转换为小写（如果不区分大小写）
        if not case_sensitive:
            answer = answer.lower()

        # 移除标点符号
        answer = answer.translate(str.maketrans('', '', string.punctuation))

        # 移除多余空格
        answer = ' '.join(answer.split())

        return answer

    def _calculate_basic_metrics(self, predictions: List[Dict], references: List[Dict],
                                dataset_config: Dict) -> Dict[str, float]:
        """计算基础指标（备用方案）"""
        total = len(predictions)
        if total == 0:
            return {}

        exact_match = 0
        f1_scores = []
        retrieval_rates = []

        for pred, ref in zip(predictions, references):
            pred_answer = self._normalize_answer(
                pred.get('answer', ''),
                dataset_config.get('case_sensitive', False)
            )

            golden_answers = ref.get('golden_answers', [])
            if isinstance(golden_answers, str):
                golden_answers = [golden_answers]

            # 精确匹配
            normalized_goldens = [
                self._normalize_answer(ans, dataset_config.get('case_sensitive', False))
                for ans in golden_answers
            ]
            if pred_answer in normalized_goldens:
                exact_match += 1

            # F1 Score
            max_f1 = 0
            for gold in normalized_goldens:
                f1 = self._calculate_f1(pred_answer, gold)
                max_f1 = max(max_f1, f1)
            f1_scores.append(max_f1)

            # 检索率
            retrieved_docs = pred.get('retrieved_docs', [])
            retrieval_rates.append(1 if retrieved_docs else 0)

        # 计算平均值
        metrics = {
            'accuracy': exact_match / total,
            'em': exact_match / total,
            'avg_F1': np.mean(f1_scores),
            'f1': np.mean(f1_scores),
            'retrieval_rate': np.mean(retrieval_rates),
            'accuracy_topk': exact_match / total,
            'vqa_score': exact_match / total,
            'avg_Recall@5': 0.0,  # 需要更多信息计算
            'retrieval_recall_top5': 0.0,
            'avg_Faithfulness': 0.0,
            'faithfulness': 0.0,
            'avg_Attribution_Precision': 0.0,
            'attribution_precision': 0.0,
            'avg_Position_Bias_Score': 0.0,
            'position_bias_score': 0.0
        }

        return metrics

    def _calculate_f1(self, pred: str, gold: str) -> float:
        """计算F1分数"""
        pred_tokens = self._tokenize(pred)
        gold_tokens = self._tokenize(gold)

        if not pred_tokens and not gold_tokens:
            return 1.0
        if not pred_tokens or not gold_tokens:
            return 0.0

        common_tokens = set(pred_tokens) & set(gold_tokens)
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(gold_tokens)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def _tokenize(self, text: str) -> List[str]:
        """分词并过滤停用词"""
        if not text:
            return []

        tokens = text.lower().split()
        tokens = [t for t in tokens if t not in self.stop_words]
        return tokens

    def _evaluate_mragbench_scenarios(self, predictions: List[Dict],
                                    references: List[Dict]) -> Dict[str, float]:
        """评测MRAG-Bench的场景准确率"""
        scenario_correct = defaultdict(int)
        scenario_total = defaultdict(int)
        total_correct = 0
        total_samples = 0

        for pred, ref in zip(predictions, references):
            scenario = ref.get('scenario', 'Unknown')
            pred_answer = pred.get('answer', '').strip().upper()
            gt_answer = ref.get('golden_answers', [''])[0].strip().upper()

            scenario_total[scenario] += 1
            scenario_total['Overall'] += 1

            total_samples += 1
            if pred_answer == gt_answer:
                scenario_correct[scenario] += 1
                scenario_correct['Overall'] += 1
                total_correct += 1

        # 计算场景准确率
        scenario_metrics = {}
        for scenario in sorted(scenario_total.keys()):
            if scenario_total[scenario] > 0:
                accuracy = scenario_correct[scenario] / scenario_total[scenario] * 100
                scenario_metrics[f'{scenario}_accuracy'] = accuracy

        return scenario_metrics

    def _print_metrics(self, metrics: Dict[str, Any], dataset_name: str):
        """打印评测指标"""
        print(f"\n{dataset_name.upper()} 评测结果:")
        print("-" * 60)

        # 核心指标
        core_metrics = [
            ('accuracy', '准确率'),
            ('avg_F1', 'F1 Score'),
            ('retrieval_rate', '检索率'),
            ('avg_Recall@5', 'Recall@5'),
            ('avg_Faithfulness', 'Faithfulness'),
            ('avg_Attribution_Precision', 'Attribution Precision'),
            ('avg_Position_Bias_Score', 'Position Bias Score')
        ]

        for key, name in core_metrics:
            value = metrics.get(key, 0.0)
            if isinstance(value, float):
                if 'accuracy' in key and 'scenario' not in key:
                    print(f"  {name:20s}: {value*100:.2f}%")
                else:
                    print(f"  {name:20s}: {value:.4f}")
            else:
                print(f"  {name:20s}: {value}")

        # MRAG-Bench场景指标
        if dataset_name == 'mrag-bench':
            print("\n场景准确率:")
            for key, value in metrics.items():
                if 'scenario' in key or 'accuracy' in key:
                    scenario = key.replace('_accuracy', '').replace('scenario_', '')
                    print(f"  {scenario:20s}: {value:.2f}%")

        print("-" * 60)


# 便捷函数
def evaluate_unified(dataset_name: str, predictions: List[Dict],
                    references: List[Dict], config: Dict = None) -> Dict[str, Any]:
    """便捷的统一评测函数"""
    evaluator = UnifiedEvaluator(config)
    return evaluator.evaluate(dataset_name, predictions, references)