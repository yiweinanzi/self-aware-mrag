#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整的7个核心评估指标实现

✅ P0-5: 7项核心指标
1. EM (Exact Match)
2. F1 (Token-level F1)
3. Recall@5 (Retrieval Recall)
4. VQA-Score
5. Faithfulness
6. Attribution Precision
7. Position Bias Score
"""

import warnings
import numpy as np
from typing import List, Dict, Any, Optional
from flashrag.evaluator.metrics import BaseMetric, F1_Score, ExactMatch, Retrieval_Recall
from flashrag.evaluator.advanced_metrics import AttributionPrecisionCalculator, PositionBiasMetric


# ============================================================================
# 核心指标4: VQA-Score
# ============================================================================

class VQA_Score(BaseMetric):
    """
    VQA-Score评估指标
    
    参考VQA v2.0评分机制：
    - 多个标注答案的软匹配
    - min(#humans that provided that answer / 3, 1)
    """
    
    metric_name = "vqa_score"
    
    def __init__(self, config):
        super().__init__(config)
    
    def calculate_vqa_score(self, prediction: str, golden_answers: list) -> float:
        """
        计算VQA分数
        
        Args:
            prediction: 预测答案
            golden_answers: 标准答案列表
        
        Returns:
            float: VQA分数 [0, 1]
        """
        if not golden_answers:
            return 0.0
        
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        
        # 归一化答案
        from flashrag.evaluator.utils import normalize_answer
        normalized_pred = normalize_answer(prediction)
        
        # 统计匹配答案的数量
        match_count = 0
        for answer in golden_answers:
            normalized_answer = normalize_answer(answer)
            if normalized_pred == normalized_answer:
                match_count += 1
        
        # VQA评分公式：min(#matching_answers / 3, 1)
        vqa_score = min(match_count / 3.0, 1.0)
        
        return vqa_score
    
    def calculate_metric(self, data):
        """
        计算整个数据集的VQA分数
        
        Args:
            data: 数据对象
        
        Returns:
            (metric_score, metric_score_list)
        """
        pred_list = data.pred
        golden_answers_list = self.get_dataset_answer(data)
        
        metric_score_list = [
            self.calculate_vqa_score(pred, golden_answers)
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        
        vqa_score = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0.0
        
        return {"vqa_score": vqa_score}, metric_score_list


# ============================================================================
# 核心指标5: Faithfulness
# ============================================================================

class Faithfulness(BaseMetric):
    """
    Faithfulness（忠实度）评估指标
    
    评估生成答案与检索文档的一致性：
    - 答案中的关键信息是否源自检索文档
    - 是否存在幻觉（hallucination）
    """
    
    metric_name = "faithfulness"
    
    def __init__(self, config):
        super().__init__(config)
        self.use_llm_judge = config.get('use_llm_judge', False)
        
        if self.use_llm_judge:
            warnings.warn("LLM判断模式需要额外的LLM调用，可能较慢")
    
    def calculate_faithfulness_simple(self, prediction: str, 
                                     retrieved_texts: List[str]) -> float:
        """
        简化版忠实度计算（基于关键词覆盖）
        
        Args:
            prediction: 预测答案
            retrieved_texts: 检索到的文档列表
        
        Returns:
            float: 忠实度分数 [0, 1]
        """
        if not prediction or not retrieved_texts:
            return 0.0
        
        from flashrag.evaluator.utils import normalize_answer
        
        # 提取预测答案中的关键词
        pred_tokens = set(normalize_answer(prediction).split())
        
        # 提取检索文档中的所有关键词
        doc_tokens = set()
        for doc in retrieved_texts:
            if doc:
                doc_tokens.update(normalize_answer(doc).split())
        
        # 计算覆盖率
        if not pred_tokens:
            return 0.0
        
        covered_tokens = pred_tokens.intersection(doc_tokens)
        faithfulness = len(covered_tokens) / len(pred_tokens)
        
        return faithfulness
    
    def calculate_faithfulness_llm(self, prediction: str,
                                   retrieved_texts: List[str]) -> float:
        """
        LLM判断版忠实度（更准确但较慢）
        
        使用LLM判断答案是否忠实于检索文档
        
        Args:
            prediction: 预测答案
            retrieved_texts: 检索到的文档列表
        
        Returns:
            float: 忠实度分数 [0, 1]
        """
        # TODO: 实现LLM判断
        # 使用Qwen3-VL或其他LLM判断
        warnings.warn("LLM判断模式暂未实现，降级到简化版")
        return self.calculate_faithfulness_simple(prediction, retrieved_texts)
    
    def calculate_metric(self, data):
        """
        计算整个数据集的忠实度
        
        Args:
            data: 数据对象（需要包含retrieved_context）
        
        Returns:
            (metric_score, metric_score_list)
        """
        pred_list = data.pred
        
        # 获取检索文档
        if hasattr(data, 'retrieval_result'):
            retrieved_contexts = [
                item.get('retrieved_docs', []) if isinstance(item, dict) else []
                for item in data.retrieval_result
            ]
        else:
            warnings.warn("数据中缺少retrieval_result，忠实度将设为0")
            retrieved_contexts = [[] for _ in pred_list]
        
        # 计算忠实度
        if self.use_llm_judge:
            metric_score_list = [
                self.calculate_faithfulness_llm(pred, docs)
                for pred, docs in zip(pred_list, retrieved_contexts)
            ]
        else:
            metric_score_list = [
                self.calculate_faithfulness_simple(pred, docs)
                for pred, docs in zip(pred_list, retrieved_contexts)
            ]
        
        faithfulness = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0.0
        
        return {"faithfulness": faithfulness}, metric_score_list


# ============================================================================
# 核心指标6: Attribution Precision（使用现有实现）
# ============================================================================

class Attribution_Precision(BaseMetric):
    """
    归因精度指标（包装现有实现）
    
    评估模型归因的准确性
    """
    
    metric_name = "attribution_precision"
    
    def __init__(self, config):
        super().__init__(config)
        self.calculator = AttributionPrecisionCalculator(
            confidence_threshold=config.get('attribution_confidence_threshold', 0.5)
        )
    
    def calculate_metric(self, data):
        """
        计算整个数据集的归因精度
        
        Args:
            data: 数据对象（需要包含attributions）
        
        Returns:
            (metric_score, metric_score_list)
        """
        # 获取归因结果
        if hasattr(data, 'attributions'):
            attributions_list = data.attributions
        else:
            warnings.warn("数据中缺少attributions，归因精度将设为0")
            return {"attribution_precision": 0.0}, [0.0] * len(data.pred)
        
        # 获取ground truth源
        if hasattr(data, 'ground_truth_sources'):
            gt_sources_list = data.ground_truth_sources
        else:
            warnings.warn("数据中缺少ground_truth_sources，使用空列表")
            gt_sources_list = [[] for _ in attributions_list]
        
        # 计算归因精度
        metric_score_list = []
        for attributions, gt_sources in zip(attributions_list, gt_sources_list):
            if attributions is None:
                metric_score_list.append(0.0)
                continue
            
            try:
                result = self.calculator.compute(
                    generated_answer="",  # 不需要
                    attributions=attributions,
                    ground_truth_sources=gt_sources
                )
                metric_score_list.append(result.get('precision', 0.0))
            except Exception as e:
                warnings.warn(f"归因精度计算失败: {e}")
                metric_score_list.append(0.0)
        
        attribution_precision = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0.0
        
        return {"attribution_precision": attribution_precision}, metric_score_list


# ============================================================================
# 核心指标7: Position Bias Score（使用现有实现）
# ============================================================================

class Position_Bias_Score(BaseMetric):
    """
    位置偏差分数（包装现有实现）
    
    评估模型对文档位置的偏差程度（越低越好）
    """
    
    metric_name = "position_bias_score"
    
    def __init__(self, config):
        super().__init__(config)
        self.calculator = PositionBiasMetric()
    
    def calculate_metric(self, data):
        """
        计算整个数据集的位置偏差分数
        
        Args:
            data: 数据对象
        
        Returns:
            (metric_score, metric_score_list)
        """
        # 位置偏差需要特殊的测试集（打乱文档顺序）
        # 这里提供一个简化版实现
        
        if hasattr(data, 'position_bias_results'):
            # 如果已经计算过位置偏差
            position_bias_score = data.position_bias_results.get('average_bias', 0.0)
            metric_score_list = data.position_bias_results.get('individual_scores', [])
        else:
            warnings.warn("数据中缺少position_bias_results，位置偏差分数将设为0")
            position_bias_score = 0.0
            metric_score_list = [0.0] * len(data.pred)
        
        return {"position_bias_score": position_bias_score}, metric_score_list


# ============================================================================
# 统一的指标计算器
# ============================================================================

class CompleteMetricsCalculator:
    """
    完整的7个核心指标计算器
    
    使用示例：
    ```python
    calculator = CompleteMetricsCalculator(config)
    results = calculator.calculate_all_metrics(data)
    
    # results包含7个核心指标：
    # - EM, F1, Recall@5, VQA-Score, Faithfulness, Attribution, Position Bias
    ```
    """
    
    def __init__(self, config):
        """
        初始化指标计算器
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 初始化7个核心指标
        self.metrics = {
            'em': ExactMatch(config),
            'f1': F1_Score(config),
            'retrieval_recall_top5': Retrieval_Recall(config),
            'vqa_score': VQA_Score(config),
            'faithfulness': Faithfulness(config),
            'attribution_precision': Attribution_Precision(config),
            'position_bias_score': Position_Bias_Score(config)
        }
    
    def calculate_all_metrics(self, data) -> Dict[str, float]:
        """
        计算所有7个核心指标
        
        Args:
            data: 数据对象
        
        Returns:
            Dict[str, float]: 所有指标的结果
        """
        all_results = {}
        
        for metric_name, metric_calculator in self.metrics.items():
            try:
                result, _ = metric_calculator.calculate_metric(data)
                all_results.update(result)
            except Exception as e:
                warnings.warn(f"计算 {metric_name} 失败: {e}")
                all_results[metric_name] = 0.0
        
        return all_results
    
    def format_results(self, results: Dict[str, float]) -> str:
        """
        格式化结果为易读的字符串
        
        Args:
            results: 指标结果字典
        
        Returns:
            str: 格式化的结果
        """
        lines = []
        lines.append("=" * 60)
        lines.append("7个核心评估指标结果")
        lines.append("=" * 60)
        
        metric_names = {
            'em': 'EM (Exact Match)',
            'f1': 'F1 (Token-level)',
            'retrieval_recall_top5': 'Recall@5 (Retrieval)',
            'vqa_score': 'VQA-Score',
            'faithfulness': 'Faithfulness',
            'attribution_precision': 'Attribution Precision',
            'position_bias_score': 'Position Bias Score'
        }
        
        for key, display_name in metric_names.items():
            value = results.get(key, 0.0)
            
            # 位置偏差分数越低越好，其他都是越高越好
            if key == 'position_bias_score':
                lines.append(f"{display_name:30s}: {value:.4f} (↓)")
            else:
                lines.append(f"{display_name:30s}: {value:.4f} (↑)")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


if __name__ == '__main__':
    print("完整的7个核心评估指标实现")
    print("=" * 60)
    print("1. EM (Exact Match)")
    print("2. F1 (Token-level F1)")
    print("3. Recall@5 (Retrieval Recall)")
    print("4. VQA-Score")
    print("5. Faithfulness")
    print("6. Attribution Precision")
    print("7. Position Bias Score")
    print("=" * 60)

