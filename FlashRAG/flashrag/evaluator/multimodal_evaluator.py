# -*- coding: utf-8 -*-
"""
多模态评估框架扩展
Multimodal Evaluator

扩展FlashRAG的评估器，添加多模态特定指标：
- CLIPScore: 图文对齐评估
- Attribution_Precision: 归因精度
- Position_Bias: 位置偏差评估

参考文档：创新点1-自感知多模态RAG-实施方案.md 第275-287行, 第1053-1166行
"""

import warnings
from typing import List, Dict, Any, Optional
import numpy as np

try:
    from flashrag.evaluator import Evaluator
    EVALUATOR_AVAILABLE = True
except ImportError:
    EVALUATOR_AVAILABLE = False
    warnings.warn("FlashRAG Evaluator未找到")

try:
    from transformers import CLIPModel, CLIPProcessor
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    warnings.warn("CLIP不可用，CLIPScore功能受限")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class MultimodalEvaluator:
    """
    扩展FlashRAG的评估器，添加多模态指标
    
    新增指标：
    - CLIPScore: 评估文本-图像对齐质量
    - Attribution_Precision: 评估归因精确度
    - Position_Bias: 量化位置偏差程度
    - Cross_Modal_Consistency: 跨模态一致性
    
    使用示例：
    ```python
    evaluator = MultimodalEvaluator()
    
    # 计算CLIPScore
    clip_score = evaluator.compute_clip_score(
        texts=generated_texts,
        images=query_images
    )
    
    # 计算归因精度
    attr_precision = evaluator.compute_attribution_precision(
        attributions=predicted_attributions,
        ground_truth=gt_attributions
    )
    ```
    """
    
    def __init__(self, config=None):
        """
        初始化多模态评估器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 加载CLIP模型（用于CLIPScore）
        self.clip_model = None
        self.clip_processor = None
        if CLIP_AVAILABLE:
            self._load_clip_model()
        
        # 基础评估器（如果可用）
        self.base_evaluator = None
        if EVALUATOR_AVAILABLE:
            self.base_evaluator = Evaluator(self.config)
    
    def _load_clip_model(self):
        """加载CLIP模型"""
        try:
            model_path = self.config.get('clip_model_path', '/root/autodl-tmp/models/clip-vit-large-patch14-336')
            self.clip_model = CLIPModel.from_pretrained(model_path)
            self.clip_processor = CLIPProcessor.from_pretrained(model_path)
            
            if torch.cuda.is_available():
                self.clip_model = self.clip_model.cuda()
            
            self.clip_model.eval()
            print(f"✅ CLIP模型加载成功（用于CLIPScore）")
        except Exception as e:
            warnings.warn(f"CLIP模型加载失败: {e}")
    
    def evaluate(self, dataset, metrics=None) -> Dict[str, float]:
        """
        完整评估
        
        Args:
            dataset: 数据集对象
            metrics: 要计算的指标列表
            
        Returns:
            dict: 评估结果
        """
        results = {}
        
        if metrics is None:
            metrics = ['EM', 'F1', 'CLIPScore', 'Attribution_Precision', 'Position_Bias']
        
        # 基础指标（使用FlashRAG）
        if self.base_evaluator and any(m in metrics for m in ['EM', 'F1', 'Recall']):
            base_results = self.base_evaluator.evaluate(dataset)
            results.update(base_results)
        
        # 多模态指标
        if 'CLIPScore' in metrics:
            results['CLIPScore'] = self.compute_clip_score_from_dataset(dataset)
        
        if 'Attribution_Precision' in metrics:
            results['Attribution_Precision'] = self.compute_attribution_precision_from_dataset(dataset)
        
        if 'Position_Bias' in metrics:
            results['Position_Bias'] = self.compute_position_bias_from_dataset(dataset)
        
        return results
    
    # =========================================================================
    # 指标1: CLIPScore
    # =========================================================================
    
    def compute_clip_score(self, texts: List[str], images: List) -> float:
        """
        计算CLIPScore
        
        评估文本-图像对齐质量
        
        Args:
            texts: 文本列表
            images: 图像列表（PIL.Image）
            
        Returns:
            float: CLIPScore（平均值）
        """
        if not CLIP_AVAILABLE or self.clip_model is None:
            warnings.warn("CLIP不可用，无法计算CLIPScore")
            return 0.0
        
        scores = []
        
        with torch.no_grad():
            for text, image in zip(texts, images):
                if image is None:
                    continue
                
                try:
                    # 处理输入
                    inputs = self.clip_processor(
                        text=[text],
                        images=image,
                        return_tensors="pt",
                        padding=True
                    )
                    
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    
                    # 计算相似度
                    outputs = self.clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    score = logits_per_image[0, 0].item()
                    
                    scores.append(score)
                
                except Exception as e:
                    warnings.warn(f"计算CLIPScore失败: {e}")
                    continue
        
        return np.mean(scores) if scores else 0.0
    
    def compute_clip_score_from_dataset(self, dataset) -> float:
        """从数据集计算CLIPScore"""
        texts = []
        images = []
        
        for item in dataset:
            pred = item.pred if hasattr(item, 'pred') else item.get('pred', '')
            image = item.image if hasattr(item, 'image') else item.get('image')
            
            if pred and image:
                texts.append(pred)
                images.append(image)
        
        if not texts:
            return 0.0
        
        return self.compute_clip_score(texts, images)
    
    # =========================================================================
    # 指标2: Attribution_Precision
    # =========================================================================
    
    def compute_attribution_precision(self, predicted_attributions: List[Dict],
                                     ground_truth_attributions: List[Dict]) -> Dict[str, float]:
        """
        计算归因精度
        
        参考文档第1094-1119行
        
        Args:
            predicted_attributions: 预测的归因结果
            ground_truth_attributions: 真实的归因结果
            
        Returns:
            dict: {'precision': ..., 'recall': ..., 'f1': ...}
        """
        if not predicted_attributions or not ground_truth_attributions:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        correct = 0
        total_predicted = 0
        total_ground_truth = 0
        
        for pred, gt in zip(predicted_attributions, ground_truth_attributions):
            if pred is None or gt is None:
                continue
            
            # 提取预测和真实的归因ID集合
            pred_ids = set()
            if isinstance(pred, dict):
                # 视觉归因
                if 'visual' in pred:
                    pred_ids.update([a.get('source_image_id') for a in pred['visual']])
                # 文本归因
                if 'text' in pred:
                    pred_ids.update([a.get('source_text_id') for a in pred['text']])
            
            gt_ids = set()
            if isinstance(gt, dict):
                if 'visual' in gt:
                    gt_ids.update([a.get('source_image_id') for a in gt['visual']])
                if 'text' in gt:
                    gt_ids.update([a.get('source_text_id') for a in gt['text']])
            
            # 计算正确归因数量
            correct += len(pred_ids & gt_ids)
            total_predicted += len(pred_ids)
            total_ground_truth += len(gt_ids)
        
        # 计算precision, recall, F1
        precision = correct / total_predicted if total_predicted > 0 else 0.0
        recall = correct / total_ground_truth if total_ground_truth > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def compute_attribution_precision_from_dataset(self, dataset) -> float:
        """从数据集计算归因精度"""
        predicted = []
        ground_truth = []
        
        for item in dataset:
            pred_attr = item.attribution if hasattr(item, 'attribution') else None
            gt_attr = item.ground_truth_attribution if hasattr(item, 'ground_truth_attribution') else None
            
            if pred_attr:
                predicted.append(pred_attr)
                ground_truth.append(gt_attr if gt_attr else {})
        
        if not predicted:
            return 0.0
        
        result = self.compute_attribution_precision(predicted, ground_truth)
        return result['f1']
    
    # =========================================================================
    # 指标3: Position_Bias
    # =========================================================================
    
    def compute_position_bias(self, model, test_samples: List[Dict]) -> float:
        """
        量化位置偏差程度
        
        参考文档第1141-1166行
        
        方法：
        1. 将相同内容放在不同位置
        2. 测量性能变化
        3. 标准差越大，位置偏差越严重
        
        Args:
            model: 模型（用于生成答案）
            test_samples: 测试样本
            
        Returns:
            float: 位置偏差分数（越低越好）
        """
        bias_scores = []
        positions = ['beginning', 'middle', 'end']
        
        for sample in test_samples:
            performance = {}
            
            for pos in positions:
                # 重排序context，将关键信息放在指定位置
                reordered_context = self._reorder_context(
                    sample['context'], 
                    pos, 
                    sample['key_info']
                )
                
                # 生成答案
                answer = model.generate(sample['query'], reordered_context)
                
                # 评估
                performance[pos] = self._evaluate_answer(
                    answer, sample['ground_truth']
                )
            
            # 计算位置敏感度（标准差）
            bias = np.std([performance[p] for p in positions])
            bias_scores.append(bias)
        
        return np.mean(bias_scores)
    
    def compute_position_bias_from_dataset(self, dataset) -> float:
        """从数据集计算位置偏差"""
        # 简化版：比较检索结果在不同位置的性能
        # 完整版需要重排序测试集
        
        if not hasattr(dataset, 'position_bias_score'):
            warnings.warn("数据集没有position_bias_score字段，返回0")
            return 0.0
        
        bias_scores = [
            item.position_bias_score for item in dataset 
            if hasattr(item, 'position_bias_score')
        ]
        
        return np.mean(bias_scores) if bias_scores else 0.0
    
    # =========================================================================
    # 指标4: Cross_Modal_Consistency
    # =========================================================================
    
    def compute_cross_modal_consistency(self, text_answers: List[str],
                                       visual_evidences: List,
                                       retrieved_contexts: List[List[Dict]]) -> float:
        """
        评估跨模态生成的一致性
        
        参考文档第1122-1138行
        
        检查：
        1. 文本描述与视觉证据是否一致
        2. 是否存在模态间的矛盾
        
        Args:
            text_answers: 文本答案列表
            visual_evidences: 视觉证据列表
            retrieved_contexts: 检索上下文列表
            
        Returns:
            float: 一致性分数 [0, 1]
        """
        if not CLIP_AVAILABLE:
            return 0.0
        
        consistency_scores = []
        
        for text, visual, context in zip(text_answers, visual_evidences, retrieved_contexts):
            if visual is None:
                continue
            
            # 使用CLIP评估文本-图像对齐
            clip_similarity = self._compute_clip_similarity(text, visual)
            
            # 检查逻辑一致性（简化版）
            logical_consistency = 1.0  # TODO: 实现逻辑一致性检查
            
            consistency = (clip_similarity + logical_consistency) / 2
            consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _compute_clip_similarity(self, text: str, image) -> float:
        """使用CLIP计算文本-图像相似度"""
        if not CLIP_AVAILABLE or self.clip_model is None:
            return 0.0
        
        try:
            inputs = self.clip_processor(
                text=[text],
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                similarity = outputs.logits_per_image[0, 0].item()
            
            # 归一化到[0, 1]
            normalized_score = (similarity + 1) / 2  # CLIP输出范围大约[-1, 1]
            
            return normalized_score
        
        except Exception as e:
            warnings.warn(f"CLIP相似度计算失败: {e}")
            return 0.0
    
    # =========================================================================
    # 辅助方法
    # =========================================================================
    
    def _reorder_context(self, context: List[str], target_position: str,
                        key_info: str) -> List[str]:
        """
        重排序context，将关键信息放在指定位置
        
        Args:
            context: 原始context列表
            target_position: 'beginning', 'middle', 'end'
            key_info: 关键信息
            
        Returns:
            List[str]: 重排序后的context
        """
        if not context:
            return context
        
        # 找到包含关键信息的项
        key_idx = -1
        for idx, item in enumerate(context):
            if key_info.lower() in item.lower():
                key_idx = idx
                break
        
        if key_idx == -1:
            return context
        
        # 重排序
        key_item = context[key_idx]
        other_items = [c for i, c in enumerate(context) if i != key_idx]
        
        if target_position == 'beginning':
            return [key_item] + other_items
        elif target_position == 'end':
            return other_items + [key_item]
        else:  # middle
            mid = len(other_items) // 2
            return other_items[:mid] + [key_item] + other_items[mid:]
    
    def _evaluate_answer(self, answer: str, ground_truth: List[str]) -> float:
        """
        评估单个答案
        
        简化版：使用Exact Match
        """
        answer_lower = answer.lower().strip()
        for gt in ground_truth:
            if gt.lower().strip() in answer_lower or answer_lower in gt.lower().strip():
                return 1.0
        return 0.0


class AttributionPrecisionCalculator:
    """
    归因精度计算器（独立工具）
    
    参考文档第1094-1119行
    """
    
    @staticmethod
    def compute(generated_answer: str, attributions: Dict,
               ground_truth_sources: List[str]) -> Dict[str, float]:
        """
        计算归因精确度
        
        Args:
            generated_answer: 生成的答案
            attributions: 归因结果 [{'region/span': ..., 'confidence': ...}]
            ground_truth_sources: 人工标注的真实来源
            
        Returns:
            dict: {'precision': ..., 'recall': ..., 'f1': ...}
        """
        if not attributions or not ground_truth_sources:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # 提取预测的来源
        predicted_sources = set()
        
        if isinstance(attributions, dict):
            # 视觉归因
            if 'visual' in attributions:
                predicted_sources.update([
                    a.get('source_image_id') for a in attributions['visual']
                    if a.get('confidence', 0) > 0.5
                ])
            # 文本归因
            if 'text' in attributions:
                predicted_sources.update([
                    a.get('source_text_id') for a in attributions['text']
                    if a.get('confidence', 0) > 0.5
                ])
        
        gt_sources = set(ground_truth_sources)
        
        # 计算交集
        correct = len(predicted_sources & gt_sources)
        
        precision = correct / len(predicted_sources) if predicted_sources else 0.0
        recall = correct / len(gt_sources) if gt_sources else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


class PositionBiasEvaluator:
    """
    位置偏差评估器（独立工具）
    
    参考文档第1141-1166行
    """
    
    @staticmethod
    def evaluate(model, test_samples: List[Dict]) -> float:
        """
        量化位置偏差程度
        
        方法：
        1. 将相同内容放在不同位置
        2. 测量性能变化
        3. 标准差作为偏差指标
        
        Args:
            model: 待评估的模型
            test_samples: 测试样本列表
                [
                    {
                        'query': ...,
                        'context': [...],
                        'key_info': ...,
                        'ground_truth': ...
                    },
                    ...
                ]
            
        Returns:
            float: 位置偏差分数（越低越好）
        """
        bias_scores = []
        positions = ['beginning', 'middle', 'end']
        
        for sample in test_samples:
            performance = {}
            
            for pos in positions:
                # 重排序
                context = PositionBiasEvaluator._reorder_context(
                    sample['context'],
                    pos,
                    sample.get('key_info', '')
                )
                
                # 生成答案（这里需要模型实现generate接口）
                try:
                    answer = model.generate(sample['query'], context)
                    
                    # 评估准确性
                    score = PositionBiasEvaluator._evaluate_answer(
                        answer, sample['ground_truth']
                    )
                    performance[pos] = score
                
                except Exception as e:
                    warnings.warn(f"生成答案失败: {e}")
                    performance[pos] = 0.0
            
            # 计算位置敏感度（标准差）
            bias = np.std(list(performance.values()))
            bias_scores.append(bias)
        
        return np.mean(bias_scores) if bias_scores else 0.0
    
    @staticmethod
    def _reorder_context(context: List[str], position: str, key_info: str) -> List[str]:
        """重排序context"""
        if not context or not key_info:
            return context
        
        # 找到关键信息
        key_idx = -1
        for idx, item in enumerate(context):
            if key_info.lower() in item.lower():
                key_idx = idx
                break
        
        if key_idx == -1:
            return context
        
        key_item = context[key_idx]
        others = [c for i, c in enumerate(context) if i != key_idx]
        
        if position == 'beginning':
            return [key_item] + others
        elif position == 'end':
            return others + [key_item]
        else:  # middle
            mid = len(others) // 2
            return others[:mid] + [key_item] + others[mid:]
    
    @staticmethod
    def _evaluate_answer(answer: str, ground_truth: List[str]) -> float:
        """评估答案（简化版EM）"""
        answer = answer.lower().strip()
        for gt in ground_truth:
            if gt.lower().strip() in answer or answer in gt.lower().strip():
                return 1.0
        return 0.0

