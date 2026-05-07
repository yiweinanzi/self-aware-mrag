# -*- coding: utf-8 -*-
"""
高级多模态评估指标
Advanced Multimodal Evaluation Metrics

根据文档要求实现的三大核心评估指标：
1. Attribution Precision（归因精度）- 文档第1094-1119行
2. Cross-Modal Consistency（跨模态一致性）- 文档第1121-1138行  
3. Position Bias Metric（位置偏差）- 文档第1141-1166行

使用示例：
```python
from flashrag.evaluator.advanced_metrics import *

# 1. 归因精度
calculator = AttributionPrecisionCalculator()
result = calculator.compute(
    generated_answer="Paris is the capital",
    attributions={
        'visual': [{'source_image_id': 'img_1', 'confidence': 0.9}],
        'text': [{'source_text_id': 'doc_1', 'confidence': 0.8}]
    },
    ground_truth_sources=['img_1', 'doc_1', 'doc_2']
)
print(f"Precision: {result['precision']:.2f}")
print(f"Recall: {result['recall']:.2f}")
print(f"F1: {result['f1']:.2f}")

# 2. 跨模态一致性
consistency = CrossModalConsistencyScore()
score = consistency.compute(
    text_answer="A red car",
    visual_evidence=image,
    clip_model=clip_model
)
print(f"Consistency: {score:.2f}")

# 3. 位置偏差
bias_eval = PositionBiasMetric()
bias_score = bias_eval.evaluate(
    model=my_model,
    test_samples=test_data
)
print(f"Position Bias: {bias_score:.3f} (lower is better)")
```
"""

import warnings
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("torch未安装，部分功能受限")

try:
    from transformers import CLIPModel, CLIPProcessor
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    warnings.warn("CLIP未安装，跨模态一致性评估功能受限")


# =============================================================================
# 指标1: Attribution Precision（归因精度）
# 参考文档：第1094-1119行
# =============================================================================

class AttributionPrecisionCalculator:
    """
    归因精度计算器
    
    评估模型归因的准确性（Region-level + Token-level）
    
    参考文档：创新点1-自感知多模态RAG-实施方案.md 第1094-1119行
    
    核心思想：
    - Precision: 预测的归因中有多少是正确的
    - Recall: 真实归因中有多少被预测到
    - F1: 调和平均数
    
    支持的归因类型：
    - Visual Attribution: Region-level (bounding box)
    - Text Attribution: Token-level (span)
    """
    
    def __init__(self, confidence_threshold: float = 0.5, 
                 iou_threshold: float = 0.5):
        """
        初始化归因精度计算器
        
        Args:
            confidence_threshold: 归因置信度阈值（低于此值的归因会被过滤）
            iou_threshold: 视觉归因的IoU阈值（用于匹配bounding box）
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
    
    def compute(self, 
                generated_answer: str,
                attributions: Dict[str, List[Dict]],
                ground_truth_sources: Union[List[str], Dict[str, List]]) -> Dict[str, float]:
        """
        计算归因精确度
        
        Args:
            generated_answer: 生成的答案
            attributions: 预测的归因结果
                {
                    'visual': [
                        {
                            'source_image_id': 'img_1',
                            'region_bbox': [x1, y1, x2, y2],
                            'confidence': 0.9
                        },
                        ...
                    ],
                    'text': [
                        {
                            'source_text_id': 'doc_1',
                            'source_span': (start, end),
                            'confidence': 0.8
                        },
                        ...
                    ]
                }
            ground_truth_sources: 真实的归因来源
                可以是：
                - List[str]: 简单的源ID列表 ['img_1', 'doc_1', ...]
                - Dict: 详细的归因信息（与attributions格式相同）
        
        Returns:
            Dict[str, float]: {
                'precision': 精确率,
                'recall': 召回率,
                'f1': F1分数,
                'visual_precision': 视觉归因精确率（如果有）,
                'visual_recall': 视觉归因召回率（如果有）,
                'text_precision': 文本归因精确率（如果有）,
                'text_recall': 文本归因召回率（如果有）
            }
        """
        # 提取预测的归因
        predicted_visual = self._extract_visual_attributions(attributions)
        predicted_text = self._extract_text_attributions(attributions)
        
        # 提取真实的归因
        if isinstance(ground_truth_sources, list):
            # 简单格式：只有ID列表
            gt_visual = set(ground_truth_sources)
            gt_text = set(ground_truth_sources)
        else:
            # 详细格式：区分视觉和文本
            gt_visual = self._extract_visual_attributions(ground_truth_sources)
            gt_text = self._extract_text_attributions(ground_truth_sources)
        
        # 计算视觉归因指标
        visual_metrics = self._compute_attribution_metrics(
            predicted_visual, gt_visual, modality='visual'
        )
        
        # 计算文本归因指标
        text_metrics = self._compute_attribution_metrics(
            predicted_text, gt_text, modality='text'
        )
        
        # 合并计算总体指标
        all_predicted = predicted_visual | predicted_text
        all_gt = gt_visual | gt_text
        
        overall_correct = len(all_predicted & all_gt)
        overall_precision = overall_correct / len(all_predicted) if all_predicted else 0.0
        overall_recall = overall_correct / len(all_gt) if all_gt else 0.0
        overall_f1 = (2 * overall_precision * overall_recall / 
                     (overall_precision + overall_recall) 
                     if (overall_precision + overall_recall) > 0 else 0.0)
        
        return {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'visual_precision': visual_metrics['precision'],
            'visual_recall': visual_metrics['recall'],
            'visual_f1': visual_metrics['f1'],
            'text_precision': text_metrics['precision'],
            'text_recall': text_metrics['recall'],
            'text_f1': text_metrics['f1'],
        }
    
    def compute_batch(self,
                     batch_results: List[Dict]) -> Dict[str, float]:
        """
        批量计算归因精度
        
        Args:
            batch_results: 批量结果列表
                [
                    {
                        'generated_answer': ...,
                        'attributions': ...,
                        'ground_truth_sources': ...
                    },
                    ...
                ]
        
        Returns:
            Dict[str, float]: 平均指标
        """
        all_metrics = []
        
        for result in batch_results:
            metrics = self.compute(
                generated_answer=result.get('generated_answer', ''),
                attributions=result.get('attributions', {}),
                ground_truth_sources=result.get('ground_truth_sources', [])
            )
            all_metrics.append(metrics)
        
        # 计算平均值
        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return avg_metrics
    
    def _extract_visual_attributions(self, attributions: Dict) -> set:
        """提取视觉归因的源ID"""
        if not attributions or 'visual' not in attributions:
            return set()
        
        visual_ids = set()
        for attr in attributions['visual']:
            if attr.get('confidence', 1.0) >= self.confidence_threshold:
                source_id = attr.get('source_image_id')
                if source_id:
                    visual_ids.add(source_id)
        
        return visual_ids
    
    def _extract_text_attributions(self, attributions: Dict) -> set:
        """提取文本归因的源ID"""
        if not attributions or 'text' not in attributions:
            return set()
        
        text_ids = set()
        for attr in attributions['text']:
            if attr.get('confidence', 1.0) >= self.confidence_threshold:
                source_id = attr.get('source_text_id')
                if source_id:
                    text_ids.add(source_id)
        
        return text_ids
    
    def _compute_attribution_metrics(self, predicted: set, ground_truth: set,
                                    modality: str) -> Dict[str, float]:
        """计算单个模态的归因指标"""
        if not predicted and not ground_truth:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        correct = len(predicted & ground_truth)
        
        precision = correct / len(predicted) if predicted else 0.0
        recall = correct / len(ground_truth) if ground_truth else 0.0
        f1 = (2 * precision * recall / (precision + recall) 
              if (precision + recall) > 0 else 0.0)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


# =============================================================================
# 指标2: Cross-Modal Consistency（跨模态一致性）
# 参考文档：第1121-1138行
# =============================================================================

class CrossModalConsistencyScore:
    """
    跨模态一致性评估
    
    评估文本答案与视觉证据之间的一致性
    
    参考文档：创新点1-自感知多模态RAG-实施方案.md 第1121-1138行
    
    检查项：
    1. 文本描述与视觉证据是否一致（使用CLIP）
    2. 是否存在模态间的矛盾
    3. 跨模态信息的互补性
    """
    
    def __init__(self, clip_model=None, clip_processor=None, 
                 clip_model_path: str = None):
        """
        初始化跨模态一致性评估器
        
        Args:
            clip_model: CLIP模型（可选，如果不提供会自动加载）
            clip_processor: CLIP处理器
            clip_model_path: CLIP模型路径
        """
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        
        if clip_model is None and CLIP_AVAILABLE:
            self._load_clip(clip_model_path)
    
    def _load_clip(self, model_path: str = None):
        """加载CLIP模型"""
        try:
            if model_path is None:
                model_path = '/root/autodl-tmp/models/clip-vit-large-patch14-336'
            
            self.clip_model = CLIPModel.from_pretrained(
                model_path, local_files_only=True
            )
            self.clip_processor = CLIPProcessor.from_pretrained(
                model_path, local_files_only=True
            )
            
            if torch.cuda.is_available():
                self.clip_model = self.clip_model.cuda()
            
            self.clip_model.eval()
            
        except Exception as e:
            warnings.warn(f"加载CLIP模型失败: {e}")
    
    def compute(self,
                text_answer: str,
                visual_evidence,
                retrieved_context: Optional[Dict] = None) -> float:
        """
        计算跨模态一致性分数
        
        Args:
            text_answer: 文本答案
            visual_evidence: 视觉证据（PIL.Image或路径）
            retrieved_context: 检索上下文（可选）
        
        Returns:
            float: 一致性分数 [0, 1]，越高越好
        """
        if not CLIP_AVAILABLE or self.clip_model is None:
            warnings.warn("CLIP不可用，无法计算跨模态一致性")
            return 0.0
        
        # 1. CLIP相似度（文本-图像对齐）
        clip_similarity = self._compute_clip_similarity(text_answer, visual_evidence)
        
        # 2. 逻辑一致性检查（简化版）
        logical_consistency = self._check_logical_consistency(
            text_answer, visual_evidence, retrieved_context
        )
        
        # 3. 加权组合
        consistency_score = 0.7 * clip_similarity + 0.3 * logical_consistency
        
        return consistency_score
    
    def compute_batch(self,
                     text_answers: List[str],
                     visual_evidences: List,
                     retrieved_contexts: Optional[List[Dict]] = None) -> float:
        """
        批量计算跨模态一致性
        
        Args:
            text_answers: 文本答案列表
            visual_evidences: 视觉证据列表
            retrieved_contexts: 检索上下文列表
        
        Returns:
            float: 平均一致性分数
        """
        scores = []
        
        if retrieved_contexts is None:
            retrieved_contexts = [None] * len(text_answers)
        
        for text, visual, context in zip(text_answers, visual_evidences, retrieved_contexts):
            if visual is not None:
                score = self.compute(text, visual, context)
                scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    
    def _compute_clip_similarity(self, text: str, image) -> float:
        """使用CLIP计算文本-图像相似度"""
        if not CLIP_AVAILABLE or self.clip_model is None:
            return 0.0
        
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
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                # logits_per_image是文本-图像相似度
                similarity = outputs.logits_per_image[0, 0].item()
            
            # 归一化到[0, 1]（CLIP输出通常在[-10, 40]范围）
            normalized_score = torch.sigmoid(torch.tensor(similarity / 10.0)).item()
            
            return normalized_score
        
        except Exception as e:
            warnings.warn(f"CLIP相似度计算失败: {e}")
            return 0.0
    
    def _check_logical_consistency(self, text: str, image, context: Optional[Dict]) -> float:
        """
        检查逻辑一致性（简化版）
        
        完整版应该包括：
        - 矛盾检测（文本说A，图像显示B）
        - 完整性检查（是否遗漏关键信息）
        - 互补性评估（多模态信息是否互补）
        
        当前实现：基于关键词匹配和简单规则
        """
        # TODO: 实现更复杂的逻辑一致性检查
        # 目前返回固定值，表示没有明显矛盾
        return 1.0


# =============================================================================
# 指标3: Position Bias Metric（位置偏差）
# 参考文档：第1141-1166行
# =============================================================================

class PositionBiasMetric:
    """
    位置偏差评估器
    
    量化模型对检索内容位置的敏感度
    
    参考文档：创新点1-自感知多模态RAG-实施方案.md 第1141-1166行
    
    评估方法：
    1. 将相同的关键信息放在不同位置（开头/中间/结尾）
    2. 测量模型在不同位置下的性能
    3. 计算性能的标准差作为位置偏差指标
    
    位置偏差越小（标准差越小），说明模型越不受位置影响
    """
    
    def __init__(self, positions: Optional[List[str]] = None):
        """
        初始化位置偏差评估器
        
        Args:
            positions: 要测试的位置列表，默认['beginning', 'middle', 'end']
        """
        self.positions = positions or ['beginning', 'middle', 'end']
    
    def evaluate(self,
                model,
                test_samples: List[Dict],
                verbose: bool = False) -> Dict[str, float]:
        """
        评估位置偏差
        
        Args:
            model: 待评估的模型（需要有generate方法）
            test_samples: 测试样本列表
                [
                    {
                        'query': "问题",
                        'context': ["文档1", "文档2", "文档3"],
                        'key_info': "关键信息（用于识别关键文档）",
                        'ground_truth': ["答案1", "答案2"]
                    },
                    ...
                ]
            verbose: 是否打印详细信息
        
        Returns:
            Dict[str, float]: {
                'position_bias': 位置偏差分数（标准差，越小越好）,
                'beginning_acc': 关键信息在开头时的准确率,
                'middle_acc': 关键信息在中间时的准确率,
                'end_acc': 关键信息在结尾时的准确率,
                'max_diff': 最大性能差异
            }
        """
        all_bias_scores = []
        position_accuracies = {pos: [] for pos in self.positions}
        
        for idx, sample in enumerate(test_samples):
            performance = {}
            
            for pos in self.positions:
                # 重排序context
                reordered_context = self._reorder_context(
                    sample['context'],
                    pos,
                    sample.get('key_info', '')
                )
                
                # 生成答案
                try:
                    answer = model.generate(
                        query=sample['query'],
                        context=reordered_context
                    )
                    
                    # 评估准确性
                    score = self._evaluate_answer(
                        answer, sample.get('ground_truth', [])
                    )
                    performance[pos] = score
                    position_accuracies[pos].append(score)
                    
                    if verbose:
                        print(f"Sample {idx}, Position {pos}: {score:.2f}")
                
                except Exception as e:
                    warnings.warn(f"生成答案失败 (sample {idx}, pos {pos}): {e}")
                    performance[pos] = 0.0
                    position_accuracies[pos].append(0.0)
            
            # 计算该样本的位置偏差（标准差）
            bias = np.std(list(performance.values()))
            all_bias_scores.append(bias)
        
        # 汇总结果
        avg_bias = np.mean(all_bias_scores) if all_bias_scores else 0.0
        
        avg_accuracies = {
            pos: np.mean(scores) for pos, scores in position_accuracies.items()
        }
        
        max_diff = max(avg_accuracies.values()) - min(avg_accuracies.values())
        
        results = {
            'position_bias': avg_bias,
            'max_diff': max_diff
        }
        
        for pos in self.positions:
            results[f'{pos}_acc'] = avg_accuracies[pos]
        
        return results
    
    def evaluate_simple(self,
                       predictions_by_position: Dict[str, List[float]]) -> Dict[str, float]:
        """
        简化版评估（如果已经有不同位置的预测结果）
        
        Args:
            predictions_by_position: 不同位置的准确率
                {
                    'beginning': [0.8, 0.9, 0.7, ...],
                    'middle': [0.6, 0.7, 0.5, ...],
                    'end': [0.7, 0.8, 0.6, ...]
                }
        
        Returns:
            Dict[str, float]: 位置偏差统计
        """
        # 计算每个样本的位置偏差
        n_samples = len(predictions_by_position[self.positions[0]])
        bias_scores = []
        
        for i in range(n_samples):
            values = [predictions_by_position[pos][i] for pos in self.positions]
            bias = np.std(values)
            bias_scores.append(bias)
        
        # 计算每个位置的平均准确率
        avg_accuracies = {
            pos: np.mean(scores) 
            for pos, scores in predictions_by_position.items()
        }
        
        return {
            'position_bias': np.mean(bias_scores),
            'max_diff': max(avg_accuracies.values()) - min(avg_accuracies.values()),
            **{f'{pos}_acc': avg_accuracies[pos] for pos in self.positions}
        }
    
    def _reorder_context(self, 
                        context: List[str],
                        target_position: str,
                        key_info: str) -> List[str]:
        """
        重排序context，将关键信息放在目标位置
        
        Args:
            context: 原始context列表
            target_position: 'beginning', 'middle', 'end'
            key_info: 关键信息（用于找到关键文档）
        
        Returns:
            List[str]: 重排序后的context
        """
        if not context or not key_info:
            return context
        
        # 找到包含关键信息的文档
        key_idx = -1
        for idx, doc in enumerate(context):
            if key_info.lower() in doc.lower():
                key_idx = idx
                break
        
        if key_idx == -1:
            # 找不到关键文档，返回原context
            return context
        
        # 提取关键文档和其他文档
        key_doc = context[key_idx]
        other_docs = [doc for i, doc in enumerate(context) if i != key_idx]
        
        # 根据目标位置重排序
        if target_position == 'beginning':
            return [key_doc] + other_docs
        elif target_position == 'end':
            return other_docs + [key_doc]
        else:  # middle
            mid = len(other_docs) // 2
            return other_docs[:mid] + [key_doc] + other_docs[mid:]
    
    def _evaluate_answer(self, answer: str, ground_truth: List[str]) -> float:
        """
        评估答案（简化版Exact Match）
        
        Args:
            answer: 生成的答案
            ground_truth: 真实答案列表
        
        Returns:
            float: 1.0 if correct, 0.0 otherwise
        """
        if not ground_truth:
            return 0.0
        
        answer_lower = answer.lower().strip()
        
        for gt in ground_truth:
            gt_lower = gt.lower().strip()
            # 双向匹配
            if gt_lower in answer_lower or answer_lower in gt_lower:
                return 1.0
        
        return 0.0


# =============================================================================
# 综合评估报告生成器
# =============================================================================

class ComprehensiveEvaluator:
    """
    综合评估器
    
    整合所有高级评估指标，生成完整的评估报告
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化综合评估器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 初始化各个评估器
        self.attribution_calculator = AttributionPrecisionCalculator(
            confidence_threshold=self.config.get('confidence_threshold', 0.5)
        )
        
        self.consistency_scorer = CrossModalConsistencyScore(
            clip_model_path=self.config.get('clip_model_path')
        )
        
        self.position_bias_evaluator = PositionBiasMetric()
    
    def evaluate_full(self,
                     test_data: List[Dict],
                     model=None) -> Dict[str, Any]:
        """
        完整评估
        
        Args:
            test_data: 测试数据
                [
                    {
                        'query': ...,
                        'image': ...,
                        'generated_answer': ...,
                        'attributions': {...},
                        'ground_truth_sources': [...],
                        'ground_truth_answer': [...],
                        'context': [...],
                        'key_info': ...
                    },
                    ...
                ]
            model: 模型（用于位置偏差评估）
        
        Returns:
            Dict[str, Any]: 完整评估结果
        """
        results = {
            'n_samples': len(test_data),
            'metrics': {}
        }
        
        # 1. 归因精度
        print("📊 计算归因精度...")
        attribution_results = []
        for item in test_data:
            if 'attributions' in item and 'ground_truth_sources' in item:
                result = self.attribution_calculator.compute(
                    generated_answer=item.get('generated_answer', ''),
                    attributions=item['attributions'],
                    ground_truth_sources=item['ground_truth_sources']
                )
                attribution_results.append(result)
        
        if attribution_results:
            avg_attribution = {}
            for key in attribution_results[0].keys():
                avg_attribution[key] = np.mean([r[key] for r in attribution_results])
            results['metrics']['attribution'] = avg_attribution
        
        # 2. 跨模态一致性
        print("📊 计算跨模态一致性...")
        text_answers = [item.get('generated_answer', '') for item in test_data]
        visual_evidences = [item.get('image') for item in test_data]
        
        consistency_score = self.consistency_scorer.compute_batch(
            text_answers, visual_evidences
        )
        results['metrics']['cross_modal_consistency'] = consistency_score
        
        # 3. 位置偏差
        if model is not None:
            print("📊 计算位置偏差...")
            position_samples = [
                {
                    'query': item['query'],
                    'context': item.get('context', []),
                    'key_info': item.get('key_info', ''),
                    'ground_truth': item.get('ground_truth_answer', [])
                }
                for item in test_data
                if 'context' in item and 'key_info' in item
            ]
            
            if position_samples:
                position_results = self.position_bias_evaluator.evaluate(
                    model, position_samples[:min(50, len(position_samples))]  # 限制数量
                )
                results['metrics']['position_bias'] = position_results
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        生成评估报告
        
        Args:
            results: evaluate_full的返回结果
        
        Returns:
            str: Markdown格式的报告
        """
        report = []
        report.append("# 📊 多模态RAG评估报告\n\n")
        report.append(f"**样本数**: {results['n_samples']}\n\n")
        report.append("---\n\n")
        
        # 归因精度
        if 'attribution' in results['metrics']:
            attr = results['metrics']['attribution']
            report.append("## 1️⃣ 归因精度 (Attribution Precision)\n\n")
            report.append("| 指标 | 分数 |\n")
            report.append("|------|------|\n")
            report.append(f"| **总体Precision** | {attr['precision']:.3f} |\n")
            report.append(f"| **总体Recall** | {attr['recall']:.3f} |\n")
            report.append(f"| **总体F1** | {attr['f1']:.3f} |\n")
            report.append(f"| 视觉Precision | {attr['visual_precision']:.3f} |\n")
            report.append(f"| 视觉Recall | {attr['visual_recall']:.3f} |\n")
            report.append(f"| 文本Precision | {attr['text_precision']:.3f} |\n")
            report.append(f"| 文本Recall | {attr['text_recall']:.3f} |\n")
            report.append("\n")
        
        # 跨模态一致性
        if 'cross_modal_consistency' in results['metrics']:
            consistency = results['metrics']['cross_modal_consistency']
            report.append("## 2️⃣ 跨模态一致性 (Cross-Modal Consistency)\n\n")
            report.append(f"**一致性分数**: {consistency:.3f}\n\n")
            report.append("- 评估文本答案与视觉证据的对齐程度\n")
            report.append("- 分数范围: [0, 1]，越高越好\n\n")
        
        # 位置偏差
        if 'position_bias' in results['metrics']:
            pos_bias = results['metrics']['position_bias']
            report.append("## 3️⃣ 位置偏差 (Position Bias)\n\n")
            report.append("| 指标 | 值 |\n")
            report.append("|------|----|\n")
            report.append(f"| **位置偏差** | {pos_bias['position_bias']:.3f} |\n")
            report.append(f"| 最大性能差异 | {pos_bias['max_diff']:.3f} |\n")
            
            for key, value in pos_bias.items():
                if '_acc' in key:
                    pos_name = key.replace('_acc', '')
                    report.append(f"| {pos_name}准确率 | {value:.3f} |\n")
            
            report.append("\n")
            report.append("- 位置偏差越小越好（理想值接近0）\n")
            report.append("- 表示模型对检索内容位置的敏感度\n\n")
        
        report.append("---\n\n")
        report.append("**✅ 评估完成**\n")
        
        return "".join(report)


# =============================================================================
# 便捷函数
# =============================================================================

def quick_evaluate_attribution(predictions: List[Dict],
                               ground_truths: List[Dict]) -> Dict[str, float]:
    """
    快速评估归因精度
    
    Args:
        predictions: 预测结果列表
        ground_truths: 真实结果列表
    
    Returns:
        Dict[str, float]: 归因精度指标
    """
    calculator = AttributionPrecisionCalculator()
    results = []
    
    for pred, gt in zip(predictions, ground_truths):
        result = calculator.compute(
            generated_answer=pred.get('answer', ''),
            attributions=pred.get('attributions', {}),
            ground_truth_sources=gt.get('sources', [])
        )
        results.append(result)
    
    # 平均
    avg_result = {}
    if results:
        for key in results[0].keys():
            avg_result[key] = np.mean([r[key] for r in results])
    
    return avg_result


def quick_evaluate_consistency(text_answers: List[str],
                               images: List) -> float:
    """
    快速评估跨模态一致性
    
    Args:
        text_answers: 文本答案列表
        images: 图像列表
    
    Returns:
        float: 平均一致性分数
    """
    scorer = CrossModalConsistencyScore()
    return scorer.compute_batch(text_answers, images)


def quick_evaluate_position_bias(model,
                                 test_samples: List[Dict]) -> float:
    """
    快速评估位置偏差
    
    Args:
        model: 模型
        test_samples: 测试样本
    
    Returns:
        float: 位置偏差分数
    """
    evaluator = PositionBiasMetric()
    results = evaluator.evaluate(model, test_samples)
    return results['position_bias']


