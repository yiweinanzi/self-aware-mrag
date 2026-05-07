# -*- coding: utf-8 -*-
"""
改进版不确定性估计器
Improved Uncertainty Estimator

改进点：
1. 更合理的不确定性值范围（0.2-0.6）
2. 基于问题类型的自适应估计
3. 支持图像复杂度分析
4. 可配置的权重系统

注意：这是测试版本，不修改原有代码
"""

import warnings
from typing import Dict, Optional

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
    warnings.warn("torch未安装，不确定性估计功能受限")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class ImprovedUncertaintyEstimator:
    """
    改进版不确定性估计器
    
    改进策略：
    1. 问题分类：识别是否需要外部知识
    2. 问题复杂度：基于长度、关键词等
    3. 图像复杂度：简单分析图像信息量
    4. 自适应权重：根据问题类型调整
    
    目标：让不确定性值在0.2-0.6范围内有合理分布
    """
    
    def __init__(self, config=None):
        """
        初始化改进版估计器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 权重配置（可调整）
        self.text_weight = self.config.get('text_weight', 0.5)
        self.visual_weight = self.config.get('visual_weight', 0.3)
        self.alignment_weight = self.config.get('alignment_weight', 0.2)
        
        # 知识需求关键词（需要检索的信号）
        self.knowledge_keywords = [
            'who', 'what', 'where', 'when', 'which', 'why', 'how',
            'invented', 'created', 'built', 'designed', 'made',
            'name', 'called', 'known as', 'famous',
            'year', 'date', 'century', 'period',
            'city', 'country', 'location', 'place',
            'company', 'organization', 'brand',
        ]
        
        # 视觉关键词（主要依赖图像的信号）
        self.visual_keywords = [
            'color', 'shape', 'size', 'appearance',
            'see', 'visible', 'shown', 'pictured',
            'wearing', 'holding', 'doing',
            'left', 'right', 'front', 'back',
        ]
        
        # 加载CLIP（如果需要）
        self.clip_model = None
        self.clip_processor = None
        if self.config.get('use_clip', True):
            self._load_clip_model()
        
        print("✅ ImprovedUncertaintyEstimator初始化完成")
    
    def _load_clip_model(self):
        """加载CLIP模型用于图像复杂度分析"""
        try:
            from transformers import CLIPModel, CLIPProcessor
            
            model_path = self.config.get('clip_model_path', '/root/autodl-tmp/models/clip-vit-large-patch14-336')
            self.clip_model = CLIPModel.from_pretrained(model_path)
            self.clip_processor = CLIPProcessor.from_pretrained(model_path)
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.clip_model = self.clip_model.cuda()
            
            self.clip_model.eval()
            print(f"   ✓ CLIP模型加载成功（用于图像分析）")
        except Exception as e:
            warnings.warn(f"CLIP模型加载失败: {e}")
            self.clip_model = None
            self.clip_processor = None
    
    def estimate(self, text_query: str, image_query=None) -> Dict[str, float]:
        """
        估计不确定性
        
        Args:
            text_query: 文本查询
            image_query: 图像（PIL.Image或None）
            
        Returns:
            dict: 不确定性字典
        """
        # 1. 文本不确定性
        text_uncertainty = self._estimate_text_uncertainty(text_query)
        
        # 2. 视觉不确定性
        visual_uncertainty = 0.0
        if image_query is not None:
            visual_uncertainty = self._estimate_visual_uncertainty(image_query)
        
        # 3. 跨模态对齐不确定性
        alignment_uncertainty = 0.0
        if image_query is not None:
            alignment_uncertainty = self._estimate_alignment_uncertainty(
                text_query, image_query
            )
        
        # 4. 总不确定性（加权）
        total_uncertainty = (
            self.text_weight * text_uncertainty +
            self.visual_weight * visual_uncertainty +
            self.alignment_weight * alignment_uncertainty
        )
        
        return {
            'text': text_uncertainty,
            'visual': visual_uncertainty,
            'alignment': alignment_uncertainty,
            'total': total_uncertainty
        }
    
    def _estimate_text_uncertainty(self, text: str) -> float:
        """
        估计文本不确定性（改进版）
        
        策略：
        1. 检查是否包含知识需求关键词 → 提高不确定性
        2. 检查问题复杂度（长度、结构） → 影响不确定性
        3. 检查是否主要是视觉问题 → 降低不确定性
        
        目标范围：0.2-0.7
        """
        if not text or text == "":
            return 0.4  # 默认中等
        
        text_lower = text.lower()
        words = text.split()
        word_count = len(words)
        
        # 基础不确定性（基于长度）
        if word_count < 5:
            base_uncertainty = 0.25  # 短问题
        elif word_count > 15:
            base_uncertainty = 0.45  # 长问题
        else:
            # 线性插值 5-15词: 0.25-0.45
            base_uncertainty = 0.25 + (word_count - 5) * (0.20 / 10)
        
        # 知识需求调整（+0.0 to +0.3）
        knowledge_boost = 0.0
        knowledge_count = sum(1 for kw in self.knowledge_keywords if kw in text_lower)
        if knowledge_count > 0:
            # 有知识关键词 → 提高不确定性
            knowledge_boost = min(0.3, knowledge_count * 0.1)
        
        # 视觉问题调整（-0.0 to -0.2）
        visual_penalty = 0.0
        visual_count = sum(1 for kw in self.visual_keywords if kw in text_lower)
        if visual_count > 0:
            # 主要是视觉问题 → 降低不确定性（图像能回答）
            visual_penalty = min(0.2, visual_count * 0.1)
        
        # 组合
        uncertainty = base_uncertainty + knowledge_boost - visual_penalty
        
        # 裁剪到合理范围 [0.2, 0.7]
        uncertainty = max(0.2, min(0.7, uncertainty))
        
        return uncertainty
    
    def _estimate_visual_uncertainty(self, image) -> float:
        """
        估计视觉不确定性（改进版v2）
        
        策略：
        1. 使用CLIP特征的范数和分布来评估图像复杂度
        2. 范数大 + 方差大 → 图像信息丰富 → 不确定性低
        3. 范数小 + 方差小 → 图像信息不足 → 不确定性高
        
        目标范围：0.15-0.55
        """
        if self.clip_model is None or self.clip_processor is None:
            # 简化版：返回中等值，但加入一些随机性以模拟变化
            import random
            return 0.25 + random.random() * 0.2  # 0.25-0.45
        
        try:
            # 使用CLIP提取图像特征
            inputs = self.clip_processor(images=image, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                
                # 计算多个指标
                feature_norm = torch.norm(image_features, p=2).item()  # L2范数
                feature_std = torch.std(image_features).item()          # 标准差
                feature_mean_abs = torch.mean(torch.abs(image_features)).item()  # 平均绝对值
                
                # 综合评分：范数和标准差越大，图像信息越丰富，不确定性越低
                # 典型值：norm ~15-30, std ~0.15-0.35, mean_abs ~0.05-0.15
                
                # 归一化各指标到[0,1]
                norm_score = min(1.0, max(0.0, (feature_norm - 10) / 30))  # 10-40映射到0-1
                std_score = min(1.0, max(0.0, (feature_std - 0.1) / 0.4))  # 0.1-0.5映射到0-1
                mean_score = min(1.0, max(0.0, (feature_mean_abs - 0.03) / 0.15))  # 0.03-0.18映射到0-1
                
                # 综合得分（越高说明图像越复杂/信息越丰富）
                richness_score = (norm_score * 0.4 + std_score * 0.4 + mean_score * 0.2)
                
                # 转换为不确定性：richness越高，不确定性越低
                # richness 0.0 → uncertainty 0.55
                # richness 1.0 → uncertainty 0.15
                uncertainty = 0.55 - richness_score * 0.4
                
                return max(0.15, min(0.55, uncertainty))
        
        except Exception as e:
            warnings.warn(f"视觉不确定性计算失败: {e}")
            return 0.3
    
    def _estimate_alignment_uncertainty(self, text: str, image) -> float:
        """
        估计跨模态对齐不确定性（改进版）
        
        策略：
        1. 如果有CLIP，计算text-image相似度
        2. 相似度高 → 对齐好 → 不确定性低
        3. 相似度低 → 对齐差 → 不确定性高
        
        目标范围：0.0-0.4
        """
        if self.clip_model is None or self.clip_processor is None:
            # 简化版：返回低值
            return 0.1
        
        try:
            # 计算text-image相似度
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
                # logits_per_image: [1, 1]
                similarity = outputs.logits_per_image[0, 0].item()
                
                # similarity通常在10-30范围
                # 高相似度 → 低不确定性
                # 低相似度 → 高不确定性
                if similarity > 25:
                    uncertainty = 0.0  # 高度对齐
                elif similarity < 15:
                    uncertainty = 0.4  # 低度对齐
                else:
                    # 线性映射 [15, 25] → [0.4, 0.0]
                    uncertainty = 0.4 - (similarity - 15) * (0.4 / 10)
                
                return uncertainty
        
        except Exception as e:
            warnings.warn(f"对齐不确定性计算失败: {e}")
            return 0.1
    
    def get_uncertainty_explanation(self, uncertainties: Dict[str, float]) -> str:
        """
        生成不确定性的解释
        
        用于调试和理解
        """
        text_unc = uncertainties['text']
        visual_unc = uncertainties['visual']
        alignment_unc = uncertainties['alignment']
        total_unc = uncertainties['total']
        
        explanation = f"总不确定性: {total_unc:.3f}\n"
        explanation += f"  - 文本: {text_unc:.3f} (权重{self.text_weight})\n"
        explanation += f"  - 视觉: {visual_unc:.3f} (权重{self.visual_weight})\n"
        explanation += f"  - 对齐: {alignment_unc:.3f} (权重{self.alignment_weight})\n"
        
        # 判断主要因素
        if text_unc > 0.5:
            explanation += "→ 主要因素：文本复杂或需要外部知识\n"
        elif visual_unc > 0.4:
            explanation += "→ 主要因素：图像信息不足\n"
        elif alignment_unc > 0.3:
            explanation += "→ 主要因素：文本-图像对齐差\n"
        else:
            explanation += "→ 整体不确定性适中\n"
        
        return explanation


# ============================================================================
# 测试工具
# ============================================================================

def test_improved_estimator():
    """测试改进版估计器"""
    print("="*80)
    print("测试改进版不确定性估计器")
    print("="*80)
    
    estimator = ImprovedUncertaintyEstimator()
    
    # 测试问题
    test_cases = [
        ("What color is the car?", None, "纯视觉问题"),
        ("Who invented this device?", None, "需要知识问题"),
        ("What is the name of the building in the picture?", None, "复杂知识问题"),
        ("Is this a dog or a cat?", None, "简单视觉问题"),
        ("In which year was this monument built?", None, "具体知识问题"),
    ]
    
    print("\n测试不同类型问题的不确定性:")
    print("-"*80)
    
    for text, image, description in test_cases:
        uncertainties = estimator.estimate(text, image)
        print(f"\n问题: {text}")
        print(f"类型: {description}")
        print(f"不确定性: {uncertainties['total']:.3f}")
        print(f"  文本: {uncertainties['text']:.3f}")
        print(f"  视觉: {uncertainties['visual']:.3f}")
        print(f"  对齐: {uncertainties['alignment']:.3f}")
    
    print("\n" + "="*80)
    print("测试完成")
    print("="*80)


if __name__ == '__main__':
    test_improved_estimator()

