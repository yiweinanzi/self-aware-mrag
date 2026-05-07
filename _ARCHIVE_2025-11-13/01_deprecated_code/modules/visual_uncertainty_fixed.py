#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visual Uncertainty - 修复版

问题：原版可能总是抛出异常，返回默认值0.5
解决：使用CLIP的视觉特征方差作为不确定性指标

参考：
- 文档第821-827行
- 简化但可靠的实现
"""

import warnings
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except:
    TORCH_AVAILABLE = False
    warnings.warn("torch未安装")

try:
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    CLIP_AVAILABLE = True
except:
    CLIP_AVAILABLE = False
    warnings.warn("CLIP未安装")


class VisualUncertaintyEstimatorFixed:
    """
    修复版的视觉不确定性估计器
    
    使用CLIP特征的方差作为不确定性指标
    
    原理：
    - 如果图像特征分布集中 → 低不确定性（模型有把握）
    - 如果图像特征分布分散 → 高不确定性（模型不确定）
    
    使用示例：
    ```python
    estimator = VisualUncertaintyEstimatorFixed(
        clip_model_path='/path/to/clip'
    )
    
    uncertainty = estimator.estimate(image)
    ```
    """
    
    def __init__(self, clip_model_path: str = None):
        """
        初始化
        
        Args:
            clip_model_path: CLIP模型路径
        """
        self.clip_model = None
        self.clip_processor = None
        
        if CLIP_AVAILABLE and clip_model_path:
            self._load_clip(clip_model_path)
    
    def _load_clip(self, model_path: str):
        """加载CLIP模型"""
        try:
            self.clip_model = CLIPModel.from_pretrained(
                model_path, local_files_only=True
            )
            self.clip_processor = CLIPProcessor.from_pretrained(
                model_path, local_files_only=True
            )
            
            if torch.cuda.is_available():
                self.clip_model = self.clip_model.cuda()
            
            self.clip_model.eval()
            
            print(f"✅ CLIP模型加载成功（用于Visual Uncertainty）")
        except Exception as e:
            warnings.warn(f"CLIP加载失败: {e}")
    
    def estimate(self, image, return_details: bool = False) -> float:
        """
        估计视觉不确定性（修复版）
        
        使用CLIP特征的标准差作为不确定性指标
        
        Args:
            image: 图像（PIL.Image或路径）
            return_details: 是否返回详细信息
            
        Returns:
            float: 视觉不确定性 [0, 1]
        """
        if not CLIP_AVAILABLE or self.clip_model is None:
            # 降级：返回固定的低不确定性
            # OK-VQA主要靠文本知识，视觉不确定性不重要
            return 0.1
        
        try:
            # 处理图像
            if not isinstance(image, Image.Image):
                if hasattr(image, 'convert'):
                    image = image.convert('RGB')
                else:
                    return 0.1
            
            # CLIP编码
            inputs = self.clip_processor(
                images=image,
                return_tensors="pt"
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                # 获取视觉特征
                vision_outputs = self.clip_model.vision_model(**inputs)
                image_features = vision_outputs.last_hidden_state  # [1, seq_len, dim]
                
                # 计算特征分布的标准差
                # 标准差越大 → 特征越分散 → 不确定性越高
                feature_std = torch.std(image_features, dim=1).mean().item()
                
                # 归一化到[0, 1]
                # 经验值：std通常在0-2范围内
                normalized_uncertainty = min(1.0, feature_std / 2.0)
                
                if return_details:
                    return {
                        'uncertainty': normalized_uncertainty,
                        'feature_std': feature_std,
                        'method': 'CLIP feature std'
                    }
                
                return normalized_uncertainty
        
        except Exception as e:
            warnings.warn(f"视觉不确定性计算失败: {e}")
            # 降级：返回低不确定性（OK-VQA主要靠文本）
            return 0.1


if __name__ == '__main__':
    print("Visual Uncertainty Estimator - 修复版")
    print("=" * 70)
    print("修复内容：")
    print("  ❌ 原版：_get_visual_attention可能总是失败")
    print("  ✅ 新版：使用CLIP特征标准差")
    print("\n方法：")
    print("  1. 使用CLIP编码图像")
    print("  2. 计算特征分布的标准差")
    print("  3. 标准差越大 → 不确定性越高")
    print("\n优势：")
    print("  - 简单可靠，不易出错")
    print("  - 基于成熟的CLIP模型")
    print("  - 适合OK-VQA场景")
    print("=" * 70)


