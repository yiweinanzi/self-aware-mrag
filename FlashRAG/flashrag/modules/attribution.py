# -*- coding: utf-8 -*-
"""
细粒度多模态证据归因模块
Fine-Grained Multimodal Attribution

实现文档中的核心创新点3：
- Region-level视觉归因
- Token-level文本归因
- Attribution confidence计算

参考文档：创新点1-自感知多模态RAG-实施方案.md 第929-979行
参考论文：VISA (Ma et al., 2024b), OMG-QA (Nan et al., 2024)
"""

import warnings
from typing import List, Dict, Any, Tuple, Optional

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("torch未安装，归因功能受限")

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    warnings.warn("pytorch-grad-cam未安装，视觉归因功能不可用")

try:
    from PIL import Image
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class FineGrainedMultimodalAttribution:
    """
    细粒度多模态归因模块
    
    功能：
    1. Region-level视觉归因（使用Grad-CAM）
    2. Token-level文本归因（基于attention）
    3. Attribution confidence计算
    
    使用示例：
    ```python
    attributor = FineGrainedMultimodalAttribution(mllm_model)
    
    # 视觉归因
    visual_attr = attributor.attribute_visual_evidence(
        image=retrieved_image,
        generated_text="A cat on a chair",
        retrieved_images=[img1, img2, img3]
    )
    
    # 文本归因
    text_attr = attributor.attribute_text_evidence(
        generated_text="The capital is Paris",
        retrieved_texts=["France info...", "Paris info..."]
    )
    ```
    """
    
    def __init__(self, mllm_model=None, config=None):
        """
        初始化归因模块
        
        Args:
            mllm_model: 多模态大模型（如LLaVA）
            config: 配置字典
        """
        self.mllm_model = mllm_model
        self.config = config or {}
        
        # Grad-CAM配置
        self.grad_cam_model = None
        self.confidence_threshold = self.config.get('attribution_confidence_threshold', 0.7)
        self.region_threshold = self.config.get('region_activation_threshold', 0.7)
        
        if mllm_model is not None and GRADCAM_AVAILABLE:
            self._init_grad_cam()
    
    def _init_grad_cam(self):
        """初始化Grad-CAM模型"""
        try:
            # 获取目标层（通常是视觉编码器的最后一层）
            # 这里需要根据具体的MLLM结构调整
            if hasattr(self.mllm_model, 'visual_encoder'):
                target_layers = [self.mllm_model.visual_encoder.layer4]
            elif hasattr(self.mllm_model, 'vision_tower'):
                # LLaVA结构
                target_layers = [self.mllm_model.vision_tower.vision_model.encoder.layers[-1]]
            else:
                warnings.warn("无法找到视觉编码器层，Grad-CAM不可用")
                return
            
            self.grad_cam_model = GradCAM(
                model=self.mllm_model,
                target_layers=target_layers
            )
            print("✅ Grad-CAM模型初始化成功")
        
        except Exception as e:
            warnings.warn(f"Grad-CAM初始化失败: {e}")
            self.grad_cam_model = None
    
    def attribute_visual_evidence(self, image, generated_text: str,
                                  retrieved_images: List,
                                  return_visualization: bool = False) -> List[Dict]:
        """
        Region-level视觉归因
        
        参考：VISA (Ma et al., 2024b)
        创新：Region-level而非image-level
        
        Args:
            image: 查询图像或生成时关注的图像
            generated_text: 生成的文本答案
            retrieved_images: 检索到的图像列表
            return_visualization: 是否返回可视化
            
        Returns:
            List[Dict]: 归因结果列表
                [
                    {
                        'region_bbox': [x, y, w, h],
                        'confidence': 0.85,
                        'source_image': image_obj,
                        'source_image_id': 'img_123'
                    },
                    ...
                ]
        """
        if self.grad_cam_model is None or not GRADCAM_AVAILABLE:
            warnings.warn("Grad-CAM不可用，使用简化归因")
            return self._attribute_visual_simplified(retrieved_images)
        
        try:
            attributions = []
            
            # 1. 使用Grad-CAM生成attention map
            attention_maps = self._generate_attention_maps(image, generated_text)
            
            # 2. 提取高激活区域
            regions = self._extract_high_activation_regions(
                attention_maps, threshold=self.region_threshold
            )
            
            # 3. 为每个区域计算归因
            for region in regions:
                # 匹配到检索的图像
                source_image, source_id = self._match_region_to_retrieved(
                    region, retrieved_images
                )
                
                # 计算归因置信度
                confidence = self._compute_region_attribution_confidence(region)
                
                if confidence >= self.confidence_threshold:
                    attributions.append({
                        'region_bbox': region['bbox'],
                        'confidence': confidence,
                        'source_image': source_image,
                        'source_image_id': source_id,
                        'activation_score': region['score']
                    })
            
            # 按置信度排序
            attributions.sort(key=lambda x: x['confidence'], reverse=True)
            
            return attributions
        
        except Exception as e:
            warnings.warn(f"视觉归因失败: {e}")
            return self._attribute_visual_simplified(retrieved_images)
    
    def attribute_text_evidence(self, generated_text: str,
                               retrieved_texts: List[str]) -> List[Dict]:
        """
        Token-level文本归因
        
        参考：OMG-QA (Nan et al., 2024)
        创新：Token-level精确归因
        
        Args:
            generated_text: 生成的文本答案
            retrieved_texts: 检索到的文本列表
            
        Returns:
            List[Dict]: Token级别的归因
                [
                    {
                        'token': 'Paris',
                        'source_span': 'capital of France is Paris',
                        'source_text_id': 'doc_1',
                        'confidence': 0.92,
                        'position': 3
                    },
                    ...
                ]
        """
        if not TORCH_AVAILABLE:
            warnings.warn("torch不可用，使用简化文本归因")
            return self._attribute_text_simplified(generated_text, retrieved_texts)
        
        try:
            attributions = []
            
            # 分词
            tokens = generated_text.split()
            
            for position, token in enumerate(tokens):
                # 找到最相关的源文本片段
                source_span, source_id = self._find_source_span(
                    token, retrieved_texts
                )
                
                # 计算归因置信度
                confidence = self._compute_token_attribution_confidence(
                    token, source_span
                )
                
                if confidence >= self.confidence_threshold:
                    attributions.append({
                        'token': token,
                        'source_span': source_span,
                        'source_text_id': source_id,
                        'confidence': confidence,
                        'position': position
                    })
            
            return attributions
        
        except Exception as e:
            warnings.warn(f"文本归因失败: {e}")
            return self._attribute_text_simplified(generated_text, retrieved_texts)
    
    # =========================================================================
    # Grad-CAM相关方法
    # =========================================================================
    
    def _generate_attention_maps(self, image, text: str) -> np.ndarray:
        """
        使用Grad-CAM生成attention map
        
        Args:
            image: PIL.Image
            text: 文本
            
        Returns:
            attention_map: [H, W] numpy数组
        """
        if self.grad_cam_model is None:
            raise ValueError("Grad-CAM模型未初始化")
        
        # 准备输入
        # 这里需要根据具体MLLM的输入格式调整
        input_tensor = self._prepare_image_tensor(image)
        
        # 生成CAM
        grayscale_cam = self.grad_cam_model(
            input_tensor=input_tensor,
            targets=None  # 可以指定特定的类别
        )
        
        # 返回第一张图的CAM
        attention_map = grayscale_cam[0]
        
        return attention_map
    
    def _extract_high_activation_regions(self, attention_map: np.ndarray,
                                        threshold: float = 0.7) -> List[Dict]:
        """
        从attention map中提取高激活区域
        
        Args:
            attention_map: [H, W] attention map
            threshold: 激活阈值
            
        Returns:
            List[Dict]: 区域列表
                [
                    {
                        'bbox': [x, y, w, h],
                        'score': 0.85,
                        'mask': binary_mask
                    },
                    ...
                ]
        """
        if not NUMPY_AVAILABLE:
            return []
        
        # 二值化
        binary_mask = (attention_map > threshold).astype(np.uint8)
        
        # 连通域分析
        from scipy import ndimage
        labeled_mask, num_regions = ndimage.label(binary_mask)
        
        regions = []
        for region_id in range(1, num_regions + 1):
            # 提取该区域的mask
            region_mask = (labeled_mask == region_id)
            
            # 计算bounding box
            coords = np.argwhere(region_mask)
            if len(coords) == 0:
                continue
            
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # 计算该区域的平均激活分数
            region_score = attention_map[region_mask].mean()
            
            regions.append({
                'bbox': [int(x_min), int(y_min), 
                        int(x_max - x_min), int(y_max - y_min)],
                'score': float(region_score),
                'mask': region_mask
            })
        
        # 按分数排序
        regions.sort(key=lambda x: x['score'], reverse=True)
        
        return regions
    
    def _match_region_to_retrieved(self, region: Dict, 
                                   retrieved_images: List) -> Tuple[Any, str]:
        """
        将区域匹配到检索的图像
        
        简化版：返回第一张检索图像
        完整版应该：计算region与每张图像的相似度
        """
        if retrieved_images and len(retrieved_images) > 0:
            return retrieved_images[0], f"img_{0}"
        else:
            return None, "unknown"
    
    def _compute_region_attribution_confidence(self, region: Dict) -> float:
        """
        计算区域归因的置信度
        
        基于：
        - 激活分数
        - 区域大小
        - 位置重要性
        """
        activation_score = region['score']
        
        # 简化版：直接使用激活分数
        confidence = activation_score
        
        return confidence
    
    # =========================================================================
    # 文本归因相关方法
    # =========================================================================
    
    def _find_source_span(self, token: str, retrieved_texts: List[str]) -> Tuple[str, str]:
        """
        找到token对应的源文本片段
        
        使用简单的字符串匹配
        完整版应该：使用语义相似度
        """
        for idx, text in enumerate(retrieved_texts):
            if token.lower() in text.lower():
                # 提取包含token的片段（前后各10个词）
                words = text.split()
                token_positions = [
                    i for i, w in enumerate(words) 
                    if token.lower() in w.lower()
                ]
                
                if token_positions:
                    pos = token_positions[0]
                    start = max(0, pos - 10)
                    end = min(len(words), pos + 11)
                    span = ' '.join(words[start:end])
                    return span, f"doc_{idx}"
        
        return "", "unknown"
    
    def _compute_token_attribution_confidence(self, token: str, 
                                             source_span: str) -> float:
        """
        计算token归因的置信度
        
        基于：
        - 精确匹配程度
        - 上下文相关性
        """
        if source_span == "":
            return 0.0
        
        # 简化版：基于字符串包含关系
        if token.lower() in source_span.lower():
            return 0.9
        else:
            return 0.1
    
    # =========================================================================
    # 简化版归因方法
    # =========================================================================
    
    def _attribute_visual_simplified(self, retrieved_images: List) -> List[Dict]:
        """
        简化版视觉归因
        
        不使用Grad-CAM，直接归因到整张图像
        """
        attributions = []
        for idx, img in enumerate(retrieved_images[:3]):  # 只取前3张
            attributions.append({
                'region_bbox': None,  # 整张图像
                'confidence': 1.0 - idx * 0.2,  # 递减置信度
                'source_image': img,
                'source_image_id': f'img_{idx}',
                'activation_score': 1.0 - idx * 0.2
            })
        
        return attributions
    
    def _attribute_text_simplified(self, generated_text: str,
                                   retrieved_texts: List[str]) -> List[Dict]:
        """
        简化版文本归因
        
        基于字符串匹配
        """
        attributions = []
        tokens = generated_text.split()
        
        for position, token in enumerate(tokens):
            # 简单匹配
            for idx, text in enumerate(retrieved_texts):
                if token.lower() in text.lower():
                    attributions.append({
                        'token': token,
                        'source_span': token,  # 简化：只返回token本身
                        'source_text_id': f'doc_{idx}',
                        'confidence': 0.8,
                        'position': position
                    })
                    break
        
        return attributions
    
    def _prepare_image_tensor(self, image) -> torch.Tensor:
        """
        准备图像tensor用于Grad-CAM
        
        Args:
            image: PIL.Image
            
        Returns:
            torch.Tensor: [1, 3, H, W]
        """
        if not NUMPY_AVAILABLE:
            raise ValueError("numpy不可用")
        
        # 转换为numpy数组
        img_array = np.array(image.resize((224, 224)))
        
        # 归一化
        img_array = img_array / 255.0
        
        # 转换为tensor: [H, W, C] -> [C, H, W]
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        
        # 添加batch维度: [C, H, W] -> [1, C, H, W]
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor


class AttributionVisualizer:
    """
    归因结果可视化工具
    
    功能：
    - 在图像上绘制bounding box
    - 生成热力图
    - 高亮文本片段
    """
    
    @staticmethod
    def visualize_visual_attribution(image, attributions: List[Dict],
                                     save_path: Optional[str] = None):
        """
        可视化视觉归因
        
        在图像上绘制bounding box和confidence
        """
        try:
            from PIL import ImageDraw, ImageFont
            
            # 复制图像
            img_with_boxes = image.copy()
            draw = ImageDraw.Draw(img_with_boxes)
            
            # 绘制每个归因区域
            for attr in attributions:
                if attr['region_bbox'] is None:
                    continue
                
                x, y, w, h = attr['region_bbox']
                confidence = attr['confidence']
                
                # 根据置信度设置颜色
                if confidence > 0.8:
                    color = 'green'
                elif confidence > 0.6:
                    color = 'yellow'
                else:
                    color = 'orange'
                
                # 绘制矩形框
                draw.rectangle(
                    [x, y, x+w, y+h],
                    outline=color,
                    width=3
                )
                
                # 添加置信度文本
                draw.text(
                    (x, y-15),
                    f"{confidence:.2f}",
                    fill=color
                )
            
            if save_path:
                img_with_boxes.save(save_path)
            
            return img_with_boxes
        
        except Exception as e:
            warnings.warn(f"可视化失败: {e}")
            return image
    
    @staticmethod
    def visualize_text_attribution(generated_text: str, attributions: List[Dict]) -> str:
        """
        可视化文本归因
        
        高亮归因的token
        
        Returns:
            str: HTML格式的文本（带高亮）
        """
        tokens = generated_text.split()
        attributed_tokens = {attr['position']: attr for attr in attributions}
        
        html_parts = []
        for pos, token in enumerate(tokens):
            if pos in attributed_tokens:
                attr = attributed_tokens[pos]
                confidence = attr['confidence']
                
                # 根据置信度设置颜色
                if confidence > 0.8:
                    color = 'lightgreen'
                elif confidence > 0.6:
                    color = 'lightyellow'
                else:
                    color = 'lightcoral'
                
                source_id = attr.get("source_text_id", "unknown")
                html_parts.append(
                    f'<span style="background-color:{color}" '
                    f'title="Source: {source_id}, '
                    f'Confidence: {confidence:.2f}">{token}</span>'
                )
            else:
                html_parts.append(token)
        
        return ' '.join(html_parts)


# 工厂函数
def create_attribution_module(mllm_model=None, config=None):
    """
    创建归因模块
    
    Args:
        mllm_model: MLLM模型（可选）
        config: 配置（可选）
        
    Returns:
        FineGrainedMultimodalAttribution实例
    """
    return FineGrainedMultimodalAttribution(mllm_model, config)

