# -*- coding: utf-8 -*-
"""
增强型多模态检索器
扩展FlashRAG的检索器到MRAG 3.0，支持：
1. CLIP视觉检索
2. 跨模态对齐
3. 位置感知融合
4. 多模态结果融合
"""

import torch
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
from PIL import Image
import warnings

try:
    from transformers import CLIPModel, CLIPProcessor
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    warnings.warn("transformers not installed, CLIP功能不可用")


class PositionAwareFusion:
    """
    位置感知融合模块
    实现文档中提到的位置去偏技术
    """
    
    def __init__(self, fusion_method='weighted', position_encoding='learned'):
        """
        Args:
            fusion_method: 融合方法 ['concat', 'weighted', 'rerank']
            position_encoding: 位置编码方式 ['learned', 'sinusoidal', 'none']
        """
        self.fusion_method = fusion_method
        self.position_encoding = position_encoding
        
    def position_weighted_pooling(self, results: List[Dict], scores: List[float], 
                                  positions: Optional[List[int]] = None) -> Tuple[List[Dict], List[float]]:
        """
        位置加权池化
        参考文档：VisRAG的position-weighted mean pooling
        
        Args:
            results: 检索结果列表
            scores: 对应的分数
            positions: 位置索引（可选）
            
        Returns:
            调整后的结果和分数
        """
        if not results:
            return results, scores
        
        n = len(results)
        if positions is None:
            positions = list(range(n))
        
        # 计算位置权重（缓解Lost in the middle问题）
        # 给开头和结尾更高的权重
        position_weights = []
        for i, pos in enumerate(positions):
            # U型权重分布：开头和结尾权重高，中间低
            if pos < n // 3:
                weight = 1.0  # 开头
            elif pos > 2 * n // 3:
                weight = 0.9  # 结尾
            else:
                weight = 0.6  # 中间（Lost in the middle区域）
            position_weights.append(weight)
        
        # 调整分数
        adjusted_scores = [s * w for s, w in zip(scores, position_weights)]
        
        # 根据调整后的分数重新排序
        sorted_items = sorted(zip(results, adjusted_scores, positions), 
                            key=lambda x: x[1], reverse=True)
        
        reordered_results = [item[0] for item in sorted_items]
        reordered_scores = [item[1] for item in sorted_items]
        
        return reordered_results, reordered_scores
    
    def mitigate_position_bias(self, results: List[Dict], scores: List[float], 
                              query: str) -> Tuple[List[Dict], List[float]]:
        """
        缓解位置偏差
        参考文档：位置感知的跨模态融合
        
        策略：
        1. 重要内容放在开头和结尾
        2. 位置编码调整
        """
        if self.fusion_method == 'weighted':
            return self.position_weighted_pooling(results, scores)
        else:
            return results, scores
    
    def fuse(self, text_results: List[Dict], text_scores: List[float],
            visual_results: List[Dict], visual_scores: List[float],
            alpha: float = 0.5) -> Tuple[List[Dict], List[float]]:
        """
        跨模态结果融合
        
        Args:
            text_results: 文本检索结果
            text_scores: 文本检索分数
            visual_results: 视觉检索结果  
            visual_scores: 视觉检索分数
            alpha: 文本权重（1-alpha为视觉权重）
            
        Returns:
            融合后的结果和分数
        """
        # 创建ID到结果的映射
        all_results = {}
        all_scores = {}
        
        # 添加文本结果
        for res, score in zip(text_results, text_scores):
            doc_id = res.get('id', res.get('docid', str(hash(str(res)))))
            if doc_id not in all_results:
                all_results[doc_id] = res
                all_scores[doc_id] = alpha * score
            else:
                all_scores[doc_id] += alpha * score
        
        # 添加视觉结果
        for res, score in zip(visual_results, visual_scores):
            doc_id = res.get('id', res.get('docid', str(hash(str(res)))))
            if doc_id not in all_results:
                all_results[doc_id] = res
                all_scores[doc_id] = (1 - alpha) * score
            else:
                all_scores[doc_id] += (1 - alpha) * score
        
        # 排序
        sorted_items = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        
        fused_results = [all_results[doc_id] for doc_id, _ in sorted_items]
        fused_scores = [score for _, score in sorted_items]
        
        # 应用位置感知调整
        fused_results, fused_scores = self.position_weighted_pooling(
            fused_results, fused_scores
        )
        
        return fused_results, fused_scores


class SelfAwareMultimodalRetriever:
    """
    自感知多模态检索器
    扩展FlashRAG检索器，实现文档中的创新点
    
    复用FlashRAG的：
    - 索引构建
    - 基础检索逻辑
    - 评估框架
    
    新增创新：
    - CLIP视觉检索
    - 跨模态对齐
    - 位置感知融合
    - 自适应模态选择
    """
    
    def __init__(self, config: Dict, text_retriever=None, visual_retriever=None):
        """
        Args:
            config: 配置字典
            text_retriever: FlashRAG的文本检索器（可选）
            visual_retriever: FlashRAG的视觉检索器（可选）
        """
        self.config = config
        self.topk = config.get('retrieval_topk', 5)
        
        # 检索器
        self.text_retriever = text_retriever
        self.visual_retriever = visual_retriever
        
        # CLIP模型（用于跨模态检索）
        self.clip_model = None
        self.clip_processor = None
        if config.get('use_clip', True) and CLIP_AVAILABLE:
            self._load_clip_model(config.get('clip_model_path', '/root/autodl-tmp/models/clip-vit-large-patch14-336'))
        
        # 位置感知融合
        self.position_fusion = PositionAwareFusion(
            fusion_method=config.get('fusion_method', 'weighted'),
            position_encoding=config.get('position_encoding', 'learned')
        )
        
        # 融合权重
        self.text_weight = config.get('text_weight', 0.5)
        self.visual_weight = config.get('visual_weight', 0.5)
    
    def _load_clip_model(self, model_path: str):
        """加载CLIP模型"""
        try:
            self.clip_model = CLIPModel.from_pretrained(model_path)
            self.clip_processor = CLIPProcessor.from_pretrained(model_path)
            
            # 移到GPU（如果可用）
            if torch.cuda.is_available():
                self.clip_model = self.clip_model.cuda()
            
            self.clip_model.eval()
            print(f"✅ CLIP模型加载成功: {model_path}")
        except Exception as e:
            warnings.warn(f"CLIP模型加载失败: {e}")
            self.clip_model = None
            self.clip_processor = None
    
    def retrieve(self, query_text: str, query_image: Optional[Image.Image] = None,
                top_k: Optional[int] = None, return_score: bool = False,
                use_position_fusion: bool = True) -> Union[List[Dict], Tuple[List[Dict], List[float]]]:
        """
        多模态检索主函数
        
        Args:
            query_text: 查询文本
            query_image: 查询图像（可选）
            top_k: 返回结果数量
            return_score: 是否返回分数
            use_position_fusion: 是否使用位置感知融合
            
        Returns:
            检索结果（和分数）
        """
        if top_k is None:
            top_k = self.topk
        
        # 纯文本检索
        if query_image is None:
            if self.text_retriever is None:
                raise ValueError("文本检索器未初始化")
            
            results, scores = self._retrieve_text(query_text, top_k)
            
            if use_position_fusion:
                results, scores = self.position_fusion.mitigate_position_bias(
                    results, scores, query_text
                )
            
            if return_score:
                return results, scores
            else:
                return results
        
        # 多模态检索（文本+图像）
        else:
            return self._retrieve_multimodal(
                query_text, query_image, top_k, return_score, use_position_fusion
            )
    
    def _retrieve_text(self, query_text: str, top_k: int) -> Tuple[List[Dict], List[float]]:
        """
        文本检索（复用FlashRAG）
        """
        if hasattr(self.text_retriever, 'search'):
            results, scores = self.text_retriever.search(query_text, num=top_k, return_score=True)
        else:
            # 兼容不同的接口
            results = self.text_retriever.retrieve(query_text, top_k)
            scores = [1.0] * len(results)  # 默认分数
        
        return results, scores
    
    def _retrieve_by_image(self, query_image: Image.Image, top_k: int) -> Tuple[List[Dict], List[float]]:
        """
        图像检索（使用CLIP或FlashRAG的视觉检索器）
        """
        # 优先使用FlashRAG的视觉检索器
        if self.visual_retriever is not None:
            if hasattr(self.visual_retriever, 'search'):
                results, scores = self.visual_retriever.search(
                    query_image, target_modal='image', num=top_k, return_score=True
                )
            else:
                results = self.visual_retriever.retrieve(query_image, top_k)
                scores = [1.0] * len(results)
            return results, scores
        
        # 降级方案：使用CLIP进行相似度计算（需要预先构建的图像库）
        elif self.clip_model is not None:
            warnings.warn("使用CLIP进行图像检索，性能可能较低。建议使用预构建的FAISS索引。")
            # 这里需要实现基于CLIP的在线检索
            # 实际应用中应该使用预构建的FAISS索引
            return [], []
        
        else:
            raise ValueError("视觉检索器未初始化，且CLIP模型不可用")
    
    def _retrieve_multimodal(self, query_text: str, query_image: Image.Image,
                            top_k: int, return_score: bool,
                            use_position_fusion: bool) -> Union[List[Dict], Tuple[List[Dict], List[float]]]:
        """
        多模态检索（文本+图像融合）
        实现文档中的跨模态检索创新
        """
        # 1. 文本检索
        text_results, text_scores = self._retrieve_text(query_text, top_k * 2)
        
        # 2. 图像检索
        image_results, image_scores = self._retrieve_by_image(query_image, top_k * 2)
        
        # 3. 跨模态融合（我们的创新）
        if len(image_results) > 0:
            fused_results, fused_scores = self.position_fusion.fuse(
                text_results, text_scores,
                image_results, image_scores,
                alpha=self.text_weight
            )
        else:
            # 降级为纯文本检索
            fused_results, fused_scores = text_results, text_scores
        
        # 4. 截取top-k
        fused_results = fused_results[:top_k]
        fused_scores = fused_scores[:top_k]
        
        if return_score:
            return fused_results, fused_scores
        else:
            return fused_results
    
    def batch_retrieve(self, query_texts: List[str], 
                      query_images: Optional[List[Optional[Image.Image]]] = None,
                      top_k: Optional[int] = None,
                      return_score: bool = False) -> Union[List[List[Dict]], Tuple[List[List[Dict]], List[List[float]]]]:
        """
        批量检索
        
        Args:
            query_texts: 查询文本列表
            query_images: 查询图像列表（可选）
            top_k: 每个查询返回的结果数量
            return_score: 是否返回分数
            
        Returns:
            批量检索结果
        """
        if top_k is None:
            top_k = self.topk
        
        if query_images is None:
            query_images = [None] * len(query_texts)
        
        assert len(query_texts) == len(query_images), "文本和图像数量不匹配"
        
        all_results = []
        all_scores = []
        
        for q_text, q_image in zip(query_texts, query_images):
            if return_score:
                results, scores = self.retrieve(q_text, q_image, top_k, return_score=True)
                all_results.append(results)
                all_scores.append(scores)
            else:
                results = self.retrieve(q_text, q_image, top_k, return_score=False)
                all_results.append(results)
        
        if return_score:
            return all_results, all_scores
        else:
            return all_results
    
    def compute_cross_modal_similarity(self, text: str, image: Image.Image) -> float:
        """
        计算跨模态相似度（使用CLIP）
        用于后续的不确定性估计
        
        Args:
            text: 文本
            image: 图像
            
        Returns:
            相似度分数
        """
        if self.clip_model is None:
            warnings.warn("CLIP模型未加载，无法计算跨模态相似度")
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
                # 计算余弦相似度
                logits_per_image = outputs.logits_per_image
                similarity = logits_per_image[0, 0].item()
            
            return similarity
        
        except Exception as e:
            warnings.warn(f"计算跨模态相似度失败: {e}")
            return 0.0


# 向后兼容的接口
class MultimodalCLIPRetriever(SelfAwareMultimodalRetriever):
    """
    MultimodalCLIPRetriever的别名
    为了向后兼容
    """
    pass