#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VisRAG Baseline - 简化版

基于论文: VisRAG: Vision-based Retrieval-Augmented Generation  
参考: arXiv:2410.10594

核心思想：
1. 直接将文档作为图像嵌入（而不是先解析为文本）
2. 使用Vision-Language Model进行检索
3. Position-weighted pooling for VLM hidden states
4. 检索到的文档图像直接用于VLM生成

简化说明：
- 使用CLIP进行图像检索（代替训练VisRAG-Ret）
- 使用LLaVA作为生成器（VisRAG-Gen）
- 重点实现：position-weighted pooling思想
- 不需要重新训练模型

对比价值：
- 展示视觉RAG vs 文本RAG的区别
- position-aware处理（vs 简单拼接）
- 与我们的方法对比位置处理策略
"""

import torch
import warnings
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

class VisRAGBaseline:
    """
    VisRAG Baseline (简化版)
    
    核心流程：
    1. 图像检索：使用CLIP检索相关文档（作为图像）
    2. Position-weighted pooling：对检索结果应用位置权重
    3. VLM生成：使用LLaVA基于多图像上下文生成答案
    
    使用示例：
    ```python
    baseline = VisRAGBaseline(
        llava_wrapper=llava,
        clip_retriever=clip_retriever
    )
    results = baseline.run(dataset)
    ```
    """
    
    def __init__(self, llava_wrapper, retriever=None, config=None):
        """
        初始化VisRAG baseline
        
        Args:
            llava_wrapper: LLaVA模型封装器
            retriever: CLIP检索器
            config: 配置字典
        """
        self.llava = llava_wrapper
        self.retriever = retriever
        self.config = config or {}
        
        # 配置
        self.top_k = self.config.get('retrieval_topk', 5)
        self.max_new_tokens = self.config.get('max_new_tokens', 100)
        self.temperature = self.config.get('temperature', 0.2)
        
        # VisRAG特定配置
        self.use_position_weighting = self.config.get('use_position_weighting', True)
        self.position_weight_decay = self.config.get('position_weight_decay', 0.5)
        
        # 总是检索（与MuRAG相同）
        self.always_retrieve = True
    
    # =========================================================================
    # 核心创新：Position-Weighted Pooling
    # =========================================================================
    
    def compute_position_weights(self, k: int, decay: float = None) -> np.ndarray:
        """
        计算位置权重
        
        VisRAG的核心思想：后面的检索结果应该得到更多关注
        （与"Lost in the Middle"观察一致）
        
        Args:
            k: 检索结果数量
            decay: 衰减因子（默认使用配置值）
            
        Returns:
            np.ndarray: 位置权重 [k]
        """
        if decay is None:
            decay = self.position_weight_decay
        
        # 策略1: 线性递增（后面位置权重更大）
        if decay == 0.0:
            weights = np.arange(1, k+1, dtype=float)  # [1, 2, 3, ..., k]
        
        # 策略2: 指数递增
        elif decay > 0:
            weights = np.exp(np.arange(k) * decay)
        
        # 策略3: 均匀权重（用于消融）
        else:
            weights = np.ones(k, dtype=float)
        
        # 归一化
        weights = weights / weights.sum()
        
        return weights
    
    def apply_position_weighted_fusion(self, retrieved_docs: List[str],
                                      retrieval_scores: List[float]) -> str:
        """
        应用位置加权融合
        
        Args:
            retrieved_docs: 检索到的文档列表
            retrieval_scores: 检索分数
            
        Returns:
            str: 融合后的上下文
        """
        if not retrieved_docs:
            return ""
        
        k = len(retrieved_docs)
        
        if not self.use_position_weighting:
            # 不使用位置权重，简单拼接（MuRAG风格）
            context = "\n\n".join([
                f"[Document {i+1}]\n{doc}"
                for i, doc in enumerate(retrieved_docs)
            ])
            return context
        
        # 计算位置权重
        position_weights = self.compute_position_weights(k)
        
        # 计算综合权重（检索分数 × 位置权重）
        retrieval_scores_norm = np.array(retrieval_scores) / (np.sum(retrieval_scores) + 1e-10)
        combined_weights = retrieval_scores_norm * position_weights
        combined_weights = combined_weights / (combined_weights.sum() + 1e-10)
        
        # 按综合权重排序（权重大的放在更显著的位置）
        sorted_indices = np.argsort(combined_weights)[::-1]
        
        # 构建上下文（重要的文档放在开头和结尾）
        # 参考"Lost in the Middle"：开头和结尾的内容更容易被注意
        context_parts = []
        
        for rank, idx in enumerate(sorted_indices):
            doc = retrieved_docs[idx]
            weight = combined_weights[idx]
            
            # 添加权重提示（帮助模型理解重要性）
            importance = "HIGH" if weight > 0.25 else "MEDIUM" if weight > 0.15 else "LOW"
            context_parts.append(
                f"[Document {rank+1}] (Importance: {importance})\n{doc}"
            )
        
        context = "\n\n".join(context_parts)
        
        return context
    
    # =========================================================================
    # 检索和生成
    # =========================================================================
    
    def retrieve(self, question: str, image=None, top_k: int = None) -> Tuple[List[str], List[float]]:
        """
        使用CLIP检索相关文档
        
        在完整的VisRAG中，会使用VisRAG-Ret模型检索文档图像
        在简化版中，我们使用文本检索器
        
        Args:
            question: 问题
            image: 查询图像
            top_k: 检索数量
            
        Returns:
            (documents, scores)
        """
        if top_k is None:
            top_k = self.top_k
        
        if self.retriever is None:
            warnings.warn("未提供检索器")
            return [], []
        
        try:
            result = self.retriever.retrieve(
                query_text=question,
                query_image=image,
                top_k=top_k
            )
            
            if isinstance(result, tuple):
                docs, scores = result
                return docs, scores
            else:
                return result, [1.0] * len(result)
        
        except Exception as e:
            warnings.warn(f"检索失败: {e}")
            return [], []
    
    def generate_answer(self, question: str, context: str, 
                       image=None) -> str:
        """
        使用LLaVA生成答案
        
        Args:
            question: 问题
            context: 位置加权后的上下文
            image: 查询图像
            
        Returns:
            str: 答案
        """
        # 构建prompt
        if context:
            prompt = f"""Based on the following retrieved information, answer the question concisely.

{context}

Question: {question}

Answer:"""
        else:
            prompt = f"""Question: {question}

Answer:"""
        
        try:
            answer = self.llava.generate(
                text=prompt,
                image=image,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature
            )
            return answer
        
        except Exception as e:
            warnings.warn(f"生成失败: {e}")
            return ""
    
    # =========================================================================
    # 完整Pipeline
    # =========================================================================
    
    def run_single(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个样本
        
        Args:
            sample: 样本字典
            
        Returns:
            Dict: 结果字典
        """
        question = sample['question']
        image = sample.get('image', None)
        
        # 步骤1: 检索（总是检索）
        retrieved_docs, retrieval_scores = self.retrieve(question, image, self.top_k)
        
        # 步骤2: Position-weighted融合
        if retrieved_docs:
            context = self.apply_position_weighted_fusion(
                retrieved_docs, retrieval_scores
            )
        else:
            context = ""
        
        # 步骤3: 生成答案
        answer = self.generate_answer(question, context, image)
        
        return {
            'id': sample.get('id'),
            'question': question,
            'prediction': answer,
            'golden_answers': sample.get('golden_answers', []),
            'retrieved_docs_count': len(retrieved_docs),
            'position_weighted': self.use_position_weighting,
            'context_length': len(context)
        }
    
    def run(self, dataset, verbose: bool = True) -> List[Dict[str, Any]]:
        """
        在数据集上运行VisRAG
        
        Args:
            dataset: 数据集
            verbose: 是否显示进度
            
        Returns:
            List[Dict]: 结果列表
        """
        results = []
        
        if verbose:
            print("VisRAG Baseline运行中...")
            print("核心特点：")
            print(f"  - Position-weighted pooling: {self.use_position_weighting}")
            print(f"  - Decay factor: {self.position_weight_decay}")
            print(f"  - Top-K: {self.top_k}")
            print()
        
        for i, sample in enumerate(dataset):
            try:
                result = self.run_single(sample)
                results.append(result)
                
                if verbose and (i + 1) % 10 == 0:
                    print(f"进度: {i+1}/{len(dataset)} ({(i+1)/len(dataset)*100:.1f}%)")
            
            except Exception as e:
                warnings.warn(f"样本{i}处理失败: {e}")
                continue
        
        if verbose:
            print(f"\n✅ 完成！处理了{len(results)}个样本")
        
        return results


# 工厂函数
def create_visrag_baseline(llava_wrapper, retriever=None, **kwargs):
    """创建VisRAG baseline"""
    return VisRAGBaseline(llava_wrapper, retriever, kwargs)


if __name__ == '__main__':
    print("VisRAG Baseline (简化版)")
    print("=" * 70)
    print("基于论文: VisRAG: Vision-based Retrieval-Augmented Generation")
    print("\n核心创新:")
    print("  1. 直接使用文档图像（而不是解析为文本）")
    print("  2. Position-weighted pooling")
    print("  3. VLM-based检索和生成")
    print("\n简化版特点:")
    print("  - 使用CLIP检索（代替VisRAG-Ret）")
    print("  - 使用LLaVA生成（VisRAG-Gen）")
    print("  - 实现position-weighted融合")
    print("\n对比价值:")
    print("  vs MuRAG: 有position-aware处理")
    print("  vs 我们的方法: 位置处理策略不同（权重 vs 注意力融合）")
    print("\n使用方法:")
    print("  from experiments.baselines.visrag_baseline import VisRAGBaseline")
    print("  baseline = VisRAGBaseline(llava, retriever)")
    print("  results = baseline.run(dataset)")
    print("=" * 70)


