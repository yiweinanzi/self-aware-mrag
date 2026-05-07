#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Position-Aware Fusion - 修复版

问题：原版的mitigate_position_bias被禁用（直接返回原序）
解决：使用VisRAG的position-weighted方法，而不是U型重排

参考：
- 文档第856-912行
- VisRAG论文：position-weighted pooling
- 实验发现：U型重排无效，但position weighting有效
"""

import torch
import numpy as np
from typing import List, Tuple, Optional

class PositionFusionFixed:
    """
    修复版的位置融合
    
    核心改进：
    1. ✅ 使用VisRAG的position-weighted方法
    2. ✅ 不使用U型重排（已证明在短文档场景无效）
    3. ✅ 结合检索分数和位置权重
    
    使用示例：
    ```python
    fusion = PositionFusionFixed(decay=0.5)
    
    # 应用位置融合
    reordered_docs, weights = fusion.apply_position_fusion(
        retrieved_docs, retrieval_scores, query
    )
    ```
    """
    
    def __init__(self, decay: float = 0.5, device: str = 'cpu'):
        """
        初始化
        
        Args:
            decay: 位置权重衰减因子（越大，后面位置权重越高）
            device: 设备
        """
        self.decay = decay
        self.device = device
        
        print(f"✅ PositionFusionFixed初始化")
        print(f"  - Decay: {decay}")
        print(f"  - Strategy: VisRAG position-weighted")
    
    def compute_position_weights(self, k: int) -> np.ndarray:
        """
        计算位置权重（参考VisRAG）
        
        VisRAG发现：后面的检索结果应该得到更多关注
        
        Args:
            k: 检索结果数量
            
        Returns:
            np.ndarray: 位置权重 [k]，归一化后和为1
        """
        # 指数递增权重（后面位置权重更高）
        weights = np.exp(np.arange(k) * self.decay)
        
        # 归一化
        weights = weights / weights.sum()
        
        return weights
    
    def apply_position_fusion(self,
                             retrieved_docs: List[str],
                             retrieval_scores: List[float],
                             query: str = None) -> Tuple[List[str], List[float]]:
        """
        应用位置感知融合（修复版）
        
        策略：
        1. 计算位置权重（后面位置权重更高）
        2. 综合权重 = 检索分数 × 位置权重
        3. 按综合权重重排序
        4. 返回Top-3
        
        Args:
            retrieved_docs: 检索到的文档列表
            retrieval_scores: 检索分数列表
            query: 查询（可选，暂未使用）
            
        Returns:
            (reordered_docs, combined_weights): 重排序的文档和权重
        """
        if not retrieved_docs:
            return [], []
        
        k = len(retrieved_docs)
        
        # 1. 计算位置权重
        position_weights = self.compute_position_weights(k)
        
        # 2. 归一化检索分数
        scores_array = np.array(retrieval_scores)
        scores_norm = scores_array / (scores_array.sum() + 1e-10)
        
        # 3. 综合权重 = 检索分数 × 位置权重
        combined_weights = scores_norm * position_weights
        combined_weights = combined_weights / (combined_weights.sum() + 1e-10)
        
        # 4. 按综合权重排序（权重高的在前）
        sorted_indices = np.argsort(combined_weights)[::-1]
        
        # 5. 重排序
        reordered_docs = [retrieved_docs[i] for i in sorted_indices]
        reordered_weights = [float(combined_weights[i]) for i in sorted_indices]
        
        # 6. 返回Top-3
        return reordered_docs[:3], reordered_weights[:3]
    
    def mitigate_position_bias(self,
                              retrieved_context: List[str],
                              query: str,
                              retrieval_scores: Optional[List[float]] = None) -> Tuple[List[str], List[float]]:
        """
        位置偏差缓解（修复版）
        
        不再直接返回原序，而是真正应用position fusion
        
        Args:
            retrieved_context: 检索到的文本列表
            query: 查询文本
            retrieval_scores: 检索分数（可选）
            
        Returns:
            (reordered_context, weights): 重排序后的文档和权重
        """
        if not retrieved_context:
            return retrieved_context, []
        
        # 如果没有提供检索分数，使用均匀分数
        if retrieval_scores is None:
            retrieval_scores = [1.0] * len(retrieved_context)
        
        # ✅ 应用位置融合（不再禁用！）
        return self.apply_position_fusion(
            retrieved_context, retrieval_scores, query
        )


# 便捷函数
def create_position_fusion(decay=0.5, device='cpu'):
    """创建位置融合模块"""
    return PositionFusionFixed(decay, device)


if __name__ == '__main__':
    print("Position Fusion - 修复版")
    print("=" * 70)
    print("修复内容：")
    print("  ❌ 原版：mitigate_position_bias被禁用（直接返回原序）")
    print("  ✅ 新版：使用VisRAG的position-weighted方法")
    print("\n核心改进：")
    print("  1. 计算位置权重（后面位置权重更高）")
    print("  2. 综合检索分数和位置权重")
    print("  3. 按综合权重重排序")
    print("  4. 不使用U型重排（已证明无效）")
    print("\n测试：")
    
    fusion = PositionFusionFixed(decay=0.5)
    
    # 模拟测试
    docs = ["Doc A", "Doc B", "Doc C", "Doc D", "Doc E"]
    scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    
    reordered, weights = fusion.apply_position_fusion(docs, scores)
    
    print("\n原始顺序：")
    for i, (doc, score) in enumerate(zip(docs, scores)):
        print(f"  {i+1}. {doc} (score: {score:.2f})")
    
    print("\n位置融合后：")
    for i, (doc, weight) in enumerate(zip(reordered, weights)):
        print(f"  {i+1}. {doc} (weight: {weight:.3f})")
    
    print("\n✅ Position Fusion修复完成！")
    print("=" * 70)


