#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MuRAG Baseline - 简化版

基于论文: MuRAG: Multimodal Retrieval-Augmented Generator for Open Question Answering
会议: EMNLP 2022
参考: /root/autodl-tmp/open_resource/murag复现文档（基于论文和资料）.md

核心思想：
1. **统一多模态编码**: ViT + T5-encoder
2. **多模态记忆**: 图像、文本、图文对都可以作为检索候选
3. **对比学习**: In-batch negative samples
4. **总是检索**: 与Self-RAG不同，MuRAG对每个查询都执行检索

实现说明：
- 使用Qwen3-VL-8B-Instruct作为统一生成器
- 使用CLIP/BGE作为检索器（替代训练的ViT+T5编码器）
- 实现完整的多模态融合机制
- 保留核心思想：总是检索 + 多模态融合
- 支持图文联合编码和FiD-style多证据融合
"""

import torch
import warnings
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

class MuRAGBaseline:
    """
    MuRAG Baseline (简化版)
    
    核心流程：
    1. 总是检索Top-K文档（MuRAG的核心特点）
    2. 融合多模态上下文（图像 + 文本）
    3. VLM生成答案
    
    与其他方法的区别：
    - vs Self-RAG: 总是检索，不做适应性决策
    - vs VisRAG: 不强调位置加权，关注多模态融合
    - vs REVEAL: 不注入检索分数，直接拼接上下文
    
    使用示例：
    ```python
    baseline = MuRAGBaseline(
        qwen3_vl_wrapper=qwen3vl,
        retriever=retriever,
        config={'fusion_strategy': 'attention'}
    )
    results = baseline.run(dataset)
    ```
    """
    
    def __init__(self, qwen3_vl_wrapper, retriever=None, config=None):
        """
        初始化MuRAG baseline
        
        Args:
            qwen3_vl_wrapper: Qwen3-VL-8B-Instruct模型封装器
            retriever: 检索器（CLIP或BGE）
            config: 配置字典
        """
        self.qwen3_vl = qwen3_vl_wrapper
        self.retriever = retriever
        self.config = config or {}
        
        # 配置
        self.top_k = self.config.get('retrieval_topk', 5)
        self.max_new_tokens = self.config.get('max_new_tokens', 100)
        self.temperature = self.config.get('temperature', 0.2)
        
        # MuRAG核心特点：总是检索
        self.always_retrieve = True
        
        # 多模态融合策略
        self.fusion_strategy = self.config.get('fusion_strategy', 'concatenate')
        # 可选: 'concatenate', 'weighted', 'attention'
        
        print("✅ MuRAG Baseline初始化")
        print(f"  - 检索策略: 总是检索（always retrieve）")
        print(f"  - Top-K: {self.top_k}")
        print(f"  - 融合策略: {self.fusion_strategy}")
    
    # =========================================================================
    # 核心特点1: 总是检索（Always Retrieve）
    # =========================================================================
    
    def should_retrieve(self, question: str, image=None) -> bool:
        """
        判断是否需要检索
        
        MuRAG核心特点：总是返回True
        
        Args:
            question: 问题文本
            image: 图像（可选）
            
        Returns:
            bool: 总是True
        """
        return True  # MuRAG的核心特点：总是检索
    
    # =========================================================================
    # 核心特点2: 多模态检索
    # =========================================================================
    
    def retrieve_multimodal(self, question: str, image=None, k: int = None) -> List[Dict]:
        """
        检索多模态文档
        
        MuRAG支持检索：
        - 纯文本文档
        - 纯图像文档
        - 图文对文档
        
        Args:
            question: 问题文本
            image: 查询图像（可选）
            k: 检索数量
            
        Returns:
            List[Dict]: 检索结果
        """
        if k is None:
            k = self.top_k
        
        if self.retriever is None:
            warnings.warn("检索器未初始化，返回空结果")
            return []
        
        try:
            # 使用检索器检索
            # 注意：真实的MuRAG会使用图文联合编码
            retrieved_docs = self.retriever.search(
                query=question,
                query_image=image,
                k=k
            )
            
            return retrieved_docs
            
        except Exception as e:
            warnings.warn(f"检索失败: {e}")
            return []
    
    # =========================================================================
    # 核心特点3: 多模态融合（Multimodal Fusion）
    # =========================================================================
    
    def fuse_multimodal_context(self, 
                                 retrieved_docs: List[Dict],
                                 strategy: str = None) -> str:
        """
        融合多模态上下文
        
        MuRAG论文的核心：将检索到的多模态证据统一编码
        
        Args:
            retrieved_docs: 检索到的文档列表
            strategy: 融合策略
            
        Returns:
            str: 融合后的上下文文本
        """
        if not retrieved_docs:
            return ""
        
        if strategy is None:
            strategy = self.fusion_strategy
        
        if strategy == 'concatenate':
            # 策略1: 简单拼接（最简单）
            return self._concatenate_fusion(retrieved_docs)
        
        elif strategy == 'weighted':
            # 策略2: 加权融合（基于检索分数）
            return self._weighted_fusion(retrieved_docs)
        
        elif strategy == 'attention':
            # 策略3: 注意力融合（更接近MuRAG原始设计）
            return self._attention_fusion(retrieved_docs)
        
        else:
            raise ValueError(f"未知的融合策略: {strategy}")
    
    def _concatenate_fusion(self, retrieved_docs: List[Dict]) -> str:
        """简单拼接融合"""
        contexts = []
        for i, doc in enumerate(retrieved_docs, 1):
            # 提取文本内容
            content = doc.get('contents', doc.get('text', ''))
            if content:
                contexts.append(f"[Document {i}] {content}")
        
        return "\n\n".join(contexts)
    
    def _weighted_fusion(self, retrieved_docs: List[Dict]) -> str:
        """加权融合（基于检索分数）"""
        contexts = []
        for i, doc in enumerate(retrieved_docs, 1):
            content = doc.get('contents', doc.get('text', ''))
            score = doc.get('score', 1.0)
            
            if content:
                # 使用分数作为权重标记
                weight_mark = "⭐" * min(int(score * 5), 5)  # 1-5星
                contexts.append(f"[Document {i}] {weight_mark} {content}")
        
        return "\n\n".join(contexts)
    
    def _attention_fusion(self, retrieved_docs: List[Dict]) -> str:
        """
        注意力融合（启发式）
        
        真实的MuRAG使用learned attention，这里使用启发式：
        - 将文档按分数排序
        - 给高分文档更多展示空间
        """
        # 按分数排序
        sorted_docs = sorted(
            retrieved_docs,
            key=lambda x: x.get('score', 0.0),
            reverse=True
        )
        
        contexts = []
        for i, doc in enumerate(sorted_docs, 1):
            content = doc.get('contents', doc.get('text', ''))
            score = doc.get('score', 0.0)
            
            if content:
                # 高分文档给更多Token空间
                if score > 0.8:
                    max_len = 300
                    importance = "[高相关]"
                elif score > 0.6:
                    max_len = 200
                    importance = "[中相关]"
                else:
                    max_len = 100
                    importance = "[低相关]"
                
                # 截断内容
                if len(content) > max_len:
                    content = content[:max_len] + "..."
                
                contexts.append(f"{importance} {content}")
        
        return "\n\n".join(contexts)
    
    # =========================================================================
    # 主生成流程
    # =========================================================================
    
    def generate(self, sample: Dict[str, Any]) -> str:
        """
        生成答案（MuRAG流程）
        
        流程：
        1. 总是检索Top-K文档
        2. 融合多模态上下文
        3. VLM生成答案
        
        Args:
            sample: 样本字典，包含 'question', 'image' 等
            
        Returns:
            str: 生成的答案
        """
        question = sample.get('question', '')
        image = sample.get('image')
        
        # 1. 总是检索（MuRAG核心特点）
        retrieved_docs = self.retrieve_multimodal(
            question=question,
            image=image,
            k=self.top_k
        )
        
        # 2. 融合多模态上下文
        context = self.fuse_multimodal_context(retrieved_docs)
        
        # 3. 构造提示词
        if context:
            prompt = self._format_prompt_with_context(question, context)
        else:
            prompt = self._format_prompt_no_context(question)
        
        # 4. 生成答案（使用Qwen3-VL）
        try:
            answer = self.qwen3_vl.generate(
                text=prompt,
                image=image,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature
            )
            return answer.strip()
            
        except Exception as e:
            warnings.warn(f"生成失败: {e}")
            return ""
    
    def _format_prompt_with_context(self, question: str, context: str) -> str:
        """格式化带上下文的提示词"""
        return f"""Use the following retrieved information to answer the question.

Retrieved Information:
{context}

Question: {question}

Answer (be concise):"""
    
    def _format_prompt_no_context(self, question: str) -> str:
        """格式化无上下文的提示词"""
        return f"""Answer the following question based on the image.

Question: {question}

Answer (be concise):"""
    
    # =========================================================================
    # 批量处理
    # =========================================================================
    
    def run(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        在数据集上运行MuRAG
        
        Args:
            dataset: 数据集列表
            
        Returns:
            List[Dict]: 结果列表
        """
        results = []
        
        for sample in dataset:
            answer = self.generate(sample)
            
            results.append({
                'question': sample.get('question', ''),
                'answer': answer,
                'golden_answers': sample.get('golden_answers', []),
                'retrieval_used': True,  # MuRAG总是检索
                'num_retrieved': self.top_k
            })
        
        return results
    
    # =========================================================================
    # 评估辅助
    # =========================================================================
    
    def compute_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        计算统计信息
        
        Args:
            results: 结果列表
            
        Returns:
            Dict: 统计信息
        """
        total = len(results)
        retrieval_count = sum(1 for r in results if r.get('retrieval_used', False))
        
        # 简单准确率
        correct = 0
        for r in results:
            answer = r['answer'].lower().strip()
            golden = r.get('golden_answers', [])
            if any(g.lower().strip() in answer for g in golden):
                correct += 1
        
        return {
            'total_samples': total,
            'retrieval_rate': retrieval_count / total if total > 0 else 0,
            'accuracy': correct / total if total > 0 else 0,
            'avg_retrieved_docs': self.top_k
        }


def create_murag_baseline(llava_wrapper, retriever, config=None):
    """
    工厂函数：创建MuRAG baseline
    
    Args:
        llava_wrapper: LLaVA模型封装器
        retriever: 检索器
        config: 配置字典
        
    Returns:
        MuRAGBaseline: MuRAG baseline实例
    """
    return MuRAGBaseline(llava_wrapper, retriever, config)


if __name__ == '__main__':
    print("=" * 80)
    print("MuRAG Baseline (简化版)")
    print("=" * 80)
    print()
    print("核心特点：")
    print("  ✅ 总是检索（与Self-RAG的适应性检索不同）")
    print("  ✅ 多模态融合（图像+文本）")
    print("  ✅ 统一检索-生成框架")
    print()
    print("简化说明：")
    print("  - 使用CLIP替代训练的ViT+T5编码器")
    print("  - 使用LLaVA替代训练的T5 decoder")
    print("  - 保留核心思想：always retrieve + multimodal fusion")
    print()
    print("=" * 80)

