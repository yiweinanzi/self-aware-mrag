#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RagVL Baseline - 简化版

基于论文: MLLM Is a Strong Reranker: Advancing Multimodal Retrieval-augmented 
         Generation via Knowledge-enhanced Reranking and Noise-injected Training
论文: arXiv:2407.21439

核心思想：
1. **MLLM作为强Reranker**: 使用VLM判断检索结果的相关性
2. **Knowledge-enhanced Reranking**: 使用图像caption增强reranking
3. **Noise-injected Training**: 训练时注入噪声提高鲁棒性
4. **两阶段流程**: CLIP粗检索 → MLLM rerank → 生成

简化说明：
- 使用提示词实现reranking（不需要重新训练）
- 使用LLaVA同时作为reranker和generator
- 保留核心的二阶段流程

对比价值：
- 展示MLLM reranking的效果
- 与简单检索的对比
- 知识增强的reranking策略
"""

import torch
import warnings
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

class RagVLBaseline:
    """
    RagVL Baseline (简化版)
    
    两阶段流程：
    1. 粗检索: CLIP/BGE检索Top-K候选（如K=20）
    2. 精排序: MLLM reranking选择最相关的N个（如N=2-5）
    3. 生成: 基于rerank后的证据生成答案
    
    使用示例：
    ```python
    baseline = RagVLBaseline(
        llava_wrapper=llava,
        retriever=retriever,
        config={'use_reranking': True}
    )
    results = baseline.run(dataset)
    ```
    """
    
    def __init__(self, llava_wrapper, retriever=None, config=None):
        """
        初始化RagVL baseline
        
        Args:
            llava_wrapper: LLaVA模型（同时用作reranker和generator）
            retriever: 粗检索器（CLIP/BGE）
            config: 配置字典
        """
        self.llava = llava_wrapper
        self.retriever = retriever
        self.config = config or {}
        
        # 配置
        self.clip_topk = self.config.get('clip_topk', 20)  # 粗检索K
        self.rerank_topk = self.config.get('rerank_topk', 2)  # 精排序后保留N
        self.max_new_tokens = self.config.get('max_new_tokens', 100)
        self.temperature = self.config.get('temperature', 0.2)
        
        # RagVL特定配置
        self.use_reranking = self.config.get('use_reranking', True)
        self.use_caption = self.config.get('use_caption', False)  # 是否使用caption增强
        self.relevance_threshold = self.config.get('relevance_threshold', 0.5)
        
        # 总是检索
        self.always_retrieve = True
        
        # Reranking提示词模板
        self._init_prompts()
    
    def _init_prompts(self):
        """初始化提示词模板"""
        # Reranking提示词（判断相关性）
        self.rerank_prompt_simple = """Question: {question}

Is this image/document relevant to answering the question?

Answer with ONLY 'Yes' or 'No':"""
        
        self.rerank_prompt_with_caption = """Image Caption: {caption}

Question: {question}

Based on the image and its caption, is the image relevant to the question?

Answer with ONLY 'Yes' or 'No':"""
        
        # 生成提示词
        self.generation_prompt = """Use the following relevant information to answer the question concisely.

Relevant Evidence:
{evidence}

Question: {question}

Answer:"""
    
    # =========================================================================
    # 核心创新1: MLLM-based Reranking
    # =========================================================================
    
    def rerank_single(self, question: str, doc: str, 
                     image=None, caption: str = None) -> Tuple[bool, float]:
        """
        使用MLLM判断单个文档的相关性
        
        Args:
            question: 问题
            doc: 文档文本
            image: 文档图像（可选）
            caption: 文档caption（可选）
            
        Returns:
            (is_relevant, relevance_score)
        """
        # 构建reranking提示词
        if self.use_caption and caption:
            prompt = self.rerank_prompt_with_caption.format(
                caption=caption,
                question=question
            )
        else:
            prompt = self.rerank_prompt_simple.format(question=question)
        
        try:
            # 使用LLaVA判断相关性
            response = self.llava.generate(
                text=prompt,
                image=image,
                max_new_tokens=10,
                temperature=0.1  # 低温度获得确定性输出
            )
            
            response_lower = response.strip().lower()
            
            # 解析响应并估计概率
            if 'yes' in response_lower:
                is_relevant = True
                # 简化版：使用固定概率
                # 完整版应该从logits提取P(Yes)
                relevance_score = 0.9
            elif 'no' in response_lower:
                is_relevant = False
                relevance_score = 0.1
            else:
                # 不确定，保守判断
                is_relevant = True
                relevance_score = 0.5
                warnings.warn(f"Reranking响应不明确: {response}")
            
            return is_relevant, relevance_score
        
        except Exception as e:
            warnings.warn(f"Reranking失败: {e}")
            return True, 0.5
    
    def rerank_documents(self, question: str, 
                        retrieved_docs: List[str],
                        retrieval_scores: List[float],
                        image=None) -> List[Tuple[str, float]]:
        """
        对检索结果进行reranking
        
        RagVL的核心：使用MLLM作为强大的reranker
        
        Args:
            question: 问题
            retrieved_docs: 粗检索的文档列表
            retrieval_scores: 粗检索分数
            image: 查询图像
            
        Returns:
            List[Tuple[str, float]]: rerank后的(文档, 分数)对，按分数降序
        """
        if not self.use_reranking:
            # 不使用reranking，直接返回原检索结果
            return [(doc, score) for doc, score in zip(retrieved_docs, retrieval_scores)]
        
        # 对每个文档进行relevance判断
        reranked = []
        
        for doc, ret_score in zip(retrieved_docs, retrieval_scores):
            is_relevant, rel_score = self.rerank_single(
                question, doc, image
            )
            
            if is_relevant:
                # 综合分数：检索分数 × 相关性分数
                combined_score = ret_score * rel_score
                reranked.append((doc, combined_score))
        
        # 按综合分数排序
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        # 只保留Top-N
        reranked = reranked[:self.rerank_topk]
        
        return reranked
    
    # =========================================================================
    # 检索和生成
    # =========================================================================
    
    def retrieve(self, question: str, image=None, 
                top_k: int = None) -> Tuple[List[str], List[float]]:
        """
        粗检索（CLIP）
        
        Args:
            question: 问题
            image: 图像
            top_k: 检索数量（默认使用clip_topk）
            
        Returns:
            (documents, scores)
        """
        if top_k is None:
            top_k = self.clip_topk
        
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
            warnings.warn(f"粗检索失败: {e}")
            return [], []
    
    def generate_answer(self, question: str, evidence_docs: List[Tuple[str, float]], 
                       image=None) -> str:
        """
        基于rerank后的证据生成答案
        
        Args:
            question: 问题
            evidence_docs: rerank后的(文档, 分数)列表
            image: 图像
            
        Returns:
            str: 答案
        """
        if not evidence_docs:
            # 无相关证据，直接回答
            prompt = f"Question: {question}\n\nAnswer:"
        else:
            # 组织证据
            evidence_parts = []
            for i, (doc, score) in enumerate(evidence_docs):
                evidence_parts.append(f"[Evidence {i+1}] (Relevance: {score:.2f})\n{doc}")
            
            evidence_str = "\n\n".join(evidence_parts)
            
            prompt = self.generation_prompt.format(
                evidence=evidence_str,
                question=question
            )
        
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
        RagVL完整流程
        
        Args:
            sample: 样本字典
            
        Returns:
            Dict: 结果字典
        """
        question = sample['question']
        image = sample.get('image', None)
        
        # 阶段1: 粗检索（CLIP, Top-20）
        retrieved_docs, retrieval_scores = self.retrieve(
            question, image, self.clip_topk
        )
        
        # 阶段2: MLLM Reranking（选Top-2）
        reranked_docs = self.rerank_documents(
            question, retrieved_docs, retrieval_scores, image
        )
        
        # 阶段3: 生成答案
        answer = self.generate_answer(question, reranked_docs, image)
        
        return {
            'id': sample.get('id'),
            'question': question,
            'prediction': answer,
            'golden_answers': sample.get('golden_answers', []),
            'retrieved_count': len(retrieved_docs),
            'reranked_count': len(reranked_docs),
            'avg_retrieval_score': np.mean(retrieval_scores) if retrieval_scores else 0.0,
            'avg_rerank_score': np.mean([s for _, s in reranked_docs]) if reranked_docs else 0.0,
            'used_reranking': self.use_reranking
        }
    
    def run(self, dataset, verbose: bool = True) -> List[Dict[str, Any]]:
        """
        在数据集上运行RagVL
        
        Args:
            dataset: 数据集
            verbose: 是否显示进度
            
        Returns:
            List[Dict]: 结果列表
        """
        results = []
        
        if verbose:
            print("RagVL Baseline运行中...")
            print("两阶段流程：")
            print(f"  1. 粗检索（CLIP）: Top-{self.clip_topk}")
            print(f"  2. 精排序（MLLM）: Top-{self.rerank_topk}")
            print(f"  3. 生成答案")
            print(f"\n配置:")
            print(f"  - Use reranking: {self.use_reranking}")
            print(f"  - Relevance threshold: {self.relevance_threshold}")
            print()
        
        total_retrieved = 0
        total_reranked = 0
        
        for i, sample in enumerate(dataset):
            try:
                result = self.run_single(sample)
                results.append(result)
                
                total_retrieved += result['retrieved_count']
                total_reranked += result['reranked_count']
                
                if verbose and (i + 1) % 10 == 0:
                    print(f"进度: {i+1}/{len(dataset)} ({(i+1)/len(dataset)*100:.1f}%)")
                    print(f"  平均检索数: {total_retrieved/(i+1):.1f}")
                    print(f"  平均rerank后: {total_reranked/(i+1):.1f}")
            
            except Exception as e:
                warnings.warn(f"样本{i}处理失败: {e}")
                continue
        
        if verbose:
            print(f"\n✅ 完成！处理了{len(results)}个样本")
            print(f"  平均粗检索: {total_retrieved/len(results):.1f} 个")
            print(f"  平均精排序后: {total_reranked/len(results):.1f} 个")
            print(f"  压缩率: {total_reranked/max(total_retrieved,1)*100:.1f}%")
        
        return results


# 工厂函数
def create_ragvl_baseline(llava_wrapper, retriever=None, **kwargs):
    """创建RagVL baseline"""
    return RagVLBaseline(llava_wrapper, retriever, kwargs)


if __name__ == '__main__':
    print("RagVL Baseline (简化版)")
    print("=" * 70)
    print("基于论文: MLLM Is a Strong Reranker")
    print("arXiv:2407.21439")
    print("\n核心创新:")
    print("  1. MLLM作为强大的Reranker")
    print("  2. Knowledge-enhanced reranking（使用caption）")
    print("  3. 两阶段检索（粗检索 + 精排序）")
    print("\n流程:")
    print("  粗检索（CLIP）Top-20")
    print("    ↓")
    print("  精排序（MLLM）Top-2")
    print("    ↓")
    print("  生成答案（MLLM）")
    print("\n对比价值:")
    print("  vs MuRAG: 有reranking机制")
    print("  vs mR²AG: 不同的判断策略（reranking vs reflection）")
    print("  vs VisRAG/REVEAL: 有精排序步骤")
    print("  vs Ours: 我们用uncertainty而非reranking")
    print("\n使用方法:")
    print("  from experiments.baselines.ragvl_baseline import RagVLBaseline")
    print("  baseline = RagVLBaseline(llava, retriever)")
    print("  results = baseline.run(dataset)")
    print("=" * 70)


