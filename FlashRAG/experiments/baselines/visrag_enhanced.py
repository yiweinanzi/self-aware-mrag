#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VisRAG Enhanced - 完整实现（基于Qwen3-VL）

核心特色:
1. BGE Reranker重排
2. 视觉优先策略
3. 多阶段检索-重排-生成
"""

import sys
sys.path.insert(0, '/root/autodl-tmp/FlashRAG')

from typing import Dict, Any, List
import warnings


class VisRAGEnhanced:
    """
    VisRAG Enhanced完整实现
    
    核心流程:
    1. 视觉优先回答
    2. 文本检索（top-10）
    3. BGE重排（top-5）
    4. 融合生成
    """
    
    def __init__(self, qwen3vl_wrapper, retriever=None, bge_reranker=None, config=None):
        """
        初始化VisRAG Enhanced
        
        Args:
            qwen3vl_wrapper: Qwen3-VL封装器
            retriever: 检索器
            bge_reranker: BGE重排器
            config: 配置
        """
        self.qwen3vl = qwen3vl_wrapper
        self.retriever = retriever
        self.bge_reranker = bge_reranker
        self.config = config or {}
        
        self.initial_topk = self.config.get('initial_topk', 10)
        self.final_topk = self.config.get('final_topk', 5)
        self.temperature = self.config.get('temperature', 0.01)
        
        # 如果没有提供reranker，尝试创建
        if self.bge_reranker is None:
            try:
                from flashrag.modules.bge_reranker import create_bge_reranker
                self.bge_reranker = create_bge_reranker()
                print("✅ VisRAG: BGE Reranker已加载")
            except Exception as e:
                warnings.warn(f"BGE Reranker加载失败: {e}，将跳过重排步骤")
    
    def run_single(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行单个样本
        
        Args:
            sample: 样本字典
            
        Returns:
            结果字典
        """
        question = sample['question']
        image = sample.get('image')
        
        # === Step 1: 视觉优先策略（可选） ===
        # visual_answer = self._visual_only_answer(sample)
        
        # === Step 2: 文本检索 (Initial Top-K) ===
        initial_docs = self._retrieve_documents(question, num=self.initial_topk)
        
        if not initial_docs:
            # 无检索结果，直接回答
            answer = self._direct_answer(sample)
            return {
                'question': question,
                'answer': answer,
                'retrieval_docs': 0,
                'reranked_docs': 0,
                'reranker_used': False
            }
        
        # === Step 3: BGE Reranking (关键!) ===
        reranked_docs = self._rerank_documents(question, initial_docs)
        
        # === Step 4: 融合生成 ===
        answer = self._generate_with_reranked_context(sample, reranked_docs)
        
        return {
            'question': question,
            'answer': answer,
            'retrieval_docs': len(initial_docs),
            'reranked_docs': len(reranked_docs),
            'reranker_used': self.bge_reranker is not None
        }
    
    # ========================================================================
    # Document Retrieval
    # ========================================================================
    
    def _retrieve_documents(self, question: str, num: int = 10) -> List[str]:
        """检索文档"""
        if self.retriever is None:
            return []
        
        try:
            # 检查retriever类型
            if hasattr(self.retriever, 'search'):
                results = self.retriever.search(question, num=num)
            elif hasattr(self.retriever, 'retrieve'):
                results = self.retriever.retrieve(query_text=question, top_k=num)
            else:
                return []
            
            # 处理结果
            if isinstance(results, tuple):
                docs, scores = results
            elif isinstance(results, list):
                docs = results
            else:
                docs = [results]
            
            # 提取文本内容
            docs_text = []
            for doc in docs:
                if isinstance(doc, dict):
                    text = doc.get('contents', doc.get('text', str(doc)))
                else:
                    text = str(doc)
                docs_text.append(text)
            
            return docs_text
        
        except Exception as e:
            warnings.warn(f"检索失败: {e}")
            return []
    
    # ========================================================================
    # BGE Reranking (核心特色!)
    # ========================================================================
    
    def _rerank_documents(self, question: str, documents: List[str]) -> List[str]:
        """
        BGE重排文档（VisRAG的核心特色）
        
        Args:
            question: 查询
            documents: 原始检索文档
            
        Returns:
            重排后的top-k文档
        """
        if self.bge_reranker is None:
            # 无reranker，返回原始top-k
            return documents[:self.final_topk]
        
        try:
            # 使用BGE重排
            reranked = self.bge_reranker.rerank(
                query=question,
                documents=documents,
                top_k=self.final_topk
            )
            return reranked
        
        except Exception as e:
            warnings.warn(f"重排失败: {e}，使用原始顺序")
            return documents[:self.final_topk]
    
    # ========================================================================
    # Answer Generation
    # ========================================================================
    
    def _generate_with_reranked_context(self, sample: Dict, reranked_docs: List[str]) -> str:
        """基于重排后的文档生成答案"""
        if not reranked_docs:
            return self._direct_answer(sample)
        
        context = "\n\n".join(reranked_docs)
        
        # 检查是否是多选题
        if all(k in sample for k in ['A', 'B', 'C', 'D']):
            prompt = f"""Using the high-quality context below (reranked for relevance), answer the question.

Context:
{context}

Question: {sample['question']}

Choices:
A. {sample['A']}
B. {sample['B']}
C. {sample['C']}
D. {sample['D']}

Answer with the letter only (A/B/C/D):"""
        else:
            prompt = f"""Using the context below, answer the question briefly.

Context:
{context}

Question: {sample['question']}

Answer:"""
        
        try:
            answer = self.qwen3vl.generate(
                text=prompt,
                image=sample.get('image'),
                max_new_tokens=10,
                temperature=self.temperature,
                do_sample=False
            )
            
            # 多选题答案映射
            if all(k in sample for k in ['A', 'B', 'C', 'D']):
                return self._map_mc_answer(answer, sample)
            
            return answer.strip()
        
        except Exception as e:
            warnings.warn(f"生成失败: {e}")
            return ""
    
    def _visual_only_answer(self, sample: Dict) -> str:
        """视觉优先回答（只基于图像）"""
        if all(k in sample for k in ['A', 'B', 'C', 'D']):
            prompt = f"""Answer based ONLY on what you see in the image.

Question: {sample['question']}

Choices:
A. {sample['A']}
B. {sample['B']}
C. {sample['C']}
D. {sample['D']}

Answer with the letter only (A/B/C/D):"""
        else:
            prompt = f"Question: {sample['question']}\n\nAnswer based on the image:"
        
        try:
            answer = self.qwen3vl.generate(
                text=prompt,
                image=sample.get('image'),
                max_new_tokens=10,
                temperature=self.temperature,
                do_sample=False
            )
            
            if all(k in sample for k in ['A', 'B', 'C', 'D']):
                return self._map_mc_answer(answer, sample)
            
            return answer.strip()
        
        except Exception as e:
            warnings.warn(f"视觉回答失败: {e}")
            return ""
    
    def _direct_answer(self, sample: Dict) -> str:
        """直接回答（后备方案）"""
        if all(k in sample for k in ['A', 'B', 'C', 'D']):
            prompt = f"""Answer this question.

Question: {sample['question']}

Choices:
A. {sample['A']}
B. {sample['B']}
C. {sample['C']}
D. {sample['D']}

Answer with the letter only (A/B/C/D):"""
        else:
            prompt = f"Question: {sample['question']}\n\nAnswer:"
        
        try:
            answer = self.qwen3vl.generate(
                text=prompt,
                image=sample.get('image'),
                max_new_tokens=10,
                temperature=self.temperature,
                do_sample=False
            )
            
            if all(k in sample for k in ['A', 'B', 'C', 'D']):
                return self._map_mc_answer(answer, sample)
            
            return answer.strip()
        
        except Exception as e:
            warnings.warn(f"直接回答失败: {e}")
            return ""
    
    def _map_mc_answer(self, response: str, sample: Dict) -> str:
        """映射多选题答案"""
        response_upper = response.strip().upper()
        
        for letter in ['A', 'B', 'C', 'D']:
            if letter in response_upper:
                return sample.get(letter, response.strip())
        
        return response.strip()


def create_visrag_enhanced(qwen3vl_wrapper, retriever=None, bge_reranker=None, **kwargs):
    """创建VisRAG Enhanced"""
    return VisRAGEnhanced(qwen3vl_wrapper, retriever, bge_reranker, kwargs)


if __name__ == '__main__':
    print("VisRAG Enhanced - 完整实现")
    print("=" * 70)
    print("核心特色:")
    print("  1. 初始检索 (top-10)")
    print("  2. BGE重排 (top-5) ← 关键!")
    print("  3. 视觉优先策略")
    print("=" * 70)
