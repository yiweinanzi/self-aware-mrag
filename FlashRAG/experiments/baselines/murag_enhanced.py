#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MuRAG Enhanced - 完整实现（基于Qwen3-VL）

核心特色:
1. FiD式多证据并行处理
2. 投票融合
"""

import sys
sys.path.insert(0, '/root/autodl-tmp/FlashRAG')

from typing import Dict, Any, List
from collections import Counter
import warnings


class MuRAGEnhanced:
    """
    MuRAG Enhanced完整实现
    
    核心流程:
    1. 检索多个证据（top-10）
    2. 每个证据独立生成答案（FiD风格）
    3. 投票选择最终答案
    """
    
    def __init__(self, qwen3vl_wrapper, retriever=None, config=None):
        self.qwen3vl = qwen3vl_wrapper
        self.retriever = retriever
        self.config = config or {}
        
        self.top_k = self.config.get('retrieval_topk', 10)
        self.ensemble_k = self.config.get('ensemble_k', 5)  # 用于投票的证据数
        self.temperature = self.config.get('temperature', 0.01)
    
    def run_single(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        question = sample['question']
        image = sample.get('image')
        
        # Step 1: 检索多个证据
        docs = self._retrieve_documents(question)
        
        if not docs:
            answer = self._direct_answer(sample)
            return {
                'question': question,
                'answer': answer,
                'sub_answers': [],
                'retrieved_docs': 0
            }
        
        # Step 2: FiD式并行处理（关键！）
        sub_answers = []
        for doc in docs[:self.ensemble_k]:
            sub_ans = self._generate_with_single_doc(sample, doc)
            if sub_ans:
                sub_answers.append(sub_ans)
        
        # Step 3: 投票融合（关键！）
        if sub_answers:
            answer = self._voting_fusion(sub_answers)
        else:
            answer = self._direct_answer(sample)
        
        return {
            'question': question,
            'answer': answer,
            'sub_answers': sub_answers,
            'retrieved_docs': len(docs)
        }
    
    def _retrieve_documents(self, question: str) -> List[str]:
        """检索文档"""
        if self.retriever is None:
            return []
        
        try:
            if hasattr(self.retriever, 'search'):
                results = self.retriever.search(question, num=self.top_k)
            elif hasattr(self.retriever, 'retrieve'):
                results = self.retriever.retrieve(query_text=question, top_k=self.top_k)
            else:
                return []
            
            if isinstance(results, tuple):
                docs, scores = results
            elif isinstance(results, list):
                docs = results
            else:
                docs = [results]
            
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
    
    def _generate_with_single_doc(self, sample: Dict, doc: str) -> str:
        """
        基于单个文档独立生成答案（FiD风格，MuRAG核心）
        """
        if all(k in sample for k in ['A', 'B', 'C', 'D']):
            prompt = f"""Based ONLY on this single evidence document, answer the question.

Evidence: {doc[:300]}...

Question: {sample['question']}

Choices:
A. {sample['A']}
B. {sample['B']}
C. {sample['C']}
D. {sample['D']}

Answer (letter only):"""
        else:
            prompt = f"""Based ONLY on this evidence, answer briefly.

Evidence: {doc[:300]}...

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
            
            if all(k in sample for k in ['A', 'B', 'C', 'D']):
                return self._map_mc_answer(answer, sample)
            
            return answer.strip()
        
        except Exception as e:
            warnings.warn(f"单文档生成失败: {e}")
            return ""
    
    def _voting_fusion(self, sub_answers: List[str]) -> str:
        """
        投票融合（MuRAG核心特色）
        """
        # 统计答案频率
        answer_counts = Counter(sub_answers)
        
        # 返回最常见的答案
        if answer_counts:
            most_common = answer_counts.most_common(1)[0]
            return most_common[0]
        
        return sub_answers[0] if sub_answers else ""
    
    def _direct_answer(self, sample: Dict) -> str:
        """直接回答（后备）"""
        if all(k in sample for k in ['A', 'B', 'C', 'D']):
            prompt = f"""Question: {sample['question']}

Choices:
A. {sample['A']}
B. {sample['B']}
C. {sample['C']}
D. {sample['D']}

Answer (letter only):"""
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
            return ""
    
    def _map_mc_answer(self, response: str, sample: Dict) -> str:
        response_upper = response.strip().upper()
        for letter in ['A', 'B', 'C', 'D']:
            if letter in response_upper:
                return sample.get(letter, response.strip())
        return response.strip()


def create_murag_enhanced(qwen3vl_wrapper, retriever=None, **kwargs):
    return MuRAGEnhanced(qwen3vl_wrapper, retriever, kwargs)

