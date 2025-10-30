#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
REVEAL Enhanced - 完整实现（基于Qwen3-VL）

核心特色:
1. Evidence-as-Instruction
2. 两阶段生成: Reasoning → Answer
"""

import sys
sys.path.insert(0, '/root/autodl-tmp/FlashRAG')

from typing import Dict, Any, List
import warnings


class REVEALEnhanced:
    """
    REVEAL Enhanced完整实现
    
    核心流程:
    1. 检索证据
    2. 生成推理过程 (Reasoning)
    3. 基于推理生成最终答案 (Answer)
    """
    
    def __init__(self, qwen3vl_wrapper, retriever=None, config=None):
        self.qwen3vl = qwen3vl_wrapper
        self.retriever = retriever
        self.config = config or {}
        
        self.top_k = self.config.get('retrieval_topk', 5)
        self.temperature = self.config.get('temperature', 0.01)
        self.reasoning_temp = self.config.get('reasoning_temp', 0.3)
    
    def run_single(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        question = sample['question']
        image = sample.get('image')
        
        # Step 1: 检索证据
        docs = self._retrieve_documents(question)
        
        if not docs:
            answer = self._direct_answer(sample)
            return {
                'question': question,
                'answer': answer,
                'reasoning': '',
                'retrieved_docs': 0
            }
        
        context = "\n\n".join(docs)
        
        # Step 2: 生成推理过程（关键！）
        reasoning = self._generate_reasoning(sample, context)
        
        # Step 3: 基于推理生成最终答案
        answer = self._generate_final_answer(sample, context, reasoning)
        
        return {
            'question': question,
            'answer': answer,
            'reasoning': reasoning,
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
    
    def _generate_reasoning(self, sample: Dict, context: str) -> str:
        """
        Stage 1: 生成推理过程（REVEAL核心特色）
        """
        prompt = f"""Given the evidence below, provide step-by-step reasoning for answering the question.

Evidence:
{context[:500]}...

Question: {sample['question']}

Step-by-step reasoning (2-3 sentences):"""
        
        try:
            reasoning = self.qwen3vl.generate(
                text=prompt,
                image=sample.get('image'),
                max_new_tokens=100,
                temperature=self.reasoning_temp,  # 稍高温度允许推理
                do_sample=True
            )
            
            return reasoning.strip()
        
        except Exception as e:
            warnings.warn(f"推理生成失败: {e}")
            return "Based on the evidence provided."
    
    def _generate_final_answer(self, sample: Dict, context: str, reasoning: str) -> str:
        """
        Stage 2: 基于推理生成最终答案
        """
        if all(k in sample for k in ['A', 'B', 'C', 'D']):
            prompt = f"""Based on the reasoning below, provide the final answer.

Question: {sample['question']}

Reasoning: {reasoning}

Choices:
A. {sample['A']}
B. {sample['B']}
C. {sample['C']}
D. {sample['D']}

Final answer (letter only A/B/C/D):"""
        else:
            prompt = f"""Based on the reasoning, provide the final answer.

Reasoning: {reasoning}

Question: {sample['question']}

Final answer (brief):"""
        
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
            warnings.warn(f"最终答案生成失败: {e}")
            return ""
    
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


def create_reveal_enhanced(qwen3vl_wrapper, retriever=None, **kwargs):
    return REVEALEnhanced(qwen3vl_wrapper, retriever, kwargs)

