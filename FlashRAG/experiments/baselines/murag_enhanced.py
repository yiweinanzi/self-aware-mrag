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
from flashrag.utils.vqa_evaluator import extract_okvqa_answer
from experiments.baselines.answer_extractor import extract_answer_smart
from experiments.baselines.evaluation_helper import evaluate_answer_correctness


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
            golden_answers = sample.get('golden_answers', [])
            # 计算correct字段
            is_correct = evaluate_answer_correctness(answer, golden_answers)
            return {
                'question': question,
                'answer': answer,
                'sub_answers': [],
                'retrieved_docs': [],  # 空列表而不是0
                'retrieved': False,  # 明确标���为未检索到
                'golden_answers': golden_answers,
                'correct': is_correct
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

        # 计算correct字段
        golden_answers = sample.get('golden_answers', [])
        is_correct = evaluate_answer_correctness(answer, golden_answers)

        return {
            'question': question,
            'answer': answer,
            'sub_answers': sub_answers,
            'retrieved_docs': docs,  # 返回文档列表而不是数量
            'retrieved': len(docs) > 0,  # 是否检索到文档
            'golden_answers': golden_answers,
            'correct': is_correct
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

            # Handle different return types from retriever
            if isinstance(results, tuple):
                docs, scores = results
            elif isinstance(results, list):
                docs = results
            elif isinstance(results, int):
                # If an integer is returned, it's likely an error code or count
                warnings.warn(f"检索器返回了整数而非文档列表: {results}")
                return []
            else:
                docs = [results]

            # 返回文档字典列表而不是文本列表
            docs_list = []
            for doc in docs:
                if isinstance(doc, dict):
                    docs_list.append(doc)  # 保留完整的文档对象
                elif isinstance(doc, int):
                    # Skip integer values in document list
                    warnings.warn(f"文档列表中包含整数: {doc}")
                    continue
                else:
                    # 如果不是字典，转换成字典格式
                    docs_list.append({
                        'contents': str(doc),
                        'id': str(hash(str(doc))),
                        'title': '',
                        'source': 'murag_retriever'
                    })

            return docs_list

        except Exception as e:
            warnings.warn(f"检索失败: {e}")
            return []
    
    def _generate_with_single_doc(self, sample: Dict, doc: Dict) -> str:
        """
        基于单个文档独立生成答案（FiD风格，MuRAG核心）
        """
        if all(k in sample for k in ['A', 'B', 'C', 'D']):
            prompt = f"""Based ONLY on this single evidence document, answer the question.

Evidence: {doc.get('contents', doc.get('text', str(doc)))[:300]}...

Question: {sample['question']}

Choices:
A. {sample['A']}
B. {sample['B']}
C. {sample['C']}
D. {sample['D']}

Answer (letter only):"""
        else:
            prompt = f"""Answer with 1-3 words only based on the evidence.

Evidence: {doc.get('contents', doc.get('text', str(doc)))[:300]}...

Question: {sample['question']}

Answer:"""
        
        try:
            answer = self.qwen3vl.generate(
                text=prompt,
                image=sample.get('image'),
                max_new_tokens=20,
                temperature=self.temperature,
                do_sample=False
            )
            
            if all(k in sample for k in ['A', 'B', 'C', 'D']):
                return self._map_mc_answer(answer, sample)

            # 使用改进的答案提取器
            return extract_answer_smart(answer)

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
            prompt = f"Answer with 1-3 words only.\n\nQuestion: {sample['question']}\n\nAnswer:"
        
        try:
            answer = self.qwen3vl.generate(
                text=prompt,
                image=sample.get('image'),
                max_new_tokens=20,
                temperature=self.temperature,
                do_sample=False
            )
            
            if all(k in sample for k in ['A', 'B', 'C', 'D']):
                return self._map_mc_answer(answer, sample)

            # 使用改进的答案提取器
            return extract_answer_smart(answer)

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

