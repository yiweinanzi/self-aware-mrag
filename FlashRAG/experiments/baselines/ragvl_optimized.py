#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RagVL Optimized - 优化版本
主要改进：
1. 限制上下文长度（每篇文档最多300字符）
2. 限制文档数量（最多3篇）
3. 移除冗余的提示词
"""

import sys
sys.path.insert(0, '/data0/home/zqwang/ACL/FlashRAG')

from typing import Dict, Any, List, Tuple
from flashrag.utils.vqa_evaluator import extract_okvqa_answer


class RagVLOptimized:
    """RagVL优化版本 - 更快的推理速度"""

    def __init__(self, qwen3vl_wrapper, retriever=None, config=None):
        self.qwen3vl = qwen3vl_wrapper
        self.retriever = retriever
        self.config = config or {}

        self.topk = self.config.get('top_k', 5)  # 检索文档数
        self.rerank_topk = self.config.get('rerank_topk', 3)  # 最终使用的文档数
        self.temperature = self.config.get('temperature', 0.01)
        self.max_doc_length = 300  # 每篇文档最大长度

        # Reranking配置
        self.use_reranking = self.config.get('use_reranking', False)  # 默认关闭

    def run_single(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """运行单个样本"""
        question = sample['question']
        image = sample.get('image')

        # === Step 1: 检索 ===
        retrieved_docs, retrieval_scores = self._retrieve(question)

        if not retrieved_docs:
            answer = self._direct_answer(sample)
            return {
                'question': question,
                'answer': answer,
                'retrieved_docs': [],
                'retrieved': False
            }

        # === Step 2: 限制文档长度和数量 ===
        processed_docs = []
        for doc in retrieved_docs[:self.rerank_topk]:
            # 截断文档
            if len(doc) > self.max_doc_length:
                doc = doc[:self.max_doc_length] + "..."
            processed_docs.append(doc)

        # === Step 3: 生成答案 ===
        answer = self._generate_answer(sample, processed_docs)

        return {
            'question': question,
            'answer': answer,
            'retrieved_docs': processed_docs,
            'retrieved': True
        }

    def _retrieve(self, question: str) -> Tuple[List[str], List[float]]:
        """检索文档"""
        if self.retriever is None:
            return [], []

        try:
            results = self.retriever.search(question, num=self.topk)

            if isinstance(results, tuple):
                docs, scores = results
            elif isinstance(results, list):
                # 处理不同的返回格式
                if results and hasattr(results[0], '__len__') and len(results[0]) == 2:
                    docs, scores = zip(*results)
                    docs, scores = list(docs), list(scores)
                else:
                    docs = results
                    scores = [1.0] * len(docs)
            else:
                return [], []

            return docs[:self.topk], scores[:self.topk]

        except Exception as e:
            print(f"[ERROR] Retrieval failed: {e}")
            return [], []

    def _generate_answer(self, sample: Dict, docs: List[str]) -> str:
        """基于文档生成答案（简化版本）"""
        if not docs:
            return self._direct_answer(sample)

        # 构建简化的prompt
        evidence = "\n".join([f"- {doc}" for doc in docs[:3]])  # 最多3篇文档

        if all(k in sample for k in ['A', 'B', 'C', 'D']):
            # 多选题
            prompt = f"""Answer the question based on the evidence.

Evidence:
{evidence}

Question: {sample['question']}

Choices:
A. {sample['A']}
B. {sample['B']}
C. {sample['C']}
D. {sample['D']}

Answer with the letter only:"""
        else:
            # 开放式问题
            prompt = f"""Based on the evidence, answer the question in 1-3 words.

Evidence:
{evidence[:500]}  # 限制总长度

Question: {sample['question']}

Answer:"""

        try:
            answer = self.qwen3vl.generate(
                text=prompt,
                image=sample.get('image'),
                temperature=self.temperature,
                max_new_tokens=10
            )

            # 清理答案
            answer = answer.strip()
            if answer.endswith('.'):
                answer = answer[:-1]

            return answer

        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            return self._direct_answer(sample)

    def _direct_answer(self, sample: Dict) -> str:
        """直接回答（不使用检索）"""
        if all(k in sample for k in ['A', 'B', 'C', 'D']):
            prompt = f"""{sample['question']}

Choices:
A. {sample['A']}
B. {sample['B']}
C. {sample['C']}
D. {sample['D']}

Answer with the letter only:"""
        else:
            prompt = f"Answer the question in 1-3 words: {sample['question']}"

        try:
            answer = self.qwen3vl.generate(
                text=prompt,
                image=sample.get('image'),
                temperature=self.temperature,
                max_new_tokens=10
            )
            return answer.strip()
        except:
            return ""