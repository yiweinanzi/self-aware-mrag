#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAM-RAG Adapted - 简化版实现

Self-Aware Memory RAG 带记忆增强
"""

import sys
sys.path.insert(0, '/root/autodl-tmp/FlashRAG')

from typing import Dict, Any, List
import warnings
from flashrag.utils.vqa_evaluator import extract_okvqa_answer
from experiments.baselines.evaluation_helper import evaluate_answer_correctness


class SAMRAGAdapted:
    """
    SAM-RAG简化版实现
    """

    def __init__(self, qwen3vl_wrapper, retriever=None, config=None):
        self.qwen3vl = qwen3vl_wrapper
        self.retriever = retriever
        self.config = config or {}

        self.top_k = self.config.get('retrieval_topk', 5)
        self.temperature = self.config.get('temperature', 0.01)
        self.memory_size = self.config.get('memory_size', 10)

        # 模拟记忆存储
        self.memory_bank = []

    def run_single(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        question = sample['question']
        image = sample.get('image')

        # 检索文档
        retrieved_docs = self._retrieve_documents(question)

        # 更新记忆
        self._update_memory(question, retrieved_docs)

        if not retrieved_docs:
            answer = self._direct_answer(sample)
            return {
                'question': question,
                'answer': answer,
                'retrieved_docs': [],
                'retrieved': False,
                'memory_size': len(self.memory_bank),
                'golden_answers': sample.get('golden_answers', []),
                'correct': False
            }

        # 生成答案
        answer = self._generate_with_memory(sample, retrieved_docs)

        # 转换为文档字典格式
        retrieved_docs_dict = []
        for i, doc in enumerate(retrieved_docs):
            if isinstance(doc, dict):
                retrieved_docs_dict.append(doc)
            else:
                retrieved_docs_dict.append({
                    'contents': str(doc),
                    'id': f'samrag_doc_{i}',
                    'title': '',
                    'source': 'samrag_retriever'
                })

        # 计算 correct 字段
        golden_answers = sample.get('golden_answers', [])
        is_correct = evaluate_answer_correctness(answer, golden_answers)

        return {
            'question': question,
            'answer': answer,
            'retrieved_docs': retrieved_docs_dict,
            'retrieved': True,
            'memory_size': len(self.memory_bank),
            'golden_answers': golden_answers,
            'correct': is_correct
        }

    def _retrieve_documents(self, question: str) -> List:
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

            # 处理结果
            if isinstance(results, tuple):
                docs, scores = results
            elif isinstance(results, list):
                docs = results
            else:
                docs = [results]

            # 返回文档字典列表
            docs_list = []
            for doc in docs:
                if isinstance(doc, dict):
                    docs_list.append(doc)
                else:
                    docs_list.append({
                        'contents': str(doc),
                        'id': str(hash(str(doc))),
                        'title': '',
                        'source': 'samrag_retriever'
                    })

            return docs_list

        except Exception as e:
            warnings.warn(f"检索失败: {e}")
            return []

    def _update_memory(self, question: str, docs: List):
        """更新记忆"""
        # 将问题和文档添加到记忆中
        memory_entry = {
            'question': question,
            'documents': docs[:3]  # 只保留前3个文档
        }
        self.memory_bank.append(memory_entry)

        # 保持记忆大小
        if len(self.memory_bank) > self.memory_size:
            self.memory_bank = self.memory_bank[-self.memory_size:]

    def _generate_with_memory(self, sample: Dict, retrieved_docs: List) -> str:
        """基于记忆生成答案"""
        question = sample['question']
        image = sample.get('image')

        # 构建记忆上下文
        memory_context = ""
        if self.memory_bank:
            memory_context = "Previous Q&A:\n"
            for i, mem in enumerate(self.memory_bank[-3:]):  # 使用最近3条记忆
                memory_context += f"Q{i+1}: {mem['question'][:100]}...\n"
                memory_context += f"A{i+1}: [Context from {len(mem['documents'])} documents]\n\n"

        # 构建当前上下文
        context_parts = []
        for doc in retrieved_docs[:3]:
            if isinstance(doc, dict):
                content = doc.get('contents', doc.get('text', str(doc)))
                context_parts.append(content)
            else:
                context_parts.append(str(doc))
        context = "\n\n".join(context_parts)

        if all(k in sample for k in ['A', 'B', 'C', 'D']):
            prompt = f"""Using the memory and context, answer the question.

Memory:
{memory_context}

Current Context:
{context}

Question: {sample['question']}

Choices:
A. {sample['A']}
B. {sample['B']}
C. {sample['C']}
D. {sample['D']}

Answer with the letter only (A/B/C/D):"""
        else:
            prompt = f"""Answer with 1-3 words only.

Memory:
{memory_context}

Current Context:
{context}

Question: {sample['question']}

Answer:"""

        try:
            answer = self.qwen3vl.generate(
                text=prompt,
                image=image,
                max_new_tokens=20,
                temperature=self.temperature,
                do_sample=False
            )

            if all(k in sample for k in ['A', 'B', 'C', 'D']):
                return self._map_mc_answer(answer, sample)

            return extract_okvqa_answer(answer.strip())

        except Exception as e:
            warnings.warn(f"生成失败: {e}")
            import traceback
            print(f"[SAM-RAG DEBUG] _generate_with_memory error: {e}")
            traceback.print_exc()
            # fallback: 尝试更简单的prompt
            try:
                simple_prompt = f"Answer directly:\n\nQuestion: {question}\n\nAnswer:"
                fallback_answer = self.qwen3vl.generate(
                    text=simple_prompt,
                    image=image,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=False
                )
                if fallback_answer:
                    return extract_okvqa_answer(fallback_answer.strip())
                else:
                    return "unknown"
            except:
                return "unknown"

    def _direct_answer(self, sample: Dict) -> str:
        """直接回答（后备）"""
        if all(k in sample for k in ['A', 'B', 'C', 'D']):
            prompt = f"""Question: {sample['question']}

Choices:
A. {sample['A']}
B. {sample['B']}
C. {sample['C']}
D. {sample['D']}

Answer with the letter only (A/B/C/D):"""
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

            return extract_okvqa_answer(answer.strip())

        except Exception as e:
            warnings.warn(f"直接回答失败: {e}")
            return "unknown"

    def _map_mc_answer(self, response: str, sample: Dict) -> str:
        """映射多选题答案"""
        response_upper = response.strip().upper()
        for letter in ['A', 'B', 'C', 'D']:
            if letter in response_upper:
                return sample.get(letter, response.strip())
        return response.strip()


def create_samrag_adapted(qwen3vl_wrapper, retriever=None, **kwargs):
    """创建SAM-RAG Adapted"""
    return SAMRAGAdapted(qwen3vl_wrapper, retriever, kwargs)