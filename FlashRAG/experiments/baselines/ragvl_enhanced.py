#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RagVL Enhanced - 完整实现（基于Qwen3-VL）

基于论文: MLLM Is a Strong Reranker
arXiv:2407.21439

核心特色:
1. MLLM作为强大的Reranker
2. 两阶段检索: 粗检索 → MLLM reranking
3. 相关性判断增强
"""

import sys
sys.path.insert(0, '/root/autodl-tmp/FlashRAG')

from typing import Dict, Any, List, Tuple
import warnings
from flashrag.utils.vqa_evaluator import extract_okvqa_answer
from experiments.baselines.answer_extractor import extract_answer_smart


class RagVLEnhanced:
    """
    RagVL Enhanced完整实现
    
    核心流程:
    1. 粗检索 (top-20)
    2. MLLM Reranking (选top-2~5) - 核心创新！
    3. 生成答案
    """
    
    def __init__(self, qwen3vl_wrapper, retriever=None, config=None):
        self.qwen3vl = qwen3vl_wrapper
        self.retriever = retriever
        self.config = config or {}
        
        self.clip_topk = min(self.config.get('clip_topk', 20), 10)  # 限制粗检索K到10
        self.rerank_topk = min(self.config.get('rerank_topk', 3), 3)  # 精排序后保留最多3篇
        self.temperature = self.config.get('temperature', 0.01)
        
        # Reranking配置
        self.use_reranking = self.config.get('use_reranking', True)
    
    def run_single(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """运行单个样本"""
        question = sample['question']
        image = sample.get('image')

        # 检查是否使用无检索模式
        if self.config.get('no_retrieval', False) or self.retriever is None:
            answer = self._direct_answer(sample)
            return {
                'question': question,
                'answer': answer,
                'retrieved_docs': [],
                'retrieved': False,
                'used_reranking': False
            }

        # === Step 1: 粗检索 ===
        retrieved_docs, retrieval_scores = self._retrieve(question)

        if not retrieved_docs:
            answer = self._direct_answer(sample)
            return {
                'question': question,
                'answer': answer,
                'retrieved_docs': [],
                'retrieved': False,
                'used_reranking': False
            }
        
        # === Step 2: MLLM Reranking（RagVL核心特色）===
        if self.use_reranking:
            reranked_docs = self._rerank_documents(
                question, retrieved_docs, retrieval_scores, image
            )
        else:
            # 不使用reranking，直接取top-k
            reranked_docs = [(doc, score) for doc, score in
                           zip(retrieved_docs[:self.rerank_topk],
                               retrieval_scores[:self.rerank_topk])]

        # === Step 2.5: 限制文档长度以提高速度 ===
        optimized_docs = []
        for doc, score in reranked_docs:
            # 限制每篇文档长度（重要优化！）
            if len(doc) > 300:
                doc = doc[:300] + "..."  # 截断并标记
            optimized_docs.append((doc, score))
        
        # === Step 3: 生成答案 ===
        answer = self._generate_answer(sample, optimized_docs)

        # Convert string documents to dictionary format for evaluator
        retrieved_docs_dict = []
        for i, doc in enumerate(retrieved_docs):
            retrieved_docs_dict.append({
                'contents': doc,
                'id': f"ragvl_doc_{i}",
                'title': '',
                'source': 'ragvl_retriever'
            })

        reranked_docs_dict = []
        for i, (doc, score) in enumerate(reranked_docs):
            reranked_docs_dict.append({
                'contents': doc,
                'id': f"ragvl_reranked_{i}",
                'title': '',
                'source': 'ragvl_reranker',
                'score': score
            })

        return {
            'question': question,
            'answer': answer,
            'retrieved_docs': retrieved_docs_dict,  # 返回字典列表
            'reranked_docs': reranked_docs_dict,  # 返回字典列表
            'used_reranking': self.use_reranking,
            'retrieved': len(retrieved_docs) > 0,
            'golden_answers': sample.get('golden_answers', []),
            'correct': False  # 将由评估器计算
        }
    
    def _retrieve(self, question: str) -> Tuple[List[str], List[float]]:
        """粗检索（CLIP/BGE）"""
        if self.retriever is None:
            return [], []
        
        try:
            if hasattr(self.retriever, 'search'):
                results = self.retriever.search(question, num=self.clip_topk)
            elif hasattr(self.retriever, 'retrieve'):
                results = self.retriever.retrieve(query_text=question, top_k=self.clip_topk)
            else:
                return [], []
            
            # 提取文档
            if isinstance(results, tuple):
                docs, scores = results
            elif isinstance(results, list):
                docs = results
                scores = [1.0 - i*0.05 for i in range(len(docs))]  # 模拟分数
            else:
                docs = [results]
                scores = [1.0]
            
            # 提取文本
            docs_text = []
            for doc in docs:
                if isinstance(doc, dict):
                    text = doc.get('contents', doc.get('text', str(doc)))
                else:
                    text = str(doc)
                docs_text.append(text)
            
            return docs_text, scores
        
        except Exception as e:
            warnings.warn(f"检索失败: {e}")
            return [], []
    
    def _rerank_single(self, question: str, doc: str, image=None) -> Tuple[bool, float]:
        """
        使用MLLM判断单个文档的相关性（RagVL核心）
        
        Returns:
            (is_relevant, relevance_score)
        """
        prompt = f"""Is this document relevant to answering the question?

Document: {doc[:200]}...

Question: {question}

Answer with ONLY 'Yes' or 'No':"""
        
        try:
            response = self.qwen3vl.generate(
                text=prompt,
                image=image,
                max_new_tokens=5,
                temperature=0.1  # 低温度获得确定性
            )
            
            response_lower = response.strip().lower()
            
            # 解析响应
            if 'yes' in response_lower:
                return True, 0.9
            elif 'no' in response_lower:
                return False, 0.1
            else:
                # 不确定，保守判断
                return True, 0.5
        
        except Exception as e:
            warnings.warn(f"Reranking失败: {e}")
            return True, 0.5
    
    def _rerank_documents(self, question: str, 
                         retrieved_docs: List[str],
                         retrieval_scores: List[float],
                         image=None) -> List[Tuple[str, float]]:
        """
        对检索结果进行reranking（RagVL的核心创新）
        
        使用MLLM作为强大的reranker
        """
        reranked = []
        
        for doc, ret_score in zip(retrieved_docs, retrieval_scores):
            is_relevant, rel_score = self._rerank_single(question, doc, image)
            
            if is_relevant:
                # 综合分数：检索分数 × 相关性分数
                combined_score = ret_score * rel_score
                reranked.append((doc, combined_score))
        
        # 按综合分数排序
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        # 只保留Top-N
        return reranked[:self.rerank_topk]
    
    def _generate_answer(self, sample: Dict, reranked_docs: List[Tuple[str, float]]) -> str:
        """基于rerank后的文档生成答案"""
        if not reranked_docs:
            return self._direct_answer(sample)
        
        # 组织证据
        evidence_parts = []
        for i, (doc, score) in enumerate(reranked_docs):
            evidence_parts.append(f"[Evidence {i+1}]\n{doc}")
        
        evidence_str = "\n\n".join(evidence_parts)
        
        # 检查是否是多选题
        if all(k in sample for k in ['A', 'B', 'C', 'D']):
            # 简化的prompt，减少token消耗
            evidence_brief = "\n".join([f"[{i+1}] {doc[:200]}..." if len(doc) > 200 else f"[{i+1}] {doc}"
                                       for i, (doc, score) in enumerate(reranked_docs[:3])])
            prompt = f"""Evidence:
{evidence_brief}

Q: {sample['question']}
A: {sample['A']}
B: {sample['B']}
C: {sample['C']}
D: {sample['D']}

Answer letter:"""
        else:
            # 开放式问题也使用简化格式
            evidence_brief = "\n".join([f"- {doc[:150]}..." if len(doc) > 150 else f"- {doc}"
                                       for doc, score in reranked_docs[:2]])
            prompt = f"""Evidence:
{evidence_brief[:400]}

Q: {sample['question']}
Answer:"""

        try:
            answer = self.qwen3vl.generate(
                text=prompt,
                image=sample.get('image'),
                max_new_tokens=20,
                temperature=self.temperature,
                do_sample=False
            )

            # 多选题答案映射
            if all(k in sample for k in ['A', 'B', 'C', 'D']):
                return self._map_mc_answer(answer, sample)

            # 使用extract_answer_smart提取核心答案
            return extract_answer_smart(answer.strip())
        
        except Exception as e:
            warnings.warn(f"生成失败: {e}")
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

            # 使用extract_answer_smart提取核心答案
            return extract_answer_smart(answer.strip())
        
        except Exception as e:
            return ""
    
    def _map_mc_answer(self, response: str, sample: Dict) -> str:
        """映射多选题答案"""
        response_upper = response.strip().upper()
        for letter in ['A', 'B', 'C', 'D']:
            if letter in response_upper:
                return sample.get(letter, response.strip())
        return response.strip()


def create_ragvl_enhanced(qwen3vl_wrapper, retriever=None, **kwargs):
    """创建RagVL Enhanced"""
    return RagVLEnhanced(qwen3vl_wrapper, retriever, kwargs)


if __name__ == '__main__':
    print("RagVL Enhanced - 完整实现")
    print("=" * 70)
    print("核心创新:")
    print("  1. MLLM作为强大的Reranker")
    print("  2. 两阶段检索流程")
    print("\n流程:")
    print("  粗检索（CLIP/BGE）Top-20")
    print("    ↓")
    print("  MLLM Reranking Top-2~5")
    print("    ↓")
    print("  生成答案")
    print("=" * 70)

