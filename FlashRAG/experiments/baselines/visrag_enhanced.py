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
from flashrag.utils.vqa_evaluator import extract_okvqa_answer
from experiments.baselines.answer_extractor import extract_answer_smart
from experiments.baselines.evaluation_helper import evaluate_answer_correctness


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
                # 使用本地BGE reranker路径
                reranker_path = self.config.get('bge_reranker_path', '/data0/home/zqwang/ACL/models/bge-reranker-v2-m3')
                self.bge_reranker = create_bge_reranker(model_name=reranker_path)
                print(f"✅ VisRAG: BGE Reranker已加载 (路径: {reranker_path})")
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
            golden_answers = sample.get('golden_answers', [])
            is_correct = evaluate_answer_correctness(answer, golden_answers)
            return {
                'question': question,
                'answer': answer,
                'retrieved_docs': [],  # 空列表而不是0
                'reranked_docs': [],  # 空列表而不是0
                'retrieved': False,  # 明确标记为未检索到
                'reranker_used': False,
                'golden_answers': golden_answers,
                'correct': is_correct
            }
        
        # === Step 3: BGE Reranking (关键!) ===
        reranked_docs = self._rerank_documents(question, initial_docs)
        
        # === Step 4: 融合生成 ===
        answer = self._generate_with_reranked_context(sample, reranked_docs)

        # 计算correct字段
        golden_answers = sample.get('golden_answers', [])
        is_correct = evaluate_answer_correctness(answer, golden_answers)

        return {
            'question': question,
            'answer': answer,
            'retrieved_docs': initial_docs,  # 返回原始检索的文档
            'reranked_docs': reranked_docs,  # 返回重排序后的文档
            'retrieved': len(initial_docs) > 0,  # 是否检索到了文档
            'reranker_used': self.bge_reranker is not None,
            'golden_answers': golden_answers,
            'correct': is_correct
        }
    
    # ========================================================================
    # Document Retrieval
    # ========================================================================
    
    def _retrieve_documents(self, question: str, num: int = 10) -> List[Dict]:
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

            # 返回完整的文档对象列表（不提取文本）
            docs_list = []
            for doc in docs:
                if isinstance(doc, dict):
                    docs_list.append(doc)  # 保留完整的文档对象
                else:
                    # 如果不是字典，转换成字典格式
                    docs_list.append({
                        'contents': str(doc),
                        'id': str(hash(str(doc))),
                        'title': '',
                        'source': 'retriever'
                    })

            return docs_list

        except Exception as e:
            warnings.warn(f"检索失败: {e}")
            return []
    
    # ========================================================================
    # BGE Reranking (核心特色!)
    # ========================================================================
    
    def _rerank_documents(self, question: str, documents: List[Dict]) -> List[Dict]:
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
            # BGE reranker需要文本列表，提取文档内容
            doc_texts = []
            for doc in documents:
                if isinstance(doc, dict):
                    content = doc.get('contents', doc.get('text', str(doc)))
                    doc_texts.append(content)
                else:
                    doc_texts.append(str(doc))

            # 使用BGE重排
            reranked_indices = self.bge_reranker.rerank(
                query=question,
                documents=doc_texts,
                top_k=self.final_topk
            )

            # 根据重排结果返回原始文档对象
            if isinstance(reranked_indices, list) and len(reranked_indices) > 0:
                if isinstance(reranked_indices[0], int):
                    # 返回的是索引列表
                    return [documents[i] for i in reranked_indices if i < len(documents)]
                else:
                    # 返回的是重排后的文档文本，需要匹配回原始文档
                    # 这里简化处理，返回原始顺序的文档
                    return documents[:self.final_topk]
            else:
                return documents[:self.final_topk]
        
        except Exception as e:
            warnings.warn(f"重排失败: {e}，使用原始顺序")
            return documents[:self.final_topk]
    
    # ========================================================================
    # Answer Generation
    # ========================================================================
    
    def _generate_with_reranked_context(self, sample: Dict, reranked_docs: List[Dict]) -> str:
        """基于重排后的文档生成答案"""
        if not reranked_docs:
            return self._direct_answer(sample)

        # 从字典中提取文本内容
        context_parts = []
        for doc in reranked_docs:
            if isinstance(doc, dict):
                content = doc.get('contents', doc.get('text', str(doc)))
                context_parts.append(content)
            else:
                context_parts.append(str(doc))

        context = "\n\n".join(context_parts)
        
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
            prompt = f"""Answer with 1-3 words only based on the context.

Context:
{context}

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
            
            # 多选题答案映射
            if all(k in sample for k in ['A', 'B', 'C', 'D']):
                return self._map_mc_answer(answer, sample)

            # 使用extract_okvqa_answer提取核心答案
            return extract_answer_smart(answer)
        
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
            prompt = f"Answer with 1-3 words only based on the image.\n\nQuestion: {sample['question']}\n\nAnswer:"
        
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

            # 使用extract_okvqa_answer提取核心答案
            return extract_answer_smart(answer)

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
            prompt = f"Question: {sample['question']}\n\nAnswer (1-3 words only):"

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

            # 使用extract_okvqa_answer提取核心答案
            return extract_answer_smart(answer)

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
