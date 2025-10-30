#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Self-RAG Enhanced - 完整实现（基于Qwen3-VL）

核心特色:
1. 自适应检索判断 ([Retrieval] token模拟)
2. 文档相关性判断 ([IsREL] token模拟)
3. 答案支持度判断 ([IsSUP] token模拟)

通过3步判断实现真正的自适应RAG
"""

import sys
sys.path.insert(0, '/root/autodl-tmp/FlashRAG')

from typing import Dict, Any, List
import warnings


class SelfRAGEnhanced:
    """
    Self-RAG Enhanced完整实现
    
    3步判断流程:
    1. 判断是否需要检索 (模拟 [Retrieval] token)
    2. 判断文档是否相关 (模拟 [IsREL] token)
    3. 判断答案是否有支持 (模拟 [IsSUP] token)
    """
    
    def __init__(self, qwen3vl_wrapper, retriever=None, config=None):
        """
        初始化Self-RAG Enhanced
        
        Args:
            qwen3vl_wrapper: Qwen3-VL封装器
            retriever: 检索器
            config: 配置
        """
        self.qwen3vl = qwen3vl_wrapper
        self.retriever = retriever
        self.config = config or {}
        
        self.top_k = self.config.get('retrieval_topk', 5)
        self.temperature = self.config.get('temperature', 0.01)
        
        # 判断温度（低温度获得更确定的判断）
        self.decision_temp = 0.05
    
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
        
        # === Step 1: Retrieval Decision (模拟 [Retrieval] token) ===
        need_retrieval, ret_confidence = self._retrieval_decision(question, image)
        
        if not need_retrieval:
            # 无需检索，直接回答
            answer = self._direct_answer(sample)
            return {
                'question': question,
                'answer': answer,
                'retrieval_decision': 'No Retrieval',
                'retrieval_triggered': False,
                'relevant_docs': 0,
                'supported': 'N/A'
            }
        
        # === Step 2: Retrieve Documents ===
        docs = self._retrieve_documents(question)
        
        if not docs:
            answer = self._direct_answer(sample)
            return {
                'question': question,
                'answer': answer,
                'retrieval_decision': 'Retrieval (no docs)',
                'retrieval_triggered': True,
                'relevant_docs': 0,
                'supported': 'N/A'
            }
        
        # === Step 3: Relevance Judgment (模拟 [IsREL] token) ===
        relevant_docs = []
        for doc in docs:
            is_relevant, rel_conf = self._relevance_judgment(question, doc, image)
            if is_relevant:
                relevant_docs.append(doc)
        
        if not relevant_docs:
            # 无相关文档，降级到直接回答
            answer = self._direct_answer(sample)
            return {
                'question': question,
                'answer': answer,
                'retrieval_decision': 'Retrieval (no relevant)',
                'retrieval_triggered': True,
                'relevant_docs': 0,
                'supported': 'No'
            }
        
        # === Step 4: Generate Answer ===
        answer = self._generate_with_context(sample, relevant_docs[:3])
        
        # === Step 5: Support Judgment (模拟 [IsSUP] token) ===
        is_supported, sup_conf = self._support_judgment(
            question, answer, relevant_docs[:3]
        )
        
        if not is_supported:
            # 答案不被支持，降级到直接回答
            answer = self._direct_answer(sample)
            supported_status = 'Not Supported (degraded)'
        else:
            supported_status = 'Supported'
        
        return {
            'question': question,
            'answer': answer,
            'retrieval_decision': 'Retrieval',
            'retrieval_triggered': True,
            'relevant_docs': len(relevant_docs),
            'supported': supported_status
        }
    
    # ========================================================================
    # Step 1: Retrieval Decision
    # ========================================================================
    
    def _retrieval_decision(self, question: str, image=None) -> tuple:
        """
        判断是否需要检索外部知识
        模拟Self-RAG的 [Retrieval] / [No Retrieval] token
        
        Returns:
            (need_retrieval: bool, confidence: float)
        """
        prompt = f"""Task: Decide if external knowledge is needed to answer this question.

Question: {question}

Think carefully: Can this question be answered just by looking at the image, or does it require external factual knowledge (like dates, names, locations, historical facts)?

Answer with ONLY ONE WORD - either 'NEED' or 'NO':"""
        
        try:
            response = self.qwen3vl.generate(
                text=prompt,
                image=image,
                max_new_tokens=5,
                temperature=self.decision_temp,
                do_sample=False
            )
            
            response_clean = response.strip().upper()
            
            if 'NEED' in response_clean and 'NO' not in response_clean:
                return (True, 0.9)
            elif 'NO' in response_clean:
                return (False, 0.9)
            else:
                # 默认：检索（保守策略）
                return (True, 0.5)
        
        except Exception as e:
            warnings.warn(f"Retrieval decision失败: {e}")
            return (True, 0.5)
    
    # ========================================================================
    # Step 2: Document Retrieval
    # ========================================================================
    
    def _retrieve_documents(self, question: str) -> List[str]:
        """检索文档"""
        if self.retriever is None:
            return []
        
        try:
            # 检查retriever类型
            if hasattr(self.retriever, 'search'):
                # DenseRetriever
                results = self.retriever.search(question, num=self.top_k)
            elif hasattr(self.retriever, 'retrieve'):
                # 其他retriever
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
    # Step 3: Relevance Judgment
    # ========================================================================
    
    def _relevance_judgment(self, question: str, document: str, image=None) -> tuple:
        """
        判断文档是否与问题相关
        模拟Self-RAG的 [IsREL] / [NoREL] token
        
        Returns:
            (is_relevant: bool, confidence: float)
        """
        # 截断过长文档
        doc_preview = document[:300] + "..." if len(document) > 300 else document
        
        prompt = f"""Task: Judge if this document is relevant to answering the question.

Question: {question}

Document: {doc_preview}

Is this document relevant and helpful for answering the question?
Answer with ONLY 'RELEVANT' or 'IRRELEVANT':"""
        
        try:
            response = self.qwen3vl.generate(
                text=prompt,
                image=None,  # 不使用图像（纯文本判断）
                max_new_tokens=5,
                temperature=self.decision_temp,
                do_sample=False
            )
            
            response_clean = response.strip().upper()
            
            if 'RELEVANT' in response_clean and 'IRRELEVANT' not in response_clean:
                return (True, 0.9)
            elif 'IRRELEVANT' in response_clean:
                return (False, 0.9)
            else:
                # 默认：相关（保守）
                return (True, 0.6)
        
        except Exception as e:
            warnings.warn(f"Relevance judgment失败: {e}")
            return (True, 0.5)
    
    # ========================================================================
    # Step 4: Answer Generation
    # ========================================================================
    
    def _generate_with_context(self, sample: Dict, relevant_docs: List[str]) -> str:
        """基于相关文档生成答案"""
        context = "\n\n".join(relevant_docs)
        
        # 检查是否是多选题
        if all(k in sample for k in ['A', 'B', 'C', 'D']):
            prompt = f"""Based on the context below, answer the question.

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
            prompt = f"""Based on the context below, answer the question briefly.

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
    
    # ========================================================================
    # Step 5: Support Judgment
    # ========================================================================
    
    def _support_judgment(self, question: str, answer: str, documents: List[str]) -> tuple:
        """
        判断答案是否被文档支持
        模拟Self-RAG的 [IsSUP] / [NoSUP] token
        
        Returns:
            (is_supported: bool, confidence: float)
        """
        context = "\n\n".join(documents)[:400]
        
        prompt = f"""Task: Verify if the answer is supported by the context.

Context: {context}...

Question: {question}
Answer: {answer}

Is the answer fully supported by (derivable from) the context?
Answer with ONLY 'SUPPORTED' or 'NOT_SUPPORTED':"""
        
        try:
            response = self.qwen3vl.generate(
                text=prompt,
                image=None,
                max_new_tokens=5,
                temperature=self.decision_temp,
                do_sample=False
            )
            
            response_clean = response.strip().upper().replace(' ', '_')
            
            if 'SUPPORTED' in response_clean and 'NOT' not in response_clean:
                return (True, 0.9)
            elif 'NOT' in response_clean or 'UNSUPPORTED' in response_clean:
                return (False, 0.9)
            else:
                # 默认：支持（保守）
                return (True, 0.6)
        
        except Exception as e:
            warnings.warn(f"Support judgment失败: {e}")
            return (True, 0.5)
    
    # ========================================================================
    # Fallback: Direct Answer
    # ========================================================================
    
    def _direct_answer(self, sample: Dict) -> str:
        """直接回答（无检索）"""
        if all(k in sample for k in ['A', 'B', 'C', 'D']):
            prompt = f"""Answer this question based on the image.

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
        
        # 提取字母
        for letter in ['A', 'B', 'C', 'D']:
            if letter in response_upper:
                # 返回对应的完整答案
                return sample.get(letter, response.strip())
        
        return response.strip()


def create_selfrag_enhanced(qwen3vl_wrapper, retriever=None, **kwargs):
    """创建Self-RAG Enhanced"""
    return SelfRAGEnhanced(qwen3vl_wrapper, retriever, kwargs)


if __name__ == '__main__':
    print("Self-RAG Enhanced - 完整实现")
    print("=" * 70)
    print("核心特色:")
    print("  1. 自适应检索判断 ([Retrieval] token模拟)")
    print("  2. 文档相关性判断 ([IsREL] token模拟)")
    print("  3. 答案支持度判断 ([IsSUP] token模拟)")
    print("=" * 70)
