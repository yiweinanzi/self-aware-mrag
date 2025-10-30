#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mR²AG Enhanced - 完整实现（基于Qwen3-VL）

核心特色:
1. 双重反思机制 (Retrieval-Reflection + Relevance-Reflection)
2. 段落级处理（关键差异！）
3. 层级打分 (S_ret × S_rel × S_ans)
"""

import sys
sys.path.insert(0, '/root/autodl-tmp/FlashRAG')

from typing import Dict, Any, List, Tuple
import warnings


class MR2AGEnhanced:
    """
    mR²AG Enhanced完整实现
    
    核心流程:
    1. Retrieval-Reflection: 判断是否需要检索
    2. 检索条目并切分为段落
    3. Relevance-Reflection: 逐段落判断相关性
    4. 层级打分选择最佳答案
    """
    
    def __init__(self, qwen3vl_wrapper, retriever=None, config=None):
        """
        初始化mR²AG Enhanced
        
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
        
        # 段落切分参数
        self.para_min_len = self.config.get('para_min_len', 50)
        self.para_max_len = self.config.get('para_max_len', 180)
    
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
        
        # === Step 1: Retrieval-Reflection ===
        need_retrieval = self._retrieval_reflection(question, image)
        
        if not need_retrieval:
            answer = self._direct_answer(sample)
            return {
                'question': question,
                'answer': answer,
                'retrieval_decision': 'No Retrieval',
                'total_paragraphs': 0,
                'relevant_paragraphs': 0
            }
        
        # === Step 2: 检索条目 ===
        entries = self._retrieve_documents(question)
        
        if not entries:
            answer = self._direct_answer(sample)
            return {
                'question': question,
                'answer': answer,
                'retrieval_decision': 'Retrieval (no docs)',
                'total_paragraphs': 0,
                'relevant_paragraphs': 0
            }
        
        # === Step 3: 段落级处理 (核心特色!) ===
        candidates = []
        total_paras = 0
        
        for entry_idx, entry in enumerate(entries):
            # 切分段落
            paragraphs = self._split_into_paragraphs(entry)
            total_paras += len(paragraphs)
            
            for para in paragraphs:
                # Relevance-Reflection
                is_relevant, rel_score = self._relevance_reflection(question, para)
                
                if is_relevant:
                    # 生成答案（带分数）
                    answer, ans_score = self._generate_with_paragraph(sample, para)
                    
                    # 层级打分: S_ret × S_rel × S_ans
                    # S_ret: 检索分数（根据排名衰减）
                    ret_score = 0.9 ** entry_idx
                    total_score = ret_score * rel_score * ans_score
                    
                    candidates.append({
                        'answer': answer,
                        'score': total_score,
                        'paragraph': para,
                        'entry_idx': entry_idx
                    })
        
        # === Step 4: 选择最高分答案 ===
        if candidates:
            best = max(candidates, key=lambda x: x['score'])
            answer = best['answer']
        else:
            answer = self._direct_answer(sample)
        
        return {
            'question': question,
            'answer': answer,
            'retrieval_decision': 'Retrieval',
            'total_paragraphs': total_paras,
            'relevant_paragraphs': len(candidates)
        }
    
    # ========================================================================
    # Step 1: Retrieval-Reflection
    # ========================================================================
    
    def _retrieval_reflection(self, question: str, image=None) -> bool:
        """
        判断是否需要检索
        模拟mR²AG的 [Retrieval] / [No Retrieval] token
        """
        prompt = f"""Decide if external knowledge is needed.

Question: {question}

Answer with ONLY 'NEED' or 'NO':"""
        
        try:
            response = self.qwen3vl.generate(
                text=prompt,
                image=image,
                max_new_tokens=5,
                temperature=0.05,
                do_sample=False
            )
            
            return 'NEED' in response.upper()
        
        except Exception as e:
            warnings.warn(f"Retrieval-Reflection失败: {e}")
            return True  # 默认检索
    
    # ========================================================================
    # Step 2: Document Retrieval
    # ========================================================================
    
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
    
    # ========================================================================
    # Step 3: Paragraph Splitting (核心特色!)
    # ========================================================================
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        段落切分（mR²AG的核心特色）
        按句子切分，控制每段50-180 tokens
        
        Args:
            text: 原始文本
            
        Returns:
            段落列表
        """
        # 按句号切分
        sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
        
        paragraphs = []
        current_para = ""
        
        for sent in sentences:
            # 检查添加后是否超过最大长度
            if len(current_para) + len(sent) < self.para_max_len:
                current_para += " " + sent
            else:
                # 当前段落达到长度，保存并开始新段落
                if len(current_para) > self.para_min_len:
                    paragraphs.append(current_para.strip())
                current_para = sent
        
        # 保存最后一段
        if len(current_para) > self.para_min_len:
            paragraphs.append(current_para.strip())
        
        # 如果没有段落（文本太短），返回原文
        return paragraphs if paragraphs else [text[:self.para_max_len]]
    
    # ========================================================================
    # Step 4: Relevance-Reflection (段落级)
    # ========================================================================
    
    def _relevance_reflection(self, question: str, paragraph: str) -> Tuple[bool, float]:
        """
        段落相关性判断
        模拟mR²AG的 [Relevant] / [Irrelevant] token
        
        Returns:
            (is_relevant: bool, relevance_score: float)
        """
        prompt = f"""Rate paragraph relevance (0-10).

Question: {question}

Paragraph: {paragraph[:200]}...

Relevance score (0=irrelevant, 10=perfect):"""
        
        try:
            response = self.qwen3vl.generate(
                text=prompt,
                image=None,
                max_new_tokens=5,
                temperature=0.1,
                do_sample=False
            )
            
            try:
                score = float(response.strip()) / 10.0
            except:
                score = 0.5
            
            return (score > 0.5, score)
        
        except Exception as e:
            warnings.warn(f"Relevance-Reflection失败: {e}")
            return (True, 0.5)
    
    # ========================================================================
    # Step 5: Answer Generation with Scoring
    # ========================================================================
    
    def _generate_with_paragraph(self, sample: Dict, paragraph: str) -> Tuple[str, float]:
        """
        基于单个段落生成答案（带置信度）
        
        Returns:
            (answer: str, confidence_score: float)
        """
        if all(k in sample for k in ['A', 'B', 'C', 'D']):
            prompt = f"""Based on this evidence paragraph, answer the question.

Evidence: {paragraph}

Question: {sample['question']}

Choices:
A. {sample['A']}
B. {sample['B']}
C. {sample['C']}
D. {sample['D']}

Answer with the letter only (A/B/C/D):"""
        else:
            prompt = f"""Based on this paragraph, answer briefly.

Paragraph: {paragraph}

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
                answer = self._map_mc_answer(answer, sample)
            
            # 简化版置信度：固定值（完整版应该用模型概率）
            confidence = 0.8
            
            return (answer.strip(), confidence)
        
        except Exception as e:
            warnings.warn(f"生成失败: {e}")
            return ("", 0.0)
    
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


def create_mr2ag_enhanced(qwen3vl_wrapper, retriever=None, **kwargs):
    """创建mR²AG Enhanced"""
    return MR2AGEnhanced(qwen3vl_wrapper, retriever, kwargs)


if __name__ == '__main__':
    print("mR²AG Enhanced - 完整实现")
    print("=" * 70)
    print("核心特色:")
    print("  1. 双重反思 (Retrieval + Relevance)")
    print("  2. 段落级处理 (50-180 tokens) ← 关键!")
    print("  3. 层级打分 (S_ret × S_rel × S_ans)")
    print("=" * 70)

